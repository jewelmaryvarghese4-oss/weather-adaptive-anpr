"""
Several worker processes find and read the plates of DIFFERENT frames at the same time, so a fast, heavy video (4K, a vehicle
passing in a few frames) no longer has to be thinned out by skipping frames.

    decoder (main process)        N workers (own YOLO + LPRNet, one frame at a time)        tracker (main process)
    frame i -> shared memory  ->  frame i: detect plates, read every crop            ->     results put back in frame order,
    frame i+1 -> shared memory -> frame i+1 on another worker                               then tracked / voted / announced
                                                                                             by LiveReader.update()

Workers do the heavy, stateless part (detector + reader). Tracking and voting need the frames in order, so they stay in ONE place
and take the results of the workers in frame order (a result that is early waits for the frames before it).

Frames travel in shared memory, never through pipes (a 4K frame is 25 MB). Used by live_plates.py (--workers N) and plate_app.py.
"""
import collections
import multiprocessing as mp
import queue
import threading
import time
from multiprocessing import shared_memory

import cv2
import numpy as np

CTX = mp.get_context("spawn")      # CUDA does not survive fork


def _worker_main(wid, lprnet_ckpt, yolo_ckpt, task_q, result_q):
    import torch
    from PIL import Image
    from ultralytics import YOLO

    from lprnet_vocab import NUM_CLASSES
    from model_lprnet import LPRNet
    from read_plate import read_plate

    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        half = dev.type == "cuda"
        lprnet = LPRNet(num_classes=NUM_CLASSES).to(dev)
        lprnet.load_state_dict(torch.load(lprnet_ckpt, map_location=dev))
        lprnet.eval()
        yolo = YOLO(yolo_ckpt)
        yolo.predict(np.zeros((64, 64, 3), np.uint8), verbose=False)          # warm-up
        read_plate(lprnet, Image.new("RGB", (120, 40)), dev)
        result_q.put(("ready", wid))
    except Exception as e:                                                      # report, don't die silently
        import traceback
        result_q.put(("error", f"worker {wid} could not start: {e}\n{traceback.format_exc()}"))
        return

    attached = {}                                                               # shared-memory blocks of the current run
    while True:
        task = task_q.get()
        if task is None:
            break
        shm_name, shape, n_slots, slot, idx, p = task
        try:
            if shm_name not in attached:
                for old in attached.values():
                    old.close()
                attached.clear()
                attached[shm_name] = shared_memory.SharedMemory(name=shm_name)
            frames = np.ndarray((n_slots,) + tuple(shape), dtype=np.uint8, buffer=attached[shm_name].buf)
            frame = frames[slot]
            dets = []
            for r in yolo.predict(frame, conf=p["conf"], imgsz=p["imgsz"], half=half, verbose=False):
                for b, c in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                    if b[2] - b[0] >= p["min_w"] and b[3] - b[1] >= p["min_h"]:
                        x1, y1, x2, y2 = (int(v) for v in b)
                        crop = cv2.cvtColor(frame[max(0, y1):y2, max(0, x1):x2], cv2.COLOR_BGR2RGB)
                        _, text = read_plate(lprnet, Image.fromarray(crop), dev)
                        dets.append((tuple(b), c, crop, text))
            result_q.put(("result", idx, slot, dets))
        except Exception as e:
            import traceback
            result_q.put(("error", f"worker {wid} failed on frame {idx}: {e}\n{traceback.format_exc()}"))
    for old in attached.values():
        old.close()


class WorkerPool:
    """N long-lived worker processes (models are loaded once per worker and reused for every run)."""

    def __init__(self, n, lprnet_ckpt, yolo_ckpt, startup_timeout=240):
        self.n, self.lprnet_ckpt, self.yolo_ckpt = n, lprnet_ckpt, yolo_ckpt
        self.task_q, self.result_q = CTX.Queue(), CTX.Queue()
        self.procs = [CTX.Process(target=_worker_main, args=(i, lprnet_ckpt, yolo_ckpt, self.task_q, self.result_q), daemon=True)
                      for i in range(n)]
        for p in self.procs:
            p.start()
        ready, deadline = 0, time.time() + startup_timeout
        while ready < n:
            try:
                kind, *rest = self.result_q.get(timeout=max(1, deadline - time.time()))
            except queue.Empty:
                self.close()
                raise RuntimeError("the workers did not start in time")
            if kind == "error":
                self.close()
                raise RuntimeError(rest[0])
            if kind == "ready":
                ready += 1

    def alive(self):
        return all(p.is_alive() for p in self.procs)

    def close(self):
        for _ in self.procs:
            try:
                self.task_q.put(None)
            except Exception:
                pass
        for p in self.procs:
            p.join(timeout=5)
            if p.is_alive():
                p.terminate()


def run_parallel(pool, cap, reader, args, fps, is_file=True, realtime=True, max_lag=1.0, stop=None, on_frame=None):
    """Generator. Decodes `cap`, lets the pool read the frames, feeds the results to reader.update() in frame order and yields
    (idx, frame_bgr, info) after each frame has been tracked (the frame may be drawn on; it is recycled after the next step).
    stop: a threading.Event that the caller sets to end early. on_frame(idx, frame): called just before a frame is tracked.

    realtime=True : pace a file at its own fps (like a live camera). If the workers cannot keep up, frames wait; only when the
                    result is more than max_lag seconds behind are frames skipped to catch up. Nothing is skipped otherwise.
    realtime=False: no pacing, nothing is ever skipped (as fast as the workers allow).
    """
    from PIL import Image

    ok, first = cap.read()
    if not ok:
        return
    shape = first.shape
    n_slots = pool.n * 3 + 2
    shm = shared_memory.SharedMemory(create=True, size=n_slots * int(np.prod(shape)))
    frames = np.ndarray((n_slots,) + shape, dtype=np.uint8, buffer=shm.buf)
    free = queue.Queue()
    for s in range(n_slots):
        free.put(s)
    inflight = collections.deque()          # frame indices handed to the workers, oldest first
    lock = threading.Lock()
    stats = {"skipped": 0, "decoded": 0, "done": False}
    halt = threading.Event()                    # our own stop signal (so the caller's event is only ever set by the caller)

    def stopping():
        return halt.is_set() or (stop is not None and stop.is_set())

    params = {"conf": args.yolo_conf, "imgsz": args.yolo_imgsz, "min_w": args.min_w, "min_h": args.min_h}
    t_start = time.perf_counter()

    def producer():
        idx, pending = 0, first
        while not stopping():
            if pending is not None:
                frame, pending = pending, None
            else:
                ok, frame = cap.read()
                if not ok:
                    break
            if is_file and realtime:                                   # keep to the video's own speed
                due = t_start + idx / fps
                wait = due - time.perf_counter()
                if wait > 0:
                    time.sleep(wait)
                elif -wait > max_lag:                                  # far behind: skip ahead, but only as far as needed
                    behind = int((-wait - max_lag * 0.5) * fps)
                    for _ in range(behind):
                        if not cap.grab():
                            break
                        idx += 1
                        stats["skipped"] += 1
                    idx += 1                                           # the frame read above is dropped too: the next read is idx + 1
                    stats["skipped"] += 1
                    ok, frame = cap.read()
                    if not ok:
                        break
            slot = None
            while slot is None and not stopping():
                try:
                    slot = free.get(timeout=0.2)
                except queue.Empty:
                    continue
            if slot is None:
                break
            np.copyto(frames[slot], frame)
            with lock:
                inflight.append(idx)
            pool.task_q.put((shm.name, shape, n_slots, slot, idx, params))
            stats["decoded"] += 1
            idx += 1
        stats["done"] = True

    th = threading.Thread(target=producer, daemon=True)
    th.start()
    pending = {}
    recycle = None
    try:
        while True:
            if recycle is not None:
                free.put(recycle)
                recycle = None
            with lock:
                head = inflight[0] if inflight else None
            if head is None:
                if stats["done"]:
                    with lock:
                        if not inflight:
                            break
                time.sleep(0.002)
                continue
            if head not in pending:
                try:
                    msg = pool.result_q.get(timeout=0.5)
                except queue.Empty:
                    if not pool.alive():
                        raise RuntimeError("a worker process stopped unexpectedly")
                    if stopping() and stats["done"]:
                        break
                    continue
                if msg[0] == "error":
                    raise RuntimeError(msg[1])
                if msg[0] == "result":
                    pending[msg[1]] = msg
                continue
            _, idx, slot, dets = pending.pop(head)
            with lock:
                inflight.popleft()
            frame = frames[slot]
            reads = [(Image.fromarray(d[2]), d[3]) for d in dets]
            if on_frame is not None:
                on_frame(idx, frame)
            reader.update([(d[0], d[1]) for d in dets], idx, lambda di, box: reads[di])
            yield idx, frame, {"skipped": stats["skipped"], "decoded": stats["decoded"], "backlog": len(inflight) + len(pending)}
            recycle = slot
    finally:
        stats["done"] = True
        halt.set()
        th.join(timeout=3)
        # drop anything still on its way, then release the shared memory
        t_end = time.time() + 5
        while time.time() < t_end:
            with lock:
                n_left = len(inflight)
            if n_left == 0:
                break
            try:
                msg = pool.result_q.get(timeout=0.3)
                if msg[0] == "result":
                    with lock:
                        if inflight and msg[1] in inflight:
                            inflight.remove(msg[1])
            except queue.Empty:
                pass
        del frames
        try:
            shm.close()
            shm.unlink()
        except Exception:
            pass
