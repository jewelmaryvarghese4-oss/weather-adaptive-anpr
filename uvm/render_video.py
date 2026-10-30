"""
Smooth annotated video after a run in which frames were skipped.

While a run is live, only some frames can be read (4K, a slow computer). Saving just those frames gives a choppy picture or frozen
frames. Here the ORIGINAL video is read again, frame by frame, and every frame gets its plate boxes and labels: where a plate was
read in two frames a few frames apart, its box is moved smoothly in between (the plate moves the same way in the frames that
were skipped), and the label is the plate's final voted text. So the result looks like the live view, at the video's real speed.
"""
import bisect
import collections
import os

import cv2
import numpy as np

from async_writer import AsyncWriter

GREEN, ORANGE = (0, 220, 0), (0, 165, 255)
MAX_GAP = 20                                                  # frames; boxes are not moved across longer gaps


def _text(img, s, org, scale, color, thick):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def render_smooth(src, out_path, reader, fps, size, last_idx, progress=None, orig_width=None):
    """Writes out_path (mp4) with `size` = (width, height); reader = the LiveReader of the run (its tracks and events).
    `src` can be a smaller copy of the video (the 720p preview, which decodes far faster than 4K); `orig_width` is then the width
    of the original video, in which the boxes of the reader are given."""
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"could not reopen {src!r}")
    vw, vh = size
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    sx = vw / max(1, orig_width or src_w)                      # original-video pixels -> output pixels
    resize = (src_w, int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (vw, vh)
    k = vw / 1280.0                                            # text size relative to a 1280-wide picture
    thick = max(1, round(2 * k))

    interp, active_at = {}, collections.defaultdict(list)
    for tid, smp in reader.samples.items():
        idxs = np.array([i for i, _ in smp])
        interp[tid] = (idxs, np.array([b for _, b in smp], dtype=float))
        for i in range(int(idxs[0]), int(idxs[-1]) + 1):
            active_at[i].append(tid)
    events = sorted(reader.events, key=lambda e: e[0])
    times = [e[0] for e in events]
    th = int(34 * k)
    thumbs = {}                                                # plate crops shown next to the plate text in the panel
    for _, _, tid in events:
        c = getattr(reader.all_tracks.get(tid), "best_crop", None)
        if c is not None:
            a = cv2.cvtColor(np.asarray(c.convert("RGB")), cv2.COLOR_RGB2BGR)
            tw = min(int(150 * k), max(1, int(a.shape[1] * th / a.shape[0])))
            thumbs[tid] = cv2.resize(a, (tw, th), interpolation=cv2.INTER_AREA)

    writer = AsyncWriter(out_path, fps, size)
    try:
        for idx in range(last_idx + 1):
            ok, frame = cap.read()
            if not ok:
                break
            img = cv2.resize(frame, size, interpolation=cv2.INTER_AREA) if resize else frame
            for tid in active_at.get(idx, ()):
                idxs, boxes = interp[tid]
                pos = int(np.searchsorted(idxs, idx))
                if 0 < pos < len(idxs) and idxs[pos] != idx and idxs[pos] - idxs[pos - 1] > MAX_GAP:
                    continue                                   # plate not seen for a long while: the box would only be a guess
                x1, y1, x2, y2 = (int(np.interp(idx, idxs, boxes[:, c]) * sx) for c in range(4))
                tr = reader.all_tracks[tid]
                color = GREEN if (tr.announced or tr.reliable) else ORANGE
                cv2.rectangle(img, (x1, y1), (x2, y2), color, max(1, round(2 * k)))
                label = tr.announced or tr.text
                if label:
                    _text(img, label, (x1, max(int(22 * k), y1 - int(8 * k))), 0.8 * k, color, thick)
            # the list of the latest confirmed plates, top left, like the live view
            n = bisect.bisect_right(times, idx / fps)
            recent = events[max(0, n - 6):n][::-1]
            box_h = int((50 + 38 * len(recent)) * k)
            roi = img[0:box_h, 0:int(590 * k)]
            roi[:] = (roi * 0.5).astype(np.uint8)               # darkened panel behind the text, the road stays visible
            _text(img, f"{idx / fps:6.1f} s", (int(12 * k), int(34 * k)), 0.9 * k, (255, 255, 255), thick)
            for j, (t, plate, tid) in enumerate(recent):
                _text(img, f"{t:6.1f} s  {plate}", (int(12 * k), int((76 + 38 * j) * k)), 0.8 * k,
                      (0, 255, 0) if j == 0 else (200, 200, 200), thick)
                tb = thumbs.get(tid)
                if tb is not None:
                    y0 = int((76 + 38 * j) * k) - int(26 * k)
                    x0 = int(420 * k)
                    if y0 >= 0 and y0 + tb.shape[0] <= vh and x0 + tb.shape[1] <= vw:
                        img[y0:y0 + tb.shape[0], x0:x0 + tb.shape[1]] = tb
            writer.write(img)
            if progress is not None and idx % 30 == 0:
                progress(idx / max(1, last_idx))
    finally:
        writer.release()
        cap.release()
