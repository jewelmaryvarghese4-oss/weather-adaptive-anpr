"""
A video writer that encodes in its own thread, so saving the annotated video does not slow the frame loop down.
write(frame, repeat=n) puts the same picture n times into the file: used to fill the gaps left by skipped frames, so the saved
video keeps the speed and length of the original. The frame must not be changed after it has been handed over.

The file is H.264 (through ffmpeg), which is small and plays smoothly everywhere. Without ffmpeg it falls back to OpenCV's mp4v,
which gives files about ten times bigger that many players stutter on.
"""
import queue
import subprocess
import threading

import cv2


def _ffmpeg_exe():
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


class AsyncWriter:
    def __init__(self, path, fps, size, maxsize=90):
        self._proc = self._w = None
        exe = _ffmpeg_exe()
        if exe:
            w, h = size
            cmd = [exe, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-r", f"{fps:.4f}",
                   "-i", "-", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2", "-an", "-c:v", "libx264", "-preset", "veryfast",
                   "-crf", "23", "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
            try:
                self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError:
                self._proc = None
        if self._proc is None:
            self._w = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
        self._size = size
        self._q = queue.Queue(maxsize=maxsize)            # a full queue makes write() wait: the encoder is the limit then
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def _put(self, frame):
        if self._proc is not None:
            try:
                self._proc.stdin.write(frame.tobytes())
            except (BrokenPipeError, OSError, ValueError):
                pass
        else:
            self._w.write(frame)

    def _run(self):
        while True:
            item = self._q.get()
            if item is None:
                return
            frame, n = item
            if (frame.shape[1], frame.shape[0]) != tuple(self._size):
                frame = cv2.resize(frame, tuple(self._size))
            for _ in range(n):
                self._put(frame)

    def isOpened(self):
        return self._proc is not None or self._w.isOpened()

    def write(self, frame, repeat=1):
        if repeat > 0:
            self._q.put((frame, repeat))

    def release(self):
        self._q.put(None)
        self._t.join()
        if self._proc is not None:
            try:
                self._proc.stdin.close()
            except OSError:
                pass
            self._proc.wait()
        else:
            self._w.release()
