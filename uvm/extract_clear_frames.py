"""
Cuts clear frames out of the training videos (dataset15/train*) for the rain-and-fog fine-tune. Test videos are never used.

    python extract_clear_frames.py --out ../datasets/rainfog_clear --per_video 80
"""
import argparse
import glob
import os

import cv2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", default=sorted(glob.glob("../datasets/dataset15/train*")))
    ap.add_argument("--out", default="../datasets/rainfog_clear")
    ap.add_argument("--per_video", type=int, default=80)
    ap.add_argument("--width", type=int, default=1920)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for v in a.videos:
        name = os.path.basename(v).split(".")[0]
        if not name.startswith("train"):
            raise SystemExit(f"refusing {v}: only train* videos may be used")
        cap = cv2.VideoCapture(v)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        step = max(1, total // a.per_video)
        n = idx = 0
        while True:
            ok = cap.grab()
            if not ok:
                break
            if idx % step == 0:
                ok, f = cap.retrieve()
                if ok:
                    s = a.width / f.shape[1]
                    if s < 1:
                        f = cv2.resize(f, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
                    cv2.imwrite(os.path.join(a.out, f"{name}_{idx:05d}.jpg"), f, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    n += 1
            idx += 1
        cap.release()
        print(f"{v}: {n} frames", flush=True)


if __name__ == "__main__":
    main()
