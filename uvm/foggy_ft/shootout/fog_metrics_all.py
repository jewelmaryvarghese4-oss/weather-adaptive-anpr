"""Fog numbers (dark-channel haze, contrast, saturation, brightness) of every folder in ~/dehaze-eval/out on the photos that carry verified plates."""
import glob, json, os
import cv2, numpy as np
H = os.path.expanduser("~/dehaze-eval")
photos = sorted({it["photo"] for it in json.load(open(H + "/data/items.json"))})
def m(p):
    b = cv2.imread(p)
    if b is None: return None
    s = 768 / max(b.shape[:2]); b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
    f = b.astype(np.float32) / 255; g = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    fine = np.abs(g - cv2.blur(g, (3, 3))).mean()
    return cv2.erode(f.min(2), np.ones((15, 15), np.uint8)).mean(), g.std(), cv2.cvtColor(b, cv2.COLOR_BGR2HSV)[..., 1].mean() / 255, g.mean(), fine
rows = []
for d in ["raw"] + sorted(os.listdir(H + "/out")):
    base = H + "/data/in1536" if d == "raw" else H + "/out/" + d
    r = [m("%s/%s.png" % (base, p)) for p in photos]
    r = [x for x in r if x is not None]
    if len(r) >= 3:
        rows.append((d, len(r)) + tuple(np.mean(r, 0)))
print("%-26s %3s %6s %9s %6s %7s %6s" % ("folder", "n", "haze", "contrast", "sat", "bright", "grain"))
for row in sorted(rows, key=lambda x: x[2]):
    print("%-26s %3d %6.3f %9.3f %6.3f %7.3f %6.4f" % row)
