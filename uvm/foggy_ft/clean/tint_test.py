"""tint strength comparison on the dense-fog photos from the earlier sheet: current | beta 0.5 | beta 0.8 | beta 1.0 (applied on the existing full-size result)"""
import os, sys
import cv2, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tint_fix import fix_tint
D = "C:/projects/jewwww/datasets/"
stems = ["20260926_112044", "20260926_112303", "20260926_111526", "20260926_113238"]
rows = []
for st in stems:
    hz = cv2.imread(D + "foggy_photos_upright/%s.jpg" % st); cur = cv2.imread(D + "foggy_photos_uvm_gopro_gf/%s_restored.png" % st)
    if hz is None or cur is None: print("missing", st); continue
    ims = [cur] + [fix_tint(hz, cur, b) for b in (0.5, 0.8, 1.0)]
    sm = [cv2.resize(i, (500, int(500 * i.shape[0] / i.shape[1])), interpolation=cv2.INTER_AREA) for i in ims]
    h = min(x.shape[0] for x in sm)
    for x, n in zip(sm, ("current", "beta 0.5", "beta 0.8", "beta 1.0")):
        cv2.putText(x, n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
    rows.append(np.hstack([x[:h] for x in sm]))
w = min(r.shape[1] for r in rows)
cv2.imwrite(D + "foggy_eval/tint_test.jpg", np.vstack([r[:, :w] for r in rows]), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("saved")
