"""three-way sheet: untouched | seamless, NO tint fix | seamless, tint fix 0.8 | old tiled (for reference)"""
import cv2, glob, os, numpy as np
D = "C:/projects/jewwww/datasets/"
stems = ["20260926_112044", "20260926_112303", "20260926_112036", "20260926_110958", "20260926_112113", "20260926_112242"]
rows = []
for st in stems:
    ims = []
    for p in (D + "foggy_photos_upright/%s.jpg" % st, D + "foggy_photos_uvm_gopro_clean_notint/%s_restored.png" % st, D + "foggy_photos_uvm_gopro_clean/%s_restored.png" % st):
        im = cv2.imread(p)
        if im is None: break
        ims.append(cv2.resize(im, (520, int(520 * im.shape[0] / im.shape[1])), interpolation=cv2.INTER_AREA))
    if len(ims) < 3: continue
    for x, n in zip(ims, ("untouched", "seamless, NO tint fix", "seamless + tint fix")):
        cv2.putText(x, st[9:] + " " + n, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
    h = min(x.shape[0] for x in ims)
    rows.append(np.hstack([x[:h] for x in ims]))
cv2.imwrite(D + "foggy_eval/clean_check/tint_three_way.jpg", np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
print("sheet rows:", len(rows))
