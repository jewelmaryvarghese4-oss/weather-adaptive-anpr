"""Downscale the new full-size outputs to the 1536 px scoring size: ~/dehaze-eval/out/<name>/<photo>.png (only the photos that carry verified plates)."""
import json, os
from PIL import Image
D = "/mnt/c/projects/jewwww/datasets/"
H = os.path.expanduser("~/dehaze-eval")
SRC = {"new_long": "foggy_photos_uvm_real_long", "blend_long": "foggy_photos_uvm_hybrid_long_blend", "phys_long_normal": "foggy_photos_phys_long_blend",
       "phys_long_se": "foggy_photos_phys_long_blend_strong_exp"}
photos = sorted({it["photo"] for it in json.load(open(H + "/data/items.json"))})
for name, folder in SRC.items():
    os.makedirs("%s/out/%s" % (H, name), exist_ok=True)
    for p in photos:
        ref = Image.open("%s/data/in1536/%s.png" % (H, p))
        im = Image.open("%s%s/%s_restored.png" % (D, folder, p)).convert("RGB").resize(ref.size, Image.LANCZOS)
        im.save("%s/out/%s/%s.png" % (H, name, p))
    print(name, len(photos), flush=True)
