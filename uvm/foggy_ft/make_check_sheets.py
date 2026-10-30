"""
Phone-friendly label-check sheets: 10 crops per sheet (2 columns x 5 rows), each crop large with its number and my reading underneath.
    python foggy_ft/make_check_sheets.py     ->  datasets/foggy_ft/check_sheets/check_01.png ...
"""
import csv
import os

from PIL import Image, ImageDraw, ImageFont

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
FT = os.path.join(D, "foggy_ft")
OUT = os.path.join(FT, "check_sheets")
os.makedirs(OUT, exist_ok=True)
for f in os.listdir(OUT):
    os.remove(os.path.join(OUT, f))
rows = list(csv.DictReader(open(os.path.join(D, "foggy_eval", "full_crop_pipeline", "readings_with_draft_truth.csv"), encoding="utf-8")))
big = ImageFont.truetype("C:/Windows/Fonts/arialbd.ttf", 38)
mid = ImageFont.truetype("C:/Windows/Fonts/consolab.ttf", 40)
sm = ImageFont.truetype("C:/Windows/Fonts/arial.ttf", 26)
CW, CH = 700, 300
per = 10
for s in range(0, len(rows), per):
    sheet = Image.new("RGB", (CW * 2, CH * 5), "white")
    d = ImageDraw.Draw(sheet)
    for j, r in enumerate(rows[s:s + per]):
        x, y = (j % 2) * CW, (j // 2) * CH
        im = Image.open(os.path.join(FT, "crops", "%03d.png" % int(r["number"]))).convert("RGB")
        k = min((CW - 40) / im.width, 150 / im.height)
        im = im.resize((int(im.width * k), int(im.height * k)), Image.LANCZOS)
        sheet.paste(im, (x + 20, y + 52))
        d.text((x + 14, y + 6), "#%s" % r["number"], fill=(200, 0, 0), font=big)
        d.text((x + 150, y + 10), "photo " + r["photo"][9:15], fill=(120, 120, 120), font=sm)
        lab = r["draft_true_plate"] or ("(not a plate?)" if r["draft_confidence"] == "not a plate" else "(I could not read it)")
        d.text((x + 14, y + 215), lab, fill=(0, 90, 0) if r["draft_confidence"] == "sure" else (170, 90, 0) if r["draft_confidence"] == "likely" else (150, 0, 0), font=mid)
        d.text((x + 14, y + 262), r["draft_confidence"] or "blank", fill=(120, 120, 120), font=sm)
        d.rectangle([x, y, x + CW - 1, y + CH - 1], outline=(200, 200, 200))
    sheet.save(os.path.join(OUT, "check_%02d.png" % (s // per + 1)))
print("sheets:", (len(rows) + per - 1) // per, "->", OUT)
