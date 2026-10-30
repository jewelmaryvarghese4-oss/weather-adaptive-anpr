import cv2
import os
from pathlib import Path

folder = r"C:\Users\Jewel\OneDrive\Desktop\jewwww\evaluation\baseline_ocr\clean_crops"
scores = []

for img_path in Path(folder).iterdir():
    img = cv2.imread(str(img_path))
    if img is None:
        continue
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    scores.append((img_path.name, variance))

scores.sort(key=lambda x: x[1])
for name, v in scores:
    print(f"{v:8.1f}  {name}")
    