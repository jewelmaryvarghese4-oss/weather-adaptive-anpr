import cv2
from pathlib import Path

folder = r"C:\Users\Jewel\OneDrive\Desktop\jewwww\evaluation\baseline_ocr\clean_crops"
threshold = 600

print(f"Checking folder: {folder}")
print(f"Exists: {Path(folder).exists()}")

if not Path(folder).exists():
    print("STOPPING: source folder not found.")
else:
    files = list(Path(folder).iterdir())
    print(f"Files found in source folder: {len(files)}")

    sharp_dir = Path(folder).parent / "clean_crops_sharp"
    sharp_dir.mkdir(exist_ok=True)
    print(f"Output folder created at: {sharp_dir}")

    kept, dropped = 0, 0
    for img_path in files:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  could not read: {img_path.name}")
            dropped += 1
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        variance = cv2.Laplacian(gray, cv2.CV_64F).var()

        if variance >= threshold:
            cv2.imwrite(str(sharp_dir / img_path.name), img)
            kept += 1
        else:
            dropped += 1

    print(f"Kept {kept} images, dropped {dropped} below threshold {threshold}")