import easyocr
import os
import csv

reader = easyocr.Reader(['en'])
allowlist = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'

hazy_dir = 'dataset/hazy'
dehazed_dir = 'dehazed_test_v3'
clear_dir = 'dataset/clear'

results = []
hazy_confs, dehazed_confs, clear_confs = [], [], []

def read_text_conf(path):
    if not os.path.exists(path):
        return "", 0.0
    out = reader.readtext(path, allowlist=allowlist)
    if not out:
        return "", 0.0
    return out[0][1], float(out[0][2])

hazy_files = sorted(os.listdir(hazy_dir))
print(f"Found {len(hazy_files)} hazy images. Running OCR on all three sets...\n")

for i, fname in enumerate(hazy_files):
    hazy_path = os.path.join(hazy_dir, fname)
    dehazed_path = os.path.join(dehazed_dir, fname)

    # ground truth clear image: strip the _vN suffix
    base_name = fname.rsplit("_v", 1)[0]
    clear_path = os.path.join(clear_dir, base_name + ".jpg")

    hazy_text, hazy_conf = read_text_conf(hazy_path)
    dehazed_text, dehazed_conf = read_text_conf(dehazed_path)
    clear_text, clear_conf = read_text_conf(clear_path)

    hazy_confs.append(hazy_conf)
    dehazed_confs.append(dehazed_conf)
    clear_confs.append(clear_conf)

    results.append([fname, hazy_text, hazy_conf, dehazed_text, dehazed_conf, clear_text, clear_conf])

    if i % 50 == 0:
        print(f"  processed {i}/{len(hazy_files)}...")

# Save full results to CSV for inspection
with open("ocr_batch_results.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "hazy_text", "hazy_conf", "dehazed_text", "dehazed_conf", "clear_text", "clear_conf"])
    writer.writerows(results)

def avg(lst):
    return sum(lst) / len(lst) if lst else 0.0

print("\n=== SUMMARY ===")
print(f"Samples processed:        {len(results)}")
print(f"Average HAZY confidence:    {avg(hazy_confs):.4f}")
print(f"Average DEHAZED confidence: {avg(dehazed_confs):.4f}")
print(f"Average CLEAR (GT) confidence: {avg(clear_confs):.4f}")
print(f"\nDehazed vs Hazy delta: {avg(dehazed_confs) - avg(hazy_confs):+.4f}")
print(f"Full per-image results saved to ocr_batch_results.csv")