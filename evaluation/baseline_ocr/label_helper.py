import os
import glob
import json

CROPS_DIR = "clean_crops"
OUTPUT_FILE = "ground_truth.json"


def load_existing_labels():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, "r") as f:
            return json.load(f)
    return {}


def save_labels(labels):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(labels, f, indent=2)


def main():
    crop_paths = sorted(glob.glob(os.path.join(CROPS_DIR, "*.jpg")))
    if not crop_paths:
        print(f"No crops found in {CROPS_DIR}")
        return

    labels = load_existing_labels()
    already_done = len(labels)
    total = len(crop_paths)

    print(f"Found {total} crops. {already_done} already labeled.")
    print("For each image that opens: read the plate, type it, press Enter.")
    print("Type 's' to skip a crop, 'q' to save and quit at any point.\n")

    for path in crop_paths:
        name = os.path.basename(path).rsplit(".", 1)[0]

        if name in labels:
            continue  # already labeled in a previous run

        # Opens the image in your default Windows photo viewer
        os.startfile(os.path.abspath(path))

        entry = input(f"[{len(labels)+1}/{total}] {name} - plate text: ").strip()

        if entry.lower() == "q":
            print("Stopping. Progress saved.")
            break
        elif entry.lower() == "s":
            continue
        elif entry:
            labels[name] = entry.upper().replace(" ", "")
            save_labels(labels)  # save after every entry - no data loss if interrupted

    print(f"\nDone for now. {len(labels)}/{total} crops labeled.")
    print(f"Saved to {OUTPUT_FILE}")
    print("Run this script again anytime to continue where you left off.")


if __name__ == "__main__":
    main()