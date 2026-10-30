"""
Downloads openly licensed Indian vehicle / number-plate photos from Wikimedia
Commons categories, for labeling with label_review.py and training LPRNet.

Always run with --dry_run first: it walks the categories and prints how many
usable images exist and a sample of titles, without downloading anything.

Each downloaded image keeps its licence + author in <out_dir>/attribution.csv
(CC BY / CC BY-SA require attribution), so keep that file with the images.

Only images with a free licence (CC BY, CC BY-SA, CC0, public domain) are taken.
Images are fetched as Commons' own scaled thumbnails (--width, default 1600 px)
instead of the often 10+ MB originals.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python download_commons.py --dry_run
    python download_commons.py --max_images 300
"""
import argparse
import csv
import html
import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request

API = "https://commons.wikimedia.org/w/api.php"
# Wikimedia asks scripts to identify themselves.
USER_AGENT = "LPRNet-research-script/1.0 (personal research project; python-urllib)"

DEFAULT_CATEGORIES = [
    "License plates of India",
    "Vehicles with license plates of India",
    "License plates of Kerala",
    "Vehicles with license plates of Kerala",
    "Motorcycle license plates in India",
]
FREE_LICENCE = re.compile(r"^(cc[- ]by(-sa)?\b|cc0|cc[- ]zero|public domain|pd\b)", re.I)


def api(params, retries=8):
    url = API + "?" + urllib.parse.urlencode({**params, "format": "json"})
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=60) as r:
                data = json.load(r)
            time.sleep(1.0)  # one request per second keeps us under Commons' rate limit
            return data
        except urllib.error.HTTPError as e:
            if e.code in (429, 503) and attempt < retries - 1:
                wait = int(e.headers.get("Retry-After") or 0) or min(60, 5 * 2 ** attempt)
                print(f"  rate-limited, waiting {wait}s ...", flush=True)
                time.sleep(wait)
                continue
            raise
        except urllib.error.URLError:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
                continue
            raise


def category_members(cat, kind):
    """Yields member titles of Category:<cat> of the given kind ('file' or 'subcat')."""
    params = {"action": "query", "list": "categorymembers", "cmtitle": f"Category:{cat}",
              "cmtype": kind, "cmlimit": "500"}
    while True:
        data = api(params)
        for m in data.get("query", {}).get("categorymembers", []):
            yield m["title"]
        if "continue" not in data:
            return
        params = {**params, **data["continue"]}


def collect_files(categories, depth):
    files, seen_cats = {}, set()
    queue = [(c, 0) for c in categories]
    while queue:
        cat, d = queue.pop(0)
        if cat in seen_cats:
            continue
        seen_cats.add(cat)
        n_before = len(files)
        for t in category_members(cat, "file"):
            files.setdefault(t, cat)
        print(f"  [{d}] Category:{cat}: +{len(files) - n_before} files", flush=True)
        if d < depth:
            for sub in category_members(cat, "subcat"):
                queue.append((sub.split(":", 1)[1], d + 1))
    return files


def image_info(titles, width):
    """Yields dicts for titles (batched 50 at a time) with url, thumb url, size and licence."""
    titles = list(titles)
    for i in range(0, len(titles), 50):
        data = api({"action": "query", "titles": "|".join(titles[i:i + 50]), "prop": "imageinfo",
                    "iiprop": "url|size|mime|extmetadata", "iiurlwidth": str(width)})
        for page in data.get("query", {}).get("pages", {}).values():
            info = (page.get("imageinfo") or [None])[0]
            if not info:
                continue
            meta = info.get("extmetadata", {})
            strip = lambda k: re.sub(r"<[^>]+>", "", html.unescape(meta.get(k, {}).get("value", ""))).strip()
            yield {"title": page["title"], "mime": info.get("mime", ""), "w": info.get("width", 0),
                   "h": info.get("height", 0), "url": info.get("thumburl") or info["url"],
                   "page": info.get("descriptionurl", ""), "licence": strip("LicenseShortName"),
                   "author": strip("Artist")[:120]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", action="append", help="Commons category (without 'Category:'), repeatable. "
                    "Default: a built-in list of Indian licence-plate categories.")
    ap.add_argument("--depth", type=int, default=2, help="how many levels of subcategories to follow")
    ap.add_argument("--max_images", type=int, default=300)
    ap.add_argument("--min_width", type=int, default=600, help="skip originals narrower than this")
    ap.add_argument("--width", type=int, default=1600, help="download thumbnails of this width")
    ap.add_argument("--out_dir", default="../datasets/dataset13_commons")
    ap.add_argument("--dry_run", action="store_true", help="count and list only, download nothing")
    args = ap.parse_args()

    cats = args.category or DEFAULT_CATEGORIES
    print(f"walking {len(cats)} categories (depth {args.depth}) ...")
    files = collect_files(cats, args.depth)
    print(f"{len(files)} files found; checking licences and sizes ...")

    keep, skipped = [], {"not jpeg/png": 0, "too small": 0, "licence": 0}
    for info in image_info(files, args.width):
        if info["mime"] not in ("image/jpeg", "image/png"):
            skipped["not jpeg/png"] += 1
        elif info["w"] < args.min_width:
            skipped["too small"] += 1
        elif not FREE_LICENCE.match(info["licence"]):
            skipped["licence"] += 1
        else:
            keep.append(info)
    print(f"\nusable: {len(keep)}   skipped: {skipped}")
    keep = keep[: args.max_images]

    if args.dry_run:
        for k in keep[:25]:
            print(f"  {k['title'][:70]:<70} {k['w']}x{k['h']}  {k['licence']}")
        print(f"\n(dry run) would download {len(keep)} images into {args.out_dir}")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    rows = []
    for n, k in enumerate(keep, 1):
        ext = ".png" if k["mime"] == "image/png" else ".jpg"
        name = f"cm_{n:04d}{ext}"
        path = os.path.join(args.out_dir, name)
        if not os.path.exists(path):  # resumable
            req = urllib.request.Request(k["url"], headers={"User-Agent": USER_AGENT})
            for attempt in range(5):
                try:
                    with urllib.request.urlopen(req, timeout=120) as r, open(path, "wb") as f:
                        f.write(r.read())
                    break
                except urllib.error.HTTPError as e:
                    if e.code in (429, 503) and attempt < 4:
                        time.sleep(2 ** attempt * 2)
                        continue
                    print(f"  failed {k['title']}: {e}")
                    break
            time.sleep(0.5)  # be polite to the servers
        rows.append((name, k["title"], k["author"], k["licence"], k["page"]))
        if n % 25 == 0:
            print(f"  {n}/{len(keep)} downloaded", flush=True)

    with open(os.path.join(args.out_dir, "attribution.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "title", "author", "licence", "source_page"])
        w.writerows(rows)
    print(f"\ndone: {len(rows)} images + attribution.csv in {args.out_dir}")


if __name__ == "__main__":
    main()
