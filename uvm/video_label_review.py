"""
Turns plates tracked in traffic videos into LPRNet training data, with a person confirming each plate.

Why: the reader's remaining errors are real two-line bike/auto plates (series letter at the end of
the top line, 'KL01C / X6171', or old 'KL 15 / 8807' plates). Synthetic plates did not fix that;
real crops from our own videos should. One typed plate number labels a whole track: dozens of crops
of the same plate at different sizes and blur levels.

1) prepare: needs runs made with  read_video.py --save_track_crops 3 ...  Builds review.html with one
   card per tracked plate (clearest crop enlarged + 3 more frames) and the model's vote pre-filled.
       python video_label_review.py prepare --runs dataset14_out1 dataset14_out2 dataset14_out3 dataset14_out5
   Open ../datasets/video_ocr_review/review.html, check/correct every card, Export labels CSV.

2) ingest: copies up to --max_per_plate crops of every confirmed plate into ../datasets/video_ocr_crops,
   writes its manifest.csv, and a combined training manifest ../datasets/final_manifest_video.csv
   (final_manifest.csv + these crops). Plates that also appear in the video-4 test set are left out.
       python video_label_review.py ingest --csv ~/Downloads/labels_video_ocr.csv
"""
import argparse
import csv
import json
import os
import re
import shutil

from PIL import Image

from read_video import levenshtein

D = "../datasets"
# state + district (1-2 digits) + 0-3 series letters + 3-4 digits: KL01CX6171, KL158057, KL01CD321, KL22M308
PLATE_SHAPE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{3,4}$")
REVIEW = f"{D}/video_ocr_review"      # defaults = round 1; round 2+ pass --review_dir/--out_dir/... (see main)
OUT = f"{D}/video_ocr_crops"
OUT_NAME = "video_ocr_crops"
BASE_MANIFEST = f"{D}/final_manifest.csv"
OUT_MANIFEST = f"{D}/final_manifest_video.csv"

HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Video plate review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#1e1e1e;color:#eee}
header{position:sticky;top:0;background:#111;padding:10px 16px;z-index:5;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(520px,1fr));gap:12px;padding:12px}
.card{background:#2a2a2a;border-radius:8px;padding:10px;border:3px solid transparent}
.card.ok{border-color:#2e9e4f}.card.fix{border-color:#d9a400}.card.skip{border-color:#777;opacity:.45}
.card img{width:100%;background:#000}
input.txt{width:100%;font:600 24px Consolas,monospace;letter-spacing:3px;box-sizing:border-box;padding:6px;margin:6px 0}
button{padding:6px 12px;margin-right:6px;cursor:pointer}
.meta{font-size:12px;color:#aaa}
.hide{display:none}
.tip{font-size:13px;color:#ffd166;max-width:900px}
</style></head><body>
<header>
  <b>Video plate review</b><span id="counts"></span>
  <label><input type="checkbox" id="hidedone"> hide done</label>
  <button id="export">Export labels CSV</button>
  <span class="tip">The box is pre-filled with the model's guess: check EVERY character against the picture.
  Typical model mistakes: a dropped letter at the end of the top line (KL01C / X6171), 8 vs B, 0 vs D vs Q, 1 vs I.
  Not a plate, or you can't read it with certainty: press skip.</span>
</header>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = 'video-plate-review:' + location.pathname;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
function persist() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} refresh(); }
function clean(t) { return t.toUpperCase().replace(/[^A-Z0-9]/g, ''); }
function build() {
  const grid = document.getElementById('grid');
  DATA.forEach(d => {
    const card = document.createElement('div');
    card.className = 'card'; card.id = 'c_' + d.id;
    card.innerHTML = '<img loading="lazy" src="' + d.img + '">' +
      '<div class="meta">' + d.label + '</div>' +
      '<input class="txt" spellcheck="false">' +
      '<div><button class="bok">correct</button><button class="bskip">skip / not a plate</button><button class="breset">reset</button></div>';
    const input = card.querySelector('.txt');
    input.value = (state[d.id] && state[d.id].text !== undefined) ? state[d.id].text : d.pred;
    input.addEventListener('input', () => {
      const t = clean(input.value); input.value = t;
      state[d.id] = { text: t, status: t === d.pred ? 'ok' : 'fix' }; persist();
    });
    input.addEventListener('keydown', e => { if (e.key === 'Enter') card.querySelector('.bok').click(); });
    card.querySelector('.bok').onclick = () => {
      const t = clean(input.value); input.value = t;
      state[d.id] = { text: t, status: t === d.pred ? 'ok' : 'fix' }; persist();
    };
    card.querySelector('.bskip').onclick = () => { state[d.id] = { text: clean(input.value), status: 'skip' }; persist(); };
    card.querySelector('.breset').onclick = () => { delete state[d.id]; input.value = d.pred; persist(); };
    grid.appendChild(card);
  });
}
function refresh() {
  let ok = 0, fix = 0, skip = 0;
  const hide = document.getElementById('hidedone').checked;
  DATA.forEach(d => {
    const s = (state[d.id] || {}).status || '';
    document.getElementById('c_' + d.id).className = 'card ' + s + (hide && s ? ' hide' : '');
    if (s === 'ok') ok++; else if (s === 'fix') fix++; else if (s === 'skip') skip++;
  });
  document.getElementById('counts').textContent = DATA.length + ' plates | correct ' + ok + ' | edited ' + fix +
    ' | skipped ' + skip + ' | left ' + (DATA.length - ok - fix - skip);
}
document.getElementById('hidedone').onchange = refresh;
document.getElementById('export').onclick = () => {
  const left = DATA.filter(d => !(state[d.id] || {}).status).length;
  if (left && !confirm(left + ' plates are untouched and will be ignored. Export anyway?')) return;
  const q = v => '"' + String(v).replace(/"/g, '""') + '"';
  const lines = ['id,pred,text,status'];
  DATA.forEach(d => { const s = state[d.id] || {};
    lines.push([q(d.id), q(d.pred), q(s.text !== undefined ? s.text : d.pred), q(s.status || '')].join(',')); });
  const blob = new Blob([lines.join('\\n')], { type: 'text/csv' });
  const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = 'labels_video_ocr.csv'; a.click();
};
build(); refresh();
</script></body></html>
"""


def load_run(run, min_crops):
    """Tracks of one read_video.py run that have at least min_crops saved crops."""
    base = f"{D}/{run}"
    info = {int(r["id"]): r for r in csv.DictReader(open(f"{base}/tracks.csv", encoding="utf-8"))}
    area = {}                                             # (track, frame) -> crop area, from readings.csv
    for r in csv.DictReader(open(f"{base}/readings.csv", encoding="utf-8")):
        area[(int(r["track"]), int(r["frame"]))] = int(r["crop_w"]) * int(r["crop_h"])
    out = []
    for tid_s in os.listdir(f"{base}/track_crops"):        # one listdir per track, no per-file stat
        tid = int(tid_s)
        if tid not in info:
            continue
        files = sorted(os.listdir(f"{base}/track_crops/{tid_s}"))
        if len(files) < min_crops:
            continue
        frames = [int(f[:-4]) for f in files]
        out.append({"run": run, "track": tid, "files": [f"{base}/track_crops/{tid_s}/{f}" for f in files],
                    "areas": [area.get((tid, fr), 0) for fr in frames], "info": info[tid]})
    return out


def card_image(t, path):
    big_i = max(range(len(t["files"])), key=lambda i: t["areas"][i])
    big = Image.open(t["files"][big_i]).convert("RGB")
    s = 150 / big.height
    big = big.resize((max(1, min(500, int(big.width * s))), 150), Image.LANCZOS)
    n = len(t["files"])
    others = [Image.open(t["files"][i]).convert("RGB") for i in (n // 6, n // 2, 5 * n // 6)]
    others = [o.resize((max(1, int(o.width * 70 / o.height)), 70), Image.LANCZOS) for o in others]
    W = max(big.width, sum(o.width for o in others) + 10 * len(others))
    canvas = Image.new("RGB", (W, 150 + 8 + 70), (0, 0, 0))
    canvas.paste(big, (0, 0))
    x = 0
    for o in others:
        canvas.paste(o, (x, 158)); x += o.width + 10
    canvas.save(path, quality=92)


def cmd_prepare(args):
    os.makedirs(f"{REVIEW}/cards", exist_ok=True)
    tracks = []
    for run in args.runs:
        tracks += load_run(run, args.min_crops)
    tracks.sort(key=lambda t: (t["run"], -len(t["files"])))
    data, meta = [], {}
    for t in tracks:
        cid = f"{t['run'].replace('/', '_')}_t{t['track']}"
        card_image(t, f"{REVIEW}/cards/{cid}.jpg")
        inf = t["info"]
        data.append({"id": cid, "pred": inf["consensus"], "img": f"cards/{cid}.jpg",
                     "label": f"{t['run']} track {t['track']} &middot; {inf['frames']} frames &middot; model vote "
                              f"<b>{inf['consensus']}</b> (agreed in {inf['support']} frames)"})
        meta[cid] = {"run": t["run"], "track": t["track"], "files": t["files"]}
    with open(f"{REVIEW}/review.html", "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", json.dumps(data)))
    json.dump(meta, open(f"{REVIEW}/cards.json", "w"))
    print(f"{len(data)} plate cards from {len(args.runs)} runs -> {os.path.abspath(REVIEW)}/review.html")


def test_plates(bench_path, test_runs):
    b = json.load(open(bench_path, encoding="utf-8-sig"))
    return {e["plate"] for e in b["plates"] if e["video_out"] in test_runs}


def cmd_ingest(args):
    meta = json.load(open(f"{REVIEW}/cards.json"))
    held = test_plates(args.benchmark, args.test_runs)
    if os.path.exists(OUT) and not args.overwrite:
        raise SystemExit(f"{OUT} exists; pass --overwrite to rebuild it")
    shutil.rmtree(OUT, ignore_errors=True)
    os.makedirs(f"{OUT}/crops")
    rows, kept, skipped_test, per_plate, rejected = [], 0, [], {}, []
    with open(os.path.expanduser(args.csv), newline="", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            if r["status"] not in ("ok", "fix") or r["id"] not in meta:
                continue
            text = "".join(ch for ch in r["text"].upper() if ch.isalnum())
            if not PLATE_SHAPE.match(text):     # typos / partial plates never become training labels
                rejected.append(text)
                continue
            if any(levenshtein(text, h) <= 1 for h in held):
                skipped_test.append(text)       # also in the video-4 test: never train on it
                continue
            files = meta[r["id"]]["files"]
            if len(files) > args.max_per_plate:  # evenly spread over the track (different sizes/blur)
                step = len(files) / args.max_per_plate
                files = [files[int(i * step)] for i in range(args.max_per_plate)]
            for p in files:
                name = f"{r['id']}_{os.path.basename(p)}"
                shutil.copyfile(p, f"{OUT}/crops/{name}")
                rows.append((f"crops/{name}", text, 1, 0, 1.0, meta[r["id"]]["run"], "video"))
            kept += 1
            per_plate[text] = per_plate.get(text, 0) + len(files)
    with open(f"{OUT}/manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "text", "verified", "dist", "conf", "src", "group"])
        w.writerows(rows)
    # combined manifest = the base manifest (round 2: the round-1 combined one) + these crops
    # (paths are relative to ../datasets)
    with open(OUT_MANIFEST, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        base = list(csv.reader(open(BASE_MANIFEST, encoding="utf-8")))
        w.writerow(base[0])
        w.writerows(base[1:])
        cols = base[0]
        for path, text, *_rest in rows:
            row = {"path": f"{OUT_NAME}/{path}", "text": text, "verified": "1", "group": "video"}
            w.writerow([row.get(c, "") for c in cols])
    print(f"{kept} confirmed plates -> {len(rows)} training crops in {OUT} ({len(per_plate)} distinct plates)")
    print(f"left out because they are in the held-out test videos: {sorted(set(skipped_test))}")
    print(f"left out because they don't look like a whole plate (typo/partial?): {rejected}")
    print(f"combined training manifest ({len(base) - 1} base rows + {len(rows)} new): {OUT_MANIFEST}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--runs", nargs="+", required=True,
                   help="run folders inside ../datasets (read_video.py --save_track_crops or live_plates.py --export_run)")
    p.add_argument("--min_crops", type=int, default=8, help="skip tracks with fewer saved crops (short/junk)")
    i = sub.add_parser("ingest")
    i.add_argument("--csv", required=True)
    i.add_argument("--max_per_plate", type=int, default=40)
    i.add_argument("--benchmark", default=f"{D}/video_benchmark.json")
    i.add_argument("--test_runs", nargs="+", default=["dataset14_out4"])
    i.add_argument("--overwrite", action="store_true")
    for s in (p, i):    # round 2+: keep round 1's review page, crops and manifest untouched
        s.add_argument("--review_dir", default="video_ocr_review", help="folder inside ../datasets")
    i.add_argument("--out_dir", default="video_ocr_crops", help="folder inside ../datasets for the training crops")
    i.add_argument("--base_manifest", default="final_manifest.csv",
                   help="manifest to extend (round 2: final_manifest_video.csv)")
    i.add_argument("--out_manifest", default="final_manifest_video.csv")
    args = ap.parse_args()
    global REVIEW, OUT, OUT_NAME, BASE_MANIFEST, OUT_MANIFEST
    REVIEW = f"{D}/{args.review_dir}"
    if args.cmd == "ingest":
        OUT_NAME, OUT = args.out_dir, f"{D}/{args.out_dir}"
        BASE_MANIFEST, OUT_MANIFEST = f"{D}/{args.base_manifest}", f"{D}/{args.out_manifest}"
    cmd_prepare(args) if args.cmd == "prepare" else cmd_ingest(args)


if __name__ == "__main__":
    main()
