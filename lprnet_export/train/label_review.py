"""
Fast labeling loop for new plate photos -- the model pre-reads every plate,
you only correct the wrong ones.

1) prepare : detect plates (YOLO) + read them (LPRNet) in a folder of photos,
             save each crop and build review.html (open it in a browser).
             In the page: press "correct" when the pre-filled text is right,
             edit the text when it's wrong, "skip" for unreadable / not a
             plate. Progress is saved in the browser; "Export labels.csv"
             downloads your answers.
2) ingest  : turn the exported CSV into training data (crops + ground_truth
             entries + duplicate copies, the same convention as the ds6_*
             scenes). --test_frac sets a share of the accepted plates aside
             as a clean held-out test set that is NEVER trained on.
3) eval    : score a checkpoint on that held-out test set.

Only rows you explicitly marked correct/edited are used; untouched rows are
ignored, so a half-finished review is safe.

Run from WSL2:
    source ~/mamba-venv/bin/activate
    cd /mnt/c/projects/jewwww/uvm
    python label_review.py prepare --image_dir ../datasets/dataset9 --out_dir ../datasets/dataset9_review --prefix ds9
    # open ../datasets/dataset9_review/review.html in Chrome, label, Export labels.csv
    python label_review.py ingest --review_dir ../datasets/dataset9_review --csv /mnt/c/Users/Jewel/Downloads/labels_ds9.csv --test_frac 0.3
    python label_review.py eval --lprnet_ckpt checkpoints_lprnet/lprnet_avg.pt
"""
import argparse
import csv
import datetime
import json
import os
import random
import re
import shutil

from PIL import Image

DEFAULT_GT = "../evaluation/baseline_ocr/ground_truth.json"
DEFAULT_DATA_ROOT = "../datasets/dataset"
DEFAULT_TEST_DIR = "../datasets/heldout_test"
VALID_TEXT = re.compile(r"^[A-Z0-9]{3,12}$")


def safe_stem(name):
    return re.sub(r"[^A-Za-z0-9]+", "_", os.path.splitext(name)[0]).strip("_")


def normalize(text):
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Plate label review</title>
<style>
body{font-family:Segoe UI,Arial,sans-serif;margin:0;background:#1e1e1e;color:#eee}
header{position:sticky;top:0;background:#111;padding:10px 16px;z-index:5;display:flex;gap:18px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(360px,1fr));gap:12px;padding:12px}
.card{background:#2a2a2a;border-radius:8px;padding:10px;border:3px solid transparent}
.card.ok{border-color:#2e9e4f}.card.fix{border-color:#d9a400}.card.skip{border-color:#777;opacity:.5}
.card img{width:100%;max-height:200px;object-fit:contain;background:#000}
input.txt{width:100%;font:600 22px Consolas,monospace;letter-spacing:2px;box-sizing:border-box;padding:6px;margin:6px 0}
button{padding:6px 12px;margin-right:6px;cursor:pointer}
.meta{font-size:12px;color:#aaa;word-break:break-all}
.hide{display:none}
</style></head><body>
<header>
  <b>Plate label review</b>
  <span id="counts"></span>
  <label><input type="checkbox" id="hidedone"> hide done</label>
  <button id="export">Export labels CSV</button>
  <span class="meta">green = correct, yellow = edited, grey = skipped. Untouched cards are ignored.</span>
</header>
<div class="grid" id="grid"></div>
<script>
const DATA = __DATA__;
const KEY = 'plate-review:' + location.pathname;
let state = {};
try { state = JSON.parse(localStorage.getItem(KEY) || '{}'); } catch (e) {}
function persist() { try { localStorage.setItem(KEY, JSON.stringify(state)); } catch (e) {} refresh(); }
function clean(t) { return t.toUpperCase().replace(/[^A-Z0-9]/g, ''); }
function build() {
  const grid = document.getElementById('grid');
  DATA.forEach(d => {
    const card = document.createElement('div');
    card.className = 'card'; card.id = 'c_' + d.id;
    card.innerHTML = '<img src="' + d.img + '">' +
      '<div class="meta">' + d.id + ' &middot; conf ' + d.conf + ' &middot; ' + d.w + 'x' + d.h +
      (d.alt ? ' &middot; format-corrected suggestion: <b>' + d.alt + '</b>' : '') + '</div>' +
      '<input class="txt" spellcheck="false">' +
      '<div><button class="bok">correct</button><button class="bskip">skip / unreadable</button><button class="breset">reset</button></div>';
    const input = card.querySelector('.txt');
    input.value = (state[d.id] && state[d.id].text !== undefined) ? state[d.id].text : d.pred;
    input.addEventListener('input', () => {
      const t = clean(input.value); input.value = t;
      state[d.id] = { text: t, status: t === d.pred ? 'ok' : 'fix' }; persist();
    });
    card.querySelector('.bok').onclick = () => {
      const t = clean(input.value); input.value = t;
      state[d.id] = { text: t, status: t === d.pred ? 'ok' : 'fix' }; persist();
    };
    card.querySelector('.bskip').onclick = () => {
      state[d.id] = { text: clean(input.value), status: 'skip' }; persist();
    };
    card.querySelector('.breset').onclick = () => {
      delete state[d.id]; input.value = d.pred; persist();
    };
    grid.appendChild(card);
  });
}
function refresh() {
  let ok = 0, fix = 0, skip = 0;
  const hide = document.getElementById('hidedone').checked;
  DATA.forEach(d => {
    const s = (state[d.id] || {}).status || '';
    const card = document.getElementById('c_' + d.id);
    card.className = 'card ' + s + (hide && s ? ' hide' : '');
    if (s === 'ok') ok++; else if (s === 'fix') fix++; else if (s === 'skip') skip++;
  });
  const left = DATA.length - ok - fix - skip;
  document.getElementById('counts').textContent =
    DATA.length + ' plates | correct ' + ok + ' | edited ' + fix + ' | skipped ' + skip + ' | left ' + left;
}
document.getElementById('hidedone').onchange = refresh;
document.getElementById('export').onclick = () => {
  const left = DATA.filter(d => !(state[d.id] || {}).status).length;
  if (left && !confirm(left + ' plates are untouched and will be ignored. Export anyway?')) return;
  const q = v => '"' + String(v).replace(/"/g, '""') + '"';
  const lines = ['id,pred,text,status'];
  DATA.forEach(d => {
    const s = state[d.id] || {};
    lines.push([q(d.id), q(d.pred), q(s.text !== undefined ? s.text : d.pred), q(s.status || '')].join(','));
  });
  const blob = new Blob([lines.join('\\n')], { type: 'text/csv' });
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob); a.download = '__EXPORT__'; a.click();
};
build(); refresh();
</script></body></html>
"""


def cmd_prepare(args):
    import torch
    from ultralytics import YOLO

    from model_lprnet import LPRNet
    from lprnet_vocab import NUM_CLASSES
    from read_plate import DEFAULT_YOLO_CKPT, _natural_sort_key, read_plate

    device = torch.device(args.device)
    lprnet = LPRNet(num_classes=NUM_CLASSES).to(device)
    lprnet.load_state_dict(torch.load(args.lprnet_ckpt, map_location=device))
    lprnet.eval()
    yolo = YOLO(args.yolo_ckpt or DEFAULT_YOLO_CKPT)

    crops_dir = os.path.join(args.out_dir, "crops")
    os.makedirs(crops_dir, exist_ok=True)
    exts = (".jpg", ".jpeg", ".png", ".webp", ".bmp")
    files = sorted((f for f in os.listdir(args.image_dir) if f.lower().endswith(exts)), key=_natural_sort_key)

    records, skipped_small = [], 0
    for fname in files:
        img = Image.open(os.path.join(args.image_dir, fname)).convert("RGB")
        res = yolo.predict(img, conf=args.yolo_conf, verbose=False)
        k = 0
        for r in res:
            for xyxy, conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                crop = img.crop(tuple(xyxy))
                if crop.width < args.min_w or crop.height < args.min_h:
                    skipped_small += 1
                    continue
                pid = f"{args.prefix}_{safe_stem(fname)}_p{k}"
                crop.save(os.path.join(crops_dir, pid + ".png"))
                raw, fmt = read_plate(lprnet, crop, device)
                records.append({
                    "id": pid, "img": f"crops/{pid}.png", "pred": raw,
                    "alt": fmt if fmt != raw else "", "conf": round(conf, 2),
                    "w": crop.width, "h": crop.height, "src": fname,
                    "box": [round(v) for v in xyxy],
                })
                k += 1

    with open(os.path.join(args.out_dir, "predictions.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "src", "box", "conf", "pred", "alt"])
        for r in records:
            w.writerow([r["id"], r["src"], r["box"], r["conf"], r["pred"], r["alt"]])

    data_json = json.dumps(records).replace("</", "<\\/")
    with open(os.path.join(args.out_dir, "review.html"), "w", encoding="utf-8") as f:
        f.write(HTML.replace("__DATA__", data_json).replace("__EXPORT__", f"labels_{args.prefix}.csv"))

    print(f"{len(files)} photos -> {len(records)} plate crops "
          f"({skipped_small} skipped as smaller than {args.min_w}x{args.min_h})")
    print(f"open: {os.path.abspath(os.path.join(args.out_dir, 'review.html'))}")


def cmd_ingest(args):
    with open(args.csv, newline="", encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))

    gt = json.load(open(args.ground_truth))
    test_labels_path = os.path.join(args.test_dir, "labels.json")
    test_labels = json.load(open(test_labels_path)) if os.path.exists(test_labels_path) else {}

    accepted, problems = [], []
    for r in rows:
        status = (r.get("status") or "").strip().lower()
        if status not in ("ok", "fix"):
            continue
        pid, text = r["id"], normalize(r.get("text"))
        crop_path = os.path.join(args.review_dir, "crops", pid + ".png")
        if not VALID_TEXT.match(text):
            problems.append(f"{pid}: text {text!r} is not 3-12 letters/digits -- skipped")
        elif not os.path.exists(crop_path):
            problems.append(f"{pid}: crop file missing -- skipped")
        elif (pid in gt or pid in test_labels) and not args.overwrite:
            problems.append(f"{pid}: id already exists (use a new --prefix or --overwrite) -- skipped")
        else:
            accepted.append((pid, text, crop_path, status))

    # Stratified split: plates you had to EDIT (the model got them wrong) are
    # split separately from plates accepted as-is. A plain random split can put
    # every hard plate in training, leaving a test set the model trivially passes.
    rng = random.Random(args.seed)
    test_set, train_set = [], []
    for st in ("fix", "ok"):
        group = sorted(a for a in accepted if a[3] == st)
        rng.shuffle(group)
        n_t = int(round(args.test_frac * len(group)))
        test_set += group[:n_t]
        train_set += group[n_t:]
    n_fix_test = sum(1 for a in test_set if a[3] == "fix")
    n_fix_all = sum(1 for a in accepted if a[3] == "fix")

    if not args.dry_run:
        if train_set:
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            shutil.copy(args.ground_truth, f"{args.ground_truth}.{stamp}.bak")
            clear_dir = os.path.join(args.data_root, "clear")
            hazy_dir = os.path.join(args.data_root, "hazy")
            os.makedirs(clear_dir, exist_ok=True)
            os.makedirs(hazy_dir, exist_ok=True)
            for pid, text, crop_path, _st in train_set:
                im = Image.open(crop_path).convert("RGB")
                im.save(os.path.join(clear_dir, pid + ".jpg"), quality=95)
                for v in range(args.copies):
                    im.save(os.path.join(hazy_dir, f"{pid}_v{v}.jpg"), quality=95)
                gt[pid] = text
            with open(args.ground_truth, "w") as f:
                json.dump(gt, f, indent=2)
        if test_set:
            os.makedirs(os.path.join(args.test_dir, "crops"), exist_ok=True)
            for pid, text, crop_path, _st in test_set:
                shutil.copy(crop_path, os.path.join(args.test_dir, "crops", pid + ".png"))
                test_labels[pid] = text
            with open(test_labels_path, "w") as f:
                json.dump(test_labels, f, indent=2)

    tag = " (dry run, nothing written)" if args.dry_run else ""
    print(f"accepted {len(accepted)} labels{tag}: {len(train_set)} -> training "
          f"(x{args.copies} duplicate copies each), {len(test_set)} -> held-out test "
          f"({n_fix_test} of the {n_fix_all} model-wrong plates are in the test set)")
    print(f"ground_truth.json now has {len(gt)} scenes; test set has {len(test_labels)} plates")
    for p in problems:
        print("  note:", p)
    if train_set and not args.dry_run:
        print("train with: --force_train " + ",".join(a[0] for a in train_set))


def cmd_eval(args):
    import torch

    from model_lprnet import LPRNet
    from lprnet_vocab import NUM_CLASSES
    from read_plate import read_plate
    from train_lprnet import levenshtein
    import two_line_plate

    labels = json.load(open(os.path.join(args.test_dir, "labels.json")))
    device = torch.device(args.device)
    lprnet = LPRNet(num_classes=NUM_CLASSES).to(device)
    lprnet.load_state_dict(torch.load(args.lprnet_ckpt, map_location=device))
    lprnet.eval()

    print(f"checkpoint: {args.lprnet_ckpt} | two-line mode: {two_line_plate.MODE}")
    n = raw_ok = fmt_ok = 0
    cer_num = cer_den = 0
    for pid, text in sorted(labels.items()):
        crop = Image.open(os.path.join(args.test_dir, "crops", pid + ".png")).convert("RGB")
        raw, fmt = read_plate(lprnet, crop, device)
        n += 1
        raw_ok += raw == text
        fmt_ok += fmt == text
        cer_num += levenshtein(raw, text)
        cer_den += len(text)
        print(f"{pid[-34:]:<34} truth={text:<12} pred={raw:<14}{'OK' if raw == text else ''}")
    print(f"\n{n} held-out plates | exact (raw) {raw_ok}/{n} = {raw_ok / max(n, 1):.1%} | "
          f"exact (format-corrected) {fmt_ok}/{n} | CER {cer_num / max(cer_den, 1):.3f}")


def main():
    import torch

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"

    p = sub.add_parser("prepare")
    p.add_argument("--image_dir", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--prefix", required=True, help="unique id prefix for this batch, e.g. ds9")
    p.add_argument("--lprnet_ckpt", default="checkpoints_lprnet/lprnet_avg.pt")
    p.add_argument("--yolo_ckpt", default=None)
    p.add_argument("--yolo_conf", type=float, default=0.4,
                   help="higher than read_plate's 0.25 to skip most false detections")
    p.add_argument("--min_w", type=int, default=50)
    p.add_argument("--min_h", type=int, default=18)
    p.add_argument("--device", default=dev)

    i = sub.add_parser("ingest")
    i.add_argument("--review_dir", required=True)
    i.add_argument("--csv", required=True, help="labels.csv exported from review.html")
    i.add_argument("--test_frac", type=float, default=0.0,
                   help="share of accepted plates kept as a held-out test set, never trained on")
    i.add_argument("--copies", type=int, default=4, help="duplicate copies per training plate")
    i.add_argument("--seed", type=int, default=0)
    i.add_argument("--ground_truth", default=DEFAULT_GT)
    i.add_argument("--data_root", default=DEFAULT_DATA_ROOT)
    i.add_argument("--test_dir", default=DEFAULT_TEST_DIR)
    i.add_argument("--overwrite", action="store_true")
    i.add_argument("--dry_run", action="store_true")

    e = sub.add_parser("eval")
    e.add_argument("--lprnet_ckpt", default="checkpoints_lprnet/lprnet_avg.pt")
    e.add_argument("--test_dir", default=DEFAULT_TEST_DIR)
    e.add_argument("--device", default=dev)

    args = ap.parse_args()
    {"prepare": cmd_prepare, "ingest": cmd_ingest, "eval": cmd_eval}[args.cmd](args)


if __name__ == "__main__":
    main()
