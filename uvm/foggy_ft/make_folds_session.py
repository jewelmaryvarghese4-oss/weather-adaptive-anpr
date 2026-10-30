"""
Manifests for the 4-fold test with the extra session data (same folds / held-out plates as make_folds.py's 'verified' run).

    python foggy_ft/make_folds_session.py

For fold k the training manifest = the rj250 manifest
  + verified foggy crops of the plates NOT in fold k (max 4 per plate, x4)               [same as the previous fold run]
  + session crops (photo crops, video tracks), max 8 per plate (x4), skipping every plate that is in fold k.
A session crop whose plate is one of our verified foggy plates is labelled with the VERIFIED text, never with the reader's reading.
Writes datasets/foggy_ft_manifest_session_fold{k}.csv. The held-out test crops stay in datasets/foggy_ft/verified_fold{k}_test.
"""
import csv
import json
import os
import random

HERE = os.path.dirname(os.path.abspath(__file__))
D = os.path.abspath(os.path.join(HERE, "..", "..", "datasets"))
FT = os.path.join(D, "foggy_ft")
fold_of = json.load(open(os.path.join(FT, "verified_folds.json")))["fold_of"]
K = 4
CAP_FOGGY, CAP_SESSION, REPEAT = 4, 8, 4
KEEP_AUTO = {"KL01DK4889", "KL02AB3092", "KL10BB8545", "TN75BE5650"}   # photo plates whose reader label I checked by eye

base = list(csv.reader(open(os.path.join(D, "final_manifest_rj250_renamed.csv"), encoding="utf-8")))
hdr, body = base[0], base[1:]

foggy = {}
for r in csv.DictReader(open(os.path.join(FT, "foggy_labels_verified.csv"), encoding="utf-8")):
    if r["status"] in ("ok", "fix") and r["label"]:
        foggy.setdefault(r["label"], []).append(("foggy_ft/crops/%03d.png" % int(r["number"]), int(r["number"])))

session = {}          # plate text -> list of (path, source)
for r in csv.DictReader(open(os.path.join(D, "session_clear", "matched.csv"), encoding="utf-8")):
    session.setdefault(r["text"], []).append((r["path"], "photo_matched"))
for r in csv.DictReader(open(os.path.join(D, "session_clear", "accepted.csv"), encoding="utf-8")):
    if r["text"] in KEEP_AUTO:
        session.setdefault(r["text"], []).append((r["path"], "photo_checked"))
for r in csv.DictReader(open(os.path.join(D, "video_tracks", "crops.csv"), encoding="utf-8")):
    t = r["matches_verified_foggy_plate"] or r["proposed_label"]
    if t in KEEP_AUTO or t in fold_of:                # video tracks: only plates that were verified by eye or are verified foggy plates
        session.setdefault(t, []).append(("video_tracks/crops/" + r["file"], "video"))
rng = random.Random(0)
for t in session:
    rng.shuffle(session[t])
    session[t] = session[t][:CAP_SESSION]
print("session plates: %d | crops: %d | of the plates, in the test set: %d" % (len(session), sum(len(v) for v in session.values()), sum(t in fold_of for t in session)))
for t in sorted(session):
    print("   %-11s %2d crops %s" % (t, len(session[t]), "(test plate, fold %d)" % fold_of[t] if t in fold_of else ""))
for p in (p for v in session.values() for p, _ in v):
    assert os.path.exists(os.path.join(D, p)), p

for k in range(K):
    rows = []
    nf = ns = 0
    for t, lst in foggy.items():
        if fold_of[t] != k:
            for path, _ in sorted(lst, key=lambda x: x[1])[:CAP_FOGGY]:
                rows += [[path, t, "1", "foggy_train"] + [""] * (len(hdr) - 4)] * REPEAT
                nf += 1
    for t, lst in session.items():
        if fold_of.get(t, -1) != k:
            for path, src in lst:
                rows += [[path, t, "1", "session_" + src] + [""] * (len(hdr) - 4)] * REPEAT
                ns += 1
    out = os.path.join(D, "foggy_ft_manifest_session_fold%d.csv" % k)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(hdr)
        w.writerows(body)
        w.writerows(rows)
    miss = sum(not os.path.exists(os.path.join(D, r[0])) for r in rows)
    print("fold %d: +%d verified foggy crops, +%d session crops (x%d) | missing files: %d | %s" % (k, nf, ns, REPEAT, miss, os.path.basename(out)))
