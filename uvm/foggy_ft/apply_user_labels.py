"""
Turns the user's answers on the check sheets into datasets/foggy_ft/foggy_labels_verified.csv.

status:  ok         my 'sure' reading, the user raised no objection
         fix        the user gave the plate (or corrected mine)
         likely     a user guess ("ig") -> NOT used for training or testing
         unreadable the user cannot read it either
         notplate   not a plate / not needed
make_folds.py only uses ok and fix.
"""
import csv
import os

D = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "datasets"))
FT = os.path.join(D, "foggy_ft")
draft = {int(r["number"]): r for r in csv.DictReader(open(os.path.join(D, "foggy_eval", "full_crop_pipeline", "readings_with_draft_truth.csv"), encoding="utf-8"))}

USER_PLATE = {3: "KL29H5018", 71: "KL22Y6347", 75: "KL22N8082", 76: "KL39K9381", 91: "KL02BB865", 99: "KL22O1362"}
USER_GUESS = {86: "KL01DG3085"}                     # "ig" = I guess
UNREADABLE = {15, 18, 23, 24, 33, 43, 50, 51, 52, 53, 59, 69, 70}
DISPUTED = {21: "user reads KL16F... and doubts my KL16P6373"}
NOT_PLATE = {85, 95}

rows = []
for n, r in sorted(draft.items()):
    if n in USER_PLATE:
        lab, st = USER_PLATE[n], "fix"
    elif n in USER_GUESS:
        lab, st = USER_GUESS[n], "likely"
    elif n in UNREADABLE or n in DISPUTED:
        lab, st = "", "unreadable"
    elif n in NOT_PLATE:
        lab, st = "", "notplate"
    elif r["draft_confidence"] == "sure":
        lab, st = r["draft_true_plate"], "ok"
    elif r["draft_confidence"] == "likely":
        lab, st = r["draft_true_plate"], "ok"          # user: "the others are correct, the listed corrections were needed"
    else:
        lab, st = "", "unreadable"
    rows.append([n, r["photo"], r["plate"], lab, st])
with open(os.path.join(FT, "foggy_labels_verified.csv"), "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["number", "photo", "plate", "label", "status"])
    w.writerows(rows)
from collections import Counter
c = Counter(r[4] for r in rows)
print("statuses:", dict(c))
ok = [r for r in rows if r[4] in ("ok", "fix")]
print("verified crops:", len(ok), "| distinct plates:", len({r[3] for r in ok}))
print("unconfirmed 'likely' crops:", [(r[0], r[3]) for r in rows if r[4] == "likely"])
