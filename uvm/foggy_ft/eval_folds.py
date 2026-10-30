"""
Scores, fold by fold, on the plates held out of that fold's training:
  installed reader | weather-only seed 0 (no foggy crops) | fold model (weather + foggy crops of the other plates)
and adds the folds up. Also scores every fold model on heldout_rj250 (clear plates) so a loss there is visible.
    python foggy_ft/eval_folds.py [tag]
"""
import os
import re
import subprocess
import sys

TAG = sys.argv[1] if len(sys.argv) > 1 else "verified"
env = {**os.environ, "TWO_LINE_MODE": "fixed"}
D = "../datasets"
CK = {"installed": "checkpoints_lprnet/lprnet_avg.pt", "weather_only": "checkpoints_compare_weather/fixed_s0/lprnet_avg.pt"}


def ev(ckpt, test_dir):
    r = subprocess.run([sys.executable, "label_review.py", "eval", "--lprnet_ckpt", ckpt, "--test_dir", test_dir],
                       capture_output=True, text=True, env=env).stdout
    m = re.search(r"(\d+) held-out plates \| exact \(raw\) (\d+)/\d+.*?format-corrected\) (\d+)/\d+ \| CER ([0-9.]+)", r)
    return (int(m.group(1)), int(m.group(2)), int(m.group(3)), float(m.group(4))) if m else None


tot = {k: [0, 0, 0] for k in ("installed", "weather_only", "fold_model")}
print("%-6s %6s | %-18s | %-18s | %-18s | %s" % ("fold", "plates", "installed", "weather only", "fold model", "fold model on heldout_rj250"))
for k in range(4):
    test = "%s/foggy_ft/%s_fold%d_test" % (D, TAG, k)
    fm = "checkpoints_foggy_ft/%s_fold%d/fixed_s0/lprnet_avg.pt" % (TAG, k)
    if not os.path.exists(fm):
        print("fold %d: no model yet (%s)" % (k, fm))
        continue
    res = {n: ev(c, test) for n, c in CK.items()}
    res["fold_model"] = ev(fm, test)
    clear = ev(fm, D + "/heldout_rj250")
    n = res["installed"][0]
    for name, v in res.items():
        tot[name][0] += v[1]
        tot[name][1] += v[2]
        tot[name][2] += v[0]
    f = lambda v: "%2d exact / %2d fmt" % (v[1], v[2])
    print("%-6d %6d | %-18s | %-18s | %-18s | %s" % (k, n, f(res["installed"]), f(res["weather_only"]), f(res["fold_model"]),
                                                   "%d exact / %d fmt of %d" % (clear[1], clear[2], clear[0]) if clear else "?"))
print("TOTAL over held-out folds (crops: %d): installed %d exact / %d fmt | weather only %d / %d | fold model %d / %d" % (
    tot["installed"][2], tot["installed"][0], tot["installed"][1], tot["weather_only"][0], tot["weather_only"][1],
    tot["fold_model"][0], tot["fold_model"][1]))
print("(installed reader on heldout_rj250 for reference: 53 exact / 56 fmt of 62)")
