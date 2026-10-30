set -e
H=~/dehaze-eval
rm -rf $H/run/nighthaze; mkdir -p $H/run
cp -r $H/repos/nighthaze_codes $H/run/nighthaze
cd $H/run/nighthaze
mkdir -p data/haze param results
cp $H/data/in1536/*.png data/haze/
ln -sf $H/weights/NightHaze/NightHaze_SPL_2w.pth.tar param/NightHaze_SPL_2w.pth.tar
python3 - <<'EOF'
import os, re
H = os.path.expanduser("~/dehaze-eval/run/nighthaze")
# 1. load the checkpoint with an allow-list (only a few harmless classes are allowed to be unpickled)
p = H + "/utils/logging.py"
s = open(p).read()
old = s[s.index("def load_checkpoint(path, device):"):]
old = old[: old.index("return torch.load(path, map_location=device)") + len("return torch.load(path, map_location=device)")]
new = ('def load_checkpoint(path, device):\n    import argparse\n'
       '    with torch.serialization.safe_globals([argparse.Namespace, torch.device]):\n'
       '        return torch.load(path, map_location=(device if device is not None else "cpu"), weights_only=True)')
assert old in s
open(p, "w").write(s.replace(old, new))
# 2. keep our photo size unless NH_NATIVE is unset (then the repo's own 512x512 nearest resize is used, as published)
p = H + "/datasets/realshaze_add.py"
s = open(p).read()
a = "        wd_new = 512\n        ht_new = 512\n        input_img = input_img.resize((wd_new, ht_new), PIL.Image.NEAREST)\n        gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.NEAREST)\n"
assert a in s
b = ("        if os.environ.get('NH_NATIVE') != '1':\n"
     "            wd_new = 512\n            ht_new = 512\n"
     "            input_img = input_img.resize((wd_new, ht_new), PIL.Image.NEAREST)\n"
     "            gt_img = gt_img.resize((wd_new, ht_new), PIL.Image.NEAREST)\n"
     "        else:\n            gt_img = input_img\n")
open(p, "w").write(s.replace(a, b))
# 3. data dir
p = H + "/configs/RealSHaze.yml"
s = open(p).read()
open(p, "w").write(re.sub(r'data_dir: ".*"', 'data_dir: "%s/data"' % H, s))
print("patched: safe checkpoint load, optional native size, data_dir")
EOF
grep -n 'data_dir' configs/RealSHaze.yml; ls data/haze | wc -l
