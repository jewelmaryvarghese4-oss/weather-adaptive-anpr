cd ~/dehaze-eval/run/nighthaze
export PYTHONPATH=~/dehaze-eval/extra
export CUDA_VISIBLE_DEVICES=0
python3 - <<'EOF'
p = "models/restoration.py"
s = open(p).read()
if "NH_STRIDE" not in s:
    assert "                        stride = 4\n" in s and "manual_batching_size = 2048\n" in s
    s = s.replace("                        stride = 4\n", "                        stride = int(os.environ.get('NH_STRIDE', 4))\n", 1)
    s = s.replace("                        manual_batching_size = 2048\n", "                        manual_batching_size = int(os.environ.get('NH_BATCH', 2048))\n", 1)
    open(p, "w").write(s)
    print("restoration.py: stride and batch size are now adjustable")
EOF
export NH_BATCH=32
run() {
  rm -rf results/haze
  echo "=== NightHaze $1 stride=$NH_STRIDE $(date +%T)"
  timeout 1500 ~/mamba-venv/bin/python -u eval_diffusion.py --sid haze --config RealSHaze_small.yml --resume param/NightHaze_SPL_2w.pth.tar 2>&1 | grep -E "corners|Error|error|Traceback|out of memory" | cut -c1-160 | tail -6
  echo "finished $(date +%T): $(ls results/haze/output 2>/dev/null | wc -l) outputs"
  mkdir -p ~/dehaze-eval/out/nighthaze_$1_trial; cp results/haze/output/*.png ~/dehaze-eval/out/nighthaze_$1_trial/ 2>/dev/null
}
unset NH_NATIVE; export NH_STRIDE=16; run published
export NH_NATIVE=1; export NH_STRIDE=56; run native
ls ~/dehaze-eval/out/nighthaze_*_trial | head
