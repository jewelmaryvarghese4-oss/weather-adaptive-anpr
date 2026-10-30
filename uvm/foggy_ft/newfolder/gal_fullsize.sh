#!/bin/bash
# Full-size family of every UVM-based model of the project on the new photo folder -> $R/full/<name>/<stem>_restored.png (resumes: a finished folder is skipped)
source ~/mamba-venv/bin/activate
unset PYTORCH_CUDA_ALLOC_CONF
cd /mnt/c/projects/jewwww/uvm
R=/mnt/c/projects/jewwww/datasets/finalrun_foggyimages_gallery/results
UP=$R/upright; F=$R/full
N=$(ls $UP/*.jpg | wc -l)
mkdir -p $F
cnt() { ls "$1"/*_restored.png 2>/dev/null | wc -l; }
ok() { [ "$(cnt $F/$1)" -ge "$N" ]; }

tile() {   # name  checkpoint-folder      (tiles of 1536 px on the whole photo, as test_uvmnet.py)
  ok $1 || python test_uvmnet.py --input $UP --checkpoint $2/uvmnet_combined_final.pt --output $F/$1 --tile_size 1536 > $F/$1.log 2>&1
  echo "$1: $(cnt $F/$1) of $N  [$(date +%T)]"
}
dp() {     # name  checkpoint  [input folder]   (one 1536 px pass + joint upsampling, real_scratch/dehaze_photos.py)
  ok $1 || python real_scratch/dehaze_photos.py --checkpoint $2 --input ${3:-$UP} --output $F/$1 > $F/$1.log 2>&1
  echo "$1: $(cnt $F/$1) of $N  [$(date +%T)]"
}
phys() {   # name  source-folder  gamma
  ok $1 || python foggy_ft/clean/physics_dehaze.py --source $2 --hazy $UP --output $F/$1 --smooth 0.03 --gamma $3 > $F/$1.log 2>&1
  echo "$1: $(cnt $F/$1) of $N  [$(date +%T)]"
}
expo() {   # input-name  output-name
  ok $2 || python foggy_ft/clean/exposure_match.py --input $F/$1 --output $F/$2 > /dev/null 2>&1
  echo "$2: $(cnt $F/$2) of $N  [$(date +%T)]"
}
blend() {  # folder-A  folder-B  out-name
  ok $3 || python foggy_ft/clean/blend_pair.py $F/$1 $F/$2 $F/$3 > /dev/null 2>&1
  echo "$3: $(cnt $F/$3) of $N  [$(date +%T)]"
}

# --- the UVM family of the earlier galleries
tile ft2 checkpoints_rainfog2
tile pon checkpoints_ponmudi
tile _raw_gf checkpoints_ponmudi_gf
tile _raw_gopro checkpoints_ponmudi_gopro
ok gf || python foggy_ft/newfolder/nf_gf.py $UP $F/_raw_gf $F/gf > /dev/null 2>&1;            echo "gf: $(cnt $F/gf) of $N"
ok gopro_gf || python foggy_ft/newfolder/nf_gf.py $UP $F/_raw_gopro $F/gopro_gf > /dev/null 2>&1; echo "gopro_gf: $(cnt $F/gopro_gf) of $N"
ok gopro_clean || python foggy_ft/clean/dehaze_clean.py --input $UP --output $F/gopro_clean --tint_beta 0.8 > /dev/null 2>&1;   echo "gopro_clean: $(cnt $F/gopro_clean) of $N"
ok gopro_clean_notint || python foggy_ft/clean/dehaze_clean.py --input $UP --output $F/gopro_clean_notint --tint_beta 0 > /dev/null 2>&1; echo "gopro_clean_notint: $(cnt $F/gopro_clean_notint) of $N"
ok hybrid || python foggy_ft/clean/hybrid.py --tint 0.0 --output $F/hybrid --hazy_dir $UP --tiled_dir $F/gopro_gf --seam_dir $F/gopro_clean_notint > /dev/null 2>&1; echo "hybrid: $(cnt $F/hybrid) of $N"
ok hybrid_tint || python foggy_ft/clean/hybrid.py --tint 0.8 --output $F/hybrid_tint --hazy_dir $UP --tiled_dir $F/gopro_gf --seam_dir $F/gopro_clean_notint > /dev/null 2>&1; echo "hybrid_tint: $(cnt $F/hybrid_tint) of $N"

# --- the newer models
dp real checkpoints_real_mixed/keep_epoch22_ema.pt
dp real_long checkpoints_real_long/best_heavy_ema.pt
dp depthfog_ep18 checkpoints_depthfog/best_val_ema.pt
blend hybrid real hybrid_real_blend
if ! ok real_on_hybrid; then
  python real_scratch/dehaze_photos.py --checkpoint checkpoints_real_mixed/keep_epoch22_ema.pt --input $F/hybrid --output $F/_roh_tmp > $F/_roh_tmp.log 2>&1
  mkdir -p $F/real_on_hybrid
  for f in $F/_roh_tmp/*_restored_restored.png; do b=$(basename "$f" _restored_restored.png); mv "$f" "$F/real_on_hybrid/${b}_restored.png"; done
fi
echo "real_on_hybrid: $(cnt $F/real_on_hybrid) of $N"
ok hybrid_real_inv100 || python foggy_ft/clean/inv_combine.py --hybrid $F/hybrid --new $F/real --out $F/hybrid_real_inv100 > /dev/null 2>&1; echo "hybrid_real_inv100: $(cnt $F/hybrid_real_inv100) of $N"

# --- physics-guided versions
phys phys_blend $F/hybrid_real_blend 1.0
phys phys_strong $F/hybrid_real_blend 1.25
expo phys_strong phys_strong_exp
phys phys_gopro $F/_raw_gopro 1.0
phys phys_gopro_strong $F/_raw_gopro 1.25
expo phys_gopro_strong phys_gopro_strong_exp
phys phys_df_strong $F/depthfog_ep18 1.25
expo phys_df_strong phys_df_strong_exp
blend hybrid real_long hybrid_long_blend
phys phys_long_strong $F/hybrid_long_blend 1.25
expo phys_long_strong phys_long_strong_exp
echo NF_FULL_FINISHED
