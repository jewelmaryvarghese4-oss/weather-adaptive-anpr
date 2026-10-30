#!/bin/bash
# Installs the rj250 seed-0 reader as the default (the current default is backed up first).
cd /mnt/c/projects/jewwww/uvm/checkpoints_lprnet
cp lprnet_avg.pt lprnet_avg_pre_rj250.pt
cp ../checkpoints_compare_rj250/fixed_s0/lprnet_avg.pt lprnet_avg.pt
md5sum lprnet_avg.pt ../checkpoints_compare_rj250/fixed_s0/lprnet_avg.pt lprnet_avg_pre_rj250.pt
echo "installed. To go back: cp lprnet_avg_pre_rj250.pt lprnet_avg.pt"
