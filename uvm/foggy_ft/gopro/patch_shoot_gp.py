import os
for p in (os.path.expanduser("~/dehaze-eval/scripts/shoot_dehaze.py"), "/mnt/c/projects/jewwww/uvm/foggy_ft/shootout/shoot_dehaze.py"):
    s = open(p).read()
    if "uvm_gp" in s:
        print("already:", p); continue
    a = 'elif a.model in ("uvm_old", "uvm_ft2", "uvm_ponmudi", "uvm_gf", "uvm_pa"):'
    c = '"uvm_pa": "checkpoints_ponmudi_plate/uvmnet_combined_final.pt"}[a.model]'
    assert s.count(a) == 1 and s.count(c) == 1, p
    s = s.replace(a, 'elif a.model in ("uvm_old", "uvm_ft2", "uvm_ponmudi", "uvm_gf", "uvm_pa", "uvm_gp"):').replace(c, '"uvm_pa": "checkpoints_ponmudi_plate/uvmnet_combined_final.pt", "uvm_gp": "checkpoints_ponmudi_gopro/uvmnet_combined_final.pt"}[a.model]')
    open(p, "w").write(s); print("patched:", p)
