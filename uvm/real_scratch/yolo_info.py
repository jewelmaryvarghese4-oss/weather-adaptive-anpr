from ultralytics import YOLO
m = YOLO("/mnt/c/projects/jewwww/exports/_stage/lprnet_export/detector/yolo_plate_detector_15k_best.pt")
print("task:", m.task, "| classes:", m.names)
y = getattr(m.model, "yaml", None)
print("yaml:", {k: y[k] for k in ("nc", "scales", "backbone") if k in y} if y else None)
print("args:", {k: v for k, v in (getattr(m, "overrides", {}) or {}).items() if k in ("model", "imgsz", "task")})
ck = m.ckpt if hasattr(m, "ckpt") and m.ckpt else {}
print("train_args model:", (ck.get("train_args") or {}).get("model"), "| imgsz:", (ck.get("train_args") or {}).get("imgsz"), "| version:", ck.get("version"))
try:
    m.info(verbose=False)
except Exception as e:
    print("info error", e)
