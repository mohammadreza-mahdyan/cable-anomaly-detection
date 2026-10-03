import os

os.environ.pop("ALL_PROXY", None)
os.environ.pop("all_proxy", None)
os.environ.pop("HTTP_PROXY", None)
os.environ.pop("HTTPS_PROXY", None)

import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

import torch
from configs.default import *
from cable_anomaly.models.vit import create_vit
from cable_anomaly.data.dataset import load_image
from cable_anomaly.matcher.context_matcher import anomaly_score
from cable_anomaly.evaluation.metrics import binary_metrics

model = create_vit(MODEL_NAME, TRAIN_FROM_BLOCK)
state = torch.load(CHECKPOINT_DIR / "vit_small_cable_96_finetuned.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

ref_middle = torch.load(CHECKPOINT_DIR / "reference_middle.pt", map_location="cpu", weights_only=True)
ref_final = torch.load(CHECKPOINT_DIR / "reference_final.pt", map_location="cpu", weights_only=True)

scores, labels, classes = [], [], []

for class_dir in sorted(TEST_ROOT.iterdir()):
    if not class_dir.is_dir():
        continue
    is_anomaly = class_dir.name != "good"
    paths = sorted([p for p in class_dir.glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}])

    class_scores = []
    for path in paths:
        score = anomaly_score(model, load_image(path), ref_middle, ref_final, CONTEXT_SIZE)
        scores.append(score)
        labels.append(int(is_anomaly))
        class_scores.append(score)

    if class_scores:
        mean = sum(class_scores) / len(class_scores)
        print(f"{class_dir.name:25s} mean={mean:.4f} min={min(class_scores):.4f} max={max(class_scores):.4f}")

metrics = binary_metrics(labels, scores)
print(f"AUROC={metrics['auroc']:.4f}")
print(f"AUPRC ={metrics['auprc']:.4f}")
