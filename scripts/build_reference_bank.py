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
from cable_anomaly.data.contexts import extract_grid_contexts
from cable_anomaly.matcher.context_matcher import extract_features

model = create_vit(MODEL_NAME, TRAIN_FROM_BLOCK)
state = torch.load(CHECKPOINT_DIR / "vit_small_cable_96_finetuned.pt", map_location="cpu", weights_only=True)
model.load_state_dict(state)
model.eval()

paths = sorted([p for p in TRAIN_GOOD.glob("*") if p.suffix.lower() in {".png",".jpg",".jpeg",".bmp",".tif",".tiff"}])
if not paths:
    raise RuntimeError(f"No healthy images found in {TRAIN_GOOD}")

middle, final = [], []
for i, path in enumerate(paths, 1):
    image = load_image(path)
    contexts = extract_grid_contexts(image, CONTEXT_SIZE)
    m, f = extract_features(model, contexts)
    middle.append(m.half())
    final.append(f.half())
    if i % 20 == 0 or i == len(paths):
        print(f"reference {i}/{len(paths)}")

torch.save(torch.stack(middle), CHECKPOINT_DIR / "reference_middle.pt")
torch.save(torch.stack(final), CHECKPOINT_DIR / "reference_final.pt")
print("reference bank saved")
