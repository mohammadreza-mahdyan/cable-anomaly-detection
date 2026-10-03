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
from torch.utils.data import DataLoader
from configs.default import *
from cable_anomaly.data.dataset import HealthyContextDataset
from cable_anomaly.models.vit import create_vit, trainable_cls
from cable_anomaly.models.finetune import ProjectionHead, info_nce
from cable_anomaly.utils.seed import seed_everything

seed_everything(42)
CHECKPOINT_DIR.mkdir(exist_ok=True)

dataset = HealthyContextDataset(TRAIN_GOOD, CONTEXT_SIZE)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

if len(dataset) == 0:
    raise RuntimeError(f"No healthy images found in {TRAIN_GOOD}")

model = create_vit(MODEL_NAME, TRAIN_FROM_BLOCK)
head = ProjectionHead()
optimizer = torch.optim.AdamW(
    list(filter(lambda p: p.requires_grad, model.parameters())) + list(head.parameters()),
    lr=LR, weight_decay=WEIGHT_DECAY
)

model.train()
head.train()

for epoch in range(EPOCHS):
    losses = []
    for x1, x2 in loader:
        z1 = head(trainable_cls(model, x1, TRAIN_FROM_BLOCK))
        z2 = head(trainable_cls(model, x2, TRAIN_FROM_BLOCK))
        loss = info_nce(z1, z2, TEMPERATURE)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    print(f"epoch {epoch+1}/{EPOCHS}  loss={sum(losses)/len(losses):.4f}")

torch.save(model.state_dict(), CHECKPOINT_DIR / "vit_small_cable_96_finetuned.pt")
print("saved:", CHECKPOINT_DIR / "vit_small_cable_96_finetuned.pt")
