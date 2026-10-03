# Cable Anomaly Detection

Healthy-only ViT fine-tuning + context-level anomaly matching for the MVTec Cable dataset.

## Dataset layout

Place the dataset directly here:

```text
data/cable/
├── train/
│   └── good/
├── test/
│   ├── good/
│   ├── bent_wire/
│   ├── cable_swap/
│   ├── combined/
│   ├── cut_inner_insulation/
│   ├── cut_outer_insulation/
│   ├── missing_cable/
│   ├── missing_wire/
│   └── poke_insulation/
├── ground_truth/
├── license.txt
└── readme.txt
```

No preprocessing or `good.pt` file is required.

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python scripts/train.py
python scripts/build_reference_bank.py
python scripts/evaluate.py
```

Outputs are written to `checkpoints/` and `results/`.

The training stage uses healthy images from `data/cable/train/good/` and creates random 96x96 contexts internally.
