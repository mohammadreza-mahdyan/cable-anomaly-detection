from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = PROJECT_ROOT / "data" / "cable"
TRAIN_GOOD = DATA_ROOT / "train" / "good"
TEST_ROOT = DATA_ROOT / "test"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
RESULTS_DIR = PROJECT_ROOT / "results"

IMAGE_SIZE = 224
CONTEXT_SIZE = 96
MODEL_NAME = "vit_small_patch16_224"

TRAIN_FROM_BLOCK = 6
BATCH_SIZE = 4
EPOCHS = 8
LR = 1e-5
WEIGHT_DECAY = 1e-4
TEMPERATURE = 0.07

REF_NUM = 16
TOP_K = 10
ALPHA = 0.5
BETA = 0.5
