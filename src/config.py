"""Paths and settings used by every part of the pipeline."""
import os
from pathlib import Path

ROOT = Path(os.environ.get("APP_ROOT", Path(__file__).resolve().parent.parent))

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
NEW_BATCH_DIR = DATA_DIR / "new_batch"  # held-out images used to try retraining

MODELS_DIR = Path(os.environ.get("MODELS_DIR", ROOT / "models"))
MODEL_PATH = MODELS_DIR / "pneumonia_mobilenetv2.h5"
METADATA_PATH = MODELS_DIR / "metadata.json"
VERSIONS_DIR = MODELS_DIR / "versions"

# uploaded images and the sqlite database live here (a docker volume in production)
STORAGE_DIR = Path(os.environ.get("STORAGE_DIR", ROOT / "storage"))
UPLOAD_DIR = STORAGE_DIR / "uploads"
DB_PATH = STORAGE_DIR / "pipeline.db"

INSIGHTS_PATH = DATA_DIR / "insights.json"

IMG_SIZE = 224
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]
SEED = 42

# retraining starts on its own once this many new labelled images are waiting
AUTO_RETRAIN_THRESHOLD = int(os.environ.get("AUTO_RETRAIN_THRESHOLD", 150))
AUTO_RETRAIN = os.environ.get("AUTO_RETRAIN", "1") == "1"
