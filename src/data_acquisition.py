"""Download the chest X-ray dataset and make the 224px copies in data/train and data/test.

Source: Kermany et al., Mendeley Data v2, doi:10.17632/rscbjbr9sj.2 (CC BY 4.0).

    python -m src.data_acquisition
"""
import hashlib
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np

from .config import CLASS_NAMES, NEW_BATCH_DIR, RAW_DIR, SEED, TEST_DIR, TRAIN_DIR
from .preprocessing import resize_xray

URL = ("https://data.mendeley.com/public-files/datasets/rscbjbr9sj/files/"
       "f12eaf6d-6023-432f-acc9-80c9d7393433/file_downloaded")
SHA256 = "13efc055629733dbab07877f8b3c9f81097840dbcdaa326a8542322c2281ce36"
ZIP_PATH = RAW_DIR / "ChestXRay2017.zip"
EXTRACTED = RAW_DIR / "chest_xray"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if ZIP_PATH.exists() and sha256(ZIP_PATH) == SHA256:
        print("zip already downloaded")
        return
    print("downloading 1.2 GB from Mendeley, this takes a while")
    subprocess.run(["curl", "-L", "--retry", "5", "-C", "-", "-o", str(ZIP_PATH), URL], check=True)
    assert sha256(ZIP_PATH) == SHA256, "checksum does not match, delete the zip and try again"


def extract():
    if (EXTRACTED / "train").exists():
        print("already extracted")
        return
    with zipfile.ZipFile(ZIP_PATH) as z:
        members = [m for m in z.namelist() if not m.startswith("__MACOSX") and not m.endswith(".DS_Store")]
        z.extractall(RAW_DIR, members)


def make_resized_copies():
    """Resize every image once to 224x224 grayscale so training does not decode 2000px files each epoch."""
    for split, out_dir in [("train", TRAIN_DIR), ("test", TEST_DIR)]:
        for label in CLASS_NAMES:
            src = EXTRACTED / split / label
            dst = out_dir / label
            dst.mkdir(parents=True, exist_ok=True)
            files = sorted(p for p in src.iterdir() if p.suffix.lower() in {".jpeg", ".jpg", ".png"})
            for p in files:
                target = dst / (p.stem + ".jpg")
                held_out = NEW_BATCH_DIR / label / target.name
                if not target.exists() and not held_out.exists():
                    resize_xray(p, target)
            print(f"{split}/{label}: {len(files)} original images")
    # the official 16-image val folder is too small to be useful, I fold it into train
    for label in CLASS_NAMES:
        for p in (EXTRACTED / "val" / label).glob("*.jpeg"):
            target = TRAIN_DIR / label / (p.stem + ".jpg")
            if not target.exists() and not (NEW_BATCH_DIR / label / target.name).exists():
                resize_xray(p, target)


def hold_out_new_batch(n_per_class: int = 60):
    """Move a few patients out of data/train into data/new_batch.

    These play the role of new X-rays arriving after the model is deployed. I upload them in
    the app to test retraining, so they must not be part of the original training data.
    """
    from .preprocessing import list_images, patient_id
    if NEW_BATCH_DIR.exists() and any(NEW_BATCH_DIR.rglob("*.jpg")):
        print("new_batch already exists")
        return
    df = list_images(TRAIN_DIR)
    rng = np.random.default_rng(SEED)
    for label, group in df.groupby("label"):
        patients = np.array(group["patient"].unique())
        rng.shuffle(patients)
        moved = 0
        for p in patients:
            for path in group[group["patient"] == p]["path"]:
                dst = NEW_BATCH_DIR / label / Path(path).name
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(path, dst)
                moved += 1
            if moved >= n_per_class:
                break
        print(f"new_batch/{label}: {moved} images")


def main():
    download()
    extract()
    make_resized_copies()
    hold_out_new_batch()


if __name__ == "__main__":
    main()
