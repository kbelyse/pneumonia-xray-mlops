"""Image preprocessing shared by the notebook, the API and the retraining job."""
import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageOps

from .config import CLASS_NAMES, IMG_SIZE, SEED

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def patient_id(filename: str) -> str:
    """person1000_bacteria_2931.jpeg -> person1000, IM-0115-0001.jpeg -> IM-0115."""
    name = Path(filename).stem
    m = re.match(r"(person\d+)_", name)
    if m:
        return m.group(1)
    m = re.match(r"((?:NORMAL2-)?IM-\d+)-", name)
    if m:
        return m.group(1)
    return name


def to_xray_array(img: Image.Image, size: int = IMG_SIZE) -> np.ndarray:
    """Grayscale, fix EXIF rotation, resize to size x size. Returns uint8 (size, size)."""
    img = ImageOps.exif_transpose(img).convert("L")
    img = img.resize((size, size), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def resize_xray(src, dst, size: int = IMG_SIZE):
    with Image.open(src) as img:
        arr = to_xray_array(img, size)
    Image.fromarray(arr).save(dst, "JPEG", quality=92)


def bytes_to_model_input(data: bytes) -> np.ndarray:
    """Raw upload bytes -> float32 (1, 224, 224, 3) in 0..255, the format the model expects."""
    with Image.open(io.BytesIO(data)) as img:
        arr = to_xray_array(img)
    rgb = np.repeat(arr[..., None], 3, axis=-1).astype("float32")
    return rgb[None]


def is_valid_image(data: bytes) -> bool:
    try:
        with Image.open(io.BytesIO(data)) as img:
            img.verify()
        return True
    except Exception:
        return False


def list_images(folder) -> pd.DataFrame:
    """One row per image in folder/NORMAL and folder/PNEUMONIA."""
    rows = []
    for y, label in enumerate(CLASS_NAMES):
        for p in sorted(Path(folder, label).glob("*")):
            if p.suffix.lower() in IMAGE_EXTS:
                rows.append({"path": str(p), "label": label, "y": y, "patient": patient_id(p.name)})
    return pd.DataFrame(rows)


def patient_split(df: pd.DataFrame, val_frac: float = 0.15, seed: int = SEED):
    """Split by patient so the same child never appears in both train and validation."""
    rng = np.random.default_rng(seed)
    val_parts = []
    for label, group in df.groupby("label"):
        patients = np.array(group["patient"].unique())
        rng.shuffle(patients)
        n_val = int(round(len(patients) * val_frac))
        val_parts.append(group[group["patient"].isin(patients[:n_val])])
    val = pd.concat(val_parts)
    train = df.drop(val.index)
    return train.reset_index(drop=True), val.reset_index(drop=True)


def class_weights(y) -> dict:
    y = np.asarray(y)
    counts = np.bincount(y, minlength=2)
    return {i: len(y) / (2 * c) for i, c in enumerate(counts)}


# ---------- tf.data ----------

def _augmenter():
    import tensorflow as tf
    from tensorflow import keras
    # small changes only: X-rays are always taken upright, and flipping would put the heart on the wrong side
    return keras.Sequential([
        keras.layers.RandomRotation(0.03, seed=SEED),
        keras.layers.RandomZoom(0.1, seed=SEED),
        keras.layers.RandomTranslation(0.05, 0.05, seed=SEED),
        keras.layers.RandomContrast(0.15, seed=SEED),
    ], name="augment")


def make_dataset(paths, labels, batch_size: int = 32, training: bool = False, augment: bool = False):
    import tensorflow as tf

    def load(path, y):
        img = tf.io.decode_image(tf.io.read_file(path), channels=1, expand_animations=False)
        img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
        img = tf.image.grayscale_to_rgb(img)
        return img, tf.cast(y, tf.float32)

    ds = tf.data.Dataset.from_tensor_slices((list(map(str, paths)), list(labels)))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(load, num_parallel_calls=tf.data.AUTOTUNE)
    if not training:
        ds = ds.cache()
    ds = ds.batch(batch_size)
    if augment:
        aug = _augmenter()
        ds = ds.map(lambda x, y: (aug(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)
