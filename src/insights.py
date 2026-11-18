"""Dataset statistics behind the Insights page of the app.

Run once after data acquisition (the notebook does this):
    python -m src.insights
"""
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from .config import CLASS_NAMES, DATA_DIR, INSIGHTS_PATH, RAW_DIR, TEST_DIR, TRAIN_DIR
from .preprocessing import list_images

INSIGHTS_IMG_DIR = DATA_DIR / "insights"


def _raw_stats(path):
    with Image.open(path) as img:
        w, h = img.size
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    # centre crop is roughly where the lungs are, the borders are mostly black or text labels
    ch, cw = arr.shape[0] // 4, arr.shape[1] // 4
    centre = arr[ch:-ch, cw:-cw]
    return {"width": w, "height": h, "brightness": float(arr.mean()), "contrast": float(arr.std()),
            "centre_brightness": float(centre.mean())}


def raw_image_table(raw_root=RAW_DIR / "chest_xray") -> pd.DataFrame:
    rows = []
    for split in ("train", "val", "test"):
        for label in CLASS_NAMES:
            for p in sorted((raw_root / split / label).glob("*.jpeg")):
                subtype = "normal"
                if "bacteria" in p.name:
                    subtype = "bacterial"
                elif "virus" in p.name:
                    subtype = "viral"
                rows.append({"path": str(p), "split": "test" if split == "test" else "train",
                             "label": label, "subtype": subtype})
    df = pd.DataFrame(rows)
    with ProcessPoolExecutor() as ex:
        stats = list(ex.map(_raw_stats, df["path"], chunksize=64))
    return pd.concat([df, pd.DataFrame(stats)], axis=1)


def _hist(values, bins):
    counts, edges = np.histogram(values, bins=bins)
    return {"counts": counts.tolist(), "edges": [round(float(e), 2) for e in edges]}


def mean_images(folder=TRAIN_DIR):
    """Average 224px image per class, plus the difference pneumonia minus normal."""
    INSIGHTS_IMG_DIR.mkdir(parents=True, exist_ok=True)
    df = list_images(folder)
    means = {}
    for label in CLASS_NAMES:
        acc = np.zeros((224, 224), np.float64)
        paths = df[df.label == label].path
        for p in paths:
            acc += np.asarray(Image.open(p), dtype=np.float64)
        means[label] = acc / len(paths)
        Image.fromarray(means[label].astype(np.uint8)).save(INSIGHTS_IMG_DIR / f"mean_{label.lower()}.png")
    diff = means["PNEUMONIA"] - means["NORMAL"]
    # red where pneumonia images are brighter, blue where darker
    scale = np.abs(diff).max()
    d = np.clip(diff / scale, -1, 1)
    rgb = np.stack([np.where(d > 0, 255, 255 * (1 + d)), 255 * (1 - np.abs(d)), np.where(d < 0, 255, 255 * (1 - d))], -1)
    Image.fromarray(rgb.astype(np.uint8)).save(INSIGHTS_IMG_DIR / "mean_difference.png")
    return diff


def build(df: pd.DataFrame = None) -> dict:
    df = raw_image_table() if df is None else df
    out = {"n_images": int(len(df))}
    out["class_counts"] = {s: df[df.split == s].label.value_counts().reindex(CLASS_NAMES).fillna(0).astype(int).to_dict()
                           for s in ("train", "test")}
    out["subtype_counts"] = df.subtype.value_counts().to_dict()
    out["features"] = {}
    for feat, bins in [("width", np.linspace(0, 3000, 31)), ("height", np.linspace(0, 2800, 29)),
                       ("brightness", np.linspace(0, 255, 35)), ("contrast", np.linspace(20, 90, 29)),
                       ("centre_brightness", np.linspace(0, 255, 35))]:
        out["features"][feat] = {label: {**_hist(df[df.label == label][feat], bins),
                                         "mean": round(float(df[df.label == label][feat].mean()), 1),
                                         "median": round(float(df[df.label == label][feat].median()), 1)}
                                 for label in CLASS_NAMES}
    df = df.assign(aspect=df.width / df.height)
    out["aspect_ratio"] = {label: round(float(df[df.label == label].aspect.median()), 3) for label in CLASS_NAMES}
    out["brightness_by_subtype"] = df.groupby("subtype").centre_brightness.median().round(1).to_dict()
    INSIGHTS_PATH.write_text(json.dumps(out, indent=1))
    return out


if __name__ == "__main__":
    table = raw_image_table()
    mean_images()
    print(json.dumps(build(table), indent=1)[:2000])
