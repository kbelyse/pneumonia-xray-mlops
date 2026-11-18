"""Model definitions, training helpers, evaluation and saving."""
import json
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss,
                             precision_score, recall_score, roc_auc_score, average_precision_score)

from .config import IMG_SIZE, METADATA_PATH, MODEL_PATH, SEED

BASE_NAME = "mobilenetv2_1.00_224"


def _keras():
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SEED)
    return tf.keras


def train_metrics():
    keras = _keras()
    return [keras.metrics.BinaryAccuracy(name="accuracy"),
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.AUC(name="auc")]


def build_baseline_cnn(dropout: float = 0.3):
    """Small CNN trained from scratch. Experiment 1, to see what transfer learning adds."""
    keras = _keras()
    L = keras.layers
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = L.Rescaling(1 / 255.0)(inputs)
    for filters in (16, 32, 64, 128):
        x = L.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = L.BatchNormalization()(x)
        x = L.MaxPooling2D()(x)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="baseline_cnn")


def build_transfer_model(dropout: float = 0.4, l2: float = 1e-4, dense_units: int = 128):
    """MobileNetV2 pretrained on ImageNet with a small classification head. Input: RGB 0..255."""
    keras = _keras()
    L = keras.layers
    base = keras.applications.MobileNetV2(include_top=False, weights="imagenet",
                                          input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    x = L.Rescaling(1 / 127.5, offset=-1)(inputs)  # same scaling MobileNetV2 was trained with
    x = base(x, training=False)
    x = L.GlobalAveragePooling2D()(x)
    x = L.Dropout(dropout)(x)
    x = L.Dense(dense_units, activation="relu", kernel_regularizer=keras.regularizers.l2(l2))(x)
    x = L.Dropout(dropout / 2)(x)
    outputs = L.Dense(1, activation="sigmoid")(x)
    return keras.Model(inputs, outputs, name="pneumonia_mobilenetv2")


def unfreeze_top(model, n_layers: int = 40):
    """Let the last n layers of MobileNetV2 learn. BatchNorm stays frozen so small batches do not wreck it."""
    keras = _keras()
    base = model.get_layer(BASE_NAME)
    base.trainable = True
    for layer in base.layers[:-n_layers]:
        layer.trainable = False
    for layer in base.layers:
        if isinstance(layer, keras.layers.BatchNormalization):
            layer.trainable = False
    return model


def compile_model(model, lr: float = 1e-3):
    keras = _keras()
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=train_metrics())
    return model


def callbacks(log_dir=None, patience: int = 4, checkpoint=None):
    keras = _keras()
    cbs = [keras.callbacks.EarlyStopping(monitor="val_loss", patience=patience, restore_best_weights=True),
           keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7)]
    if log_dir:
        cbs.append(keras.callbacks.TensorBoard(log_dir=str(log_dir)))
    if checkpoint:
        cbs.append(keras.callbacks.ModelCheckpoint(str(checkpoint), monitor="val_loss", save_best_only=True))
    return cbs


def predict_proba(model, ds) -> np.ndarray:
    return model.predict(ds, verbose=0).ravel()


def evaluate(y_true, prob, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    prob = np.clip(np.asarray(prob), 1e-7, 1 - 1e-7)
    y_pred = (prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "loss": float(log_loss(y_true, prob, labels=[0, 1])),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)) if tn + fp else 0.0,
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, prob)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, prob)) if len(set(y_true)) > 1 else None,
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
        "n": int(len(y_true)),
    }


def best_threshold(y_true, prob) -> float:
    """Threshold with the best F1 on validation data."""
    grid = np.linspace(0.05, 0.95, 91)
    scores = [f1_score(y_true, (prob >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def save_model(model, metadata: dict, path: Path = MODEL_PATH, metadata_path: Path = METADATA_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.h5")
    model.save(tmp)
    tmp.replace(path)  # atomic swap so the API never reads a half-written file
    metadata = {**metadata, "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp_meta = metadata_path.with_suffix(".tmp.json")
    tmp_meta.write_text(json.dumps(metadata, indent=2))
    tmp_meta.replace(metadata_path)


def load_model(path: Path = MODEL_PATH):
    keras = _keras()
    return keras.models.load_model(path, compile=False)


def load_metadata(path: Path = METADATA_PATH) -> dict:
    return json.loads(Path(path).read_text())
