"""Single image prediction. The API keeps one Predictor per container."""
import threading
import time

import numpy as np

from .config import CLASS_NAMES, METADATA_PATH, MODEL_PATH
from .model import load_metadata, load_model
from .preprocessing import bytes_to_model_input


class Predictor:
    def __init__(self, model_path=MODEL_PATH, metadata_path=METADATA_PATH):
        self.model_path = model_path
        self.metadata_path = metadata_path
        self._lock = threading.Lock()
        # one prediction at a time, the API also runs them all on a single thread
        self._infer_lock = threading.Lock()
        self._mtime = None
        self.model = None
        self.metadata = {}
        self.reload_if_changed()

    def reload_if_changed(self):
        """After a retrain the file on disk changes. Every container picks up the new model on its next request."""
        mtime = self.metadata_path.stat().st_mtime
        if mtime == self._mtime:
            return
        with self._lock:
            if mtime == self._mtime:
                return
            model = load_model(self.model_path)
            model(np.zeros((1, 224, 224, 3), "float32"), training=False)  # warm up
            self.model, self.metadata, self._mtime = model, load_metadata(self.metadata_path), mtime

    @property
    def threshold(self) -> float:
        return float(self.metadata.get("threshold", 0.5))

    def predict_array(self, x: np.ndarray) -> float:
        self.reload_if_changed()
        with self._infer_lock:
            return float(self.model(x, training=False).numpy().ravel()[0])

    def predict_bytes(self, data: bytes) -> dict:
        start = time.perf_counter()
        prob = self.predict_array(bytes_to_model_input(data))
        label = CLASS_NAMES[int(prob >= self.threshold)]
        return {
            "label": label,
            "pneumonia_probability": round(prob, 4),
            "confidence": round(prob if label == "PNEUMONIA" else 1 - prob, 4),
            "threshold": self.threshold,
            "model_version": self.metadata.get("version"),
            "latency_ms": round((time.perf_counter() - start) * 1000, 1),
        }


def predict_file(path, predictor: Predictor = None) -> dict:
    predictor = predictor or Predictor()
    with open(path, "rb") as f:
        return predictor.predict_bytes(f.read())
