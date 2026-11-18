"""FastAPI app: prediction, uploads, retraining, model status and dataset insights.

    uvicorn src.api:app --port 8000
"""
import asyncio
import hashlib
import io
import json
import os
import socket
import threading
import time
import uuid
import zipfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List

from fastapi import BackgroundTasks, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import database as db
from . import retrain
from .config import (AUTO_RETRAIN, AUTO_RETRAIN_THRESHOLD, CLASS_NAMES, INSIGHTS_PATH, ROOT,
                     UPLOAD_DIR)
from .insights import INSIGHTS_IMG_DIR
from .preprocessing import IMAGE_EXTS, is_valid_image
from .prediction import Predictor

UI_DIR = ROOT / "ui"
STARTED = time.time()
HOST = socket.gethostname()

app = FastAPI(title="Pneumonia X-ray classifier", version="1.0")
predictor = Predictor()
# every prediction runs on this one thread. When FastAPI's 40 worker threads all called
# TensorFlow, memory kept growing under load until the container was killed
infer_pool = ThreadPoolExecutor(max_workers=1)
db.fail_stale_runs()

stats_lock = threading.Lock()
stats = {"requests": 0, "errors": 0}
recent_latency = deque(maxlen=200)  # (timestamp, ms) of the last predictions in this container


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(UI_DIR / "index.html")


@app.get("/health")
def health():
    """Uptime and basic numbers for this container."""
    lat = [ms for _, ms in recent_latency]
    return {
        "status": "ok",
        "container": HOST,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(STARTED)),
        "uptime_seconds": round(time.time() - STARTED),
        "model_version": predictor.metadata.get("version"),
        "predictions_served": stats["requests"],
        "errors": stats["errors"],
        "avg_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
        "recent_latency": [{"t": round(t), "ms": ms} for t, ms in list(recent_latency)[-50:]],
    }


@app.get("/model")
def model_info():
    predictor.reload_if_changed()
    return predictor.metadata


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    data = await file.read()
    if not is_valid_image(data):
        with stats_lock:
            stats["errors"] += 1
        raise HTTPException(400, "That file is not an image I can read")
    result = await asyncio.get_running_loop().run_in_executor(infer_pool, predictor.predict_bytes, data)
    with stats_lock:
        stats["requests"] += 1
        recent_latency.append((time.time(), result["latency_ms"]))
    result["container"] = HOST
    result["filename"] = file.filename
    return result


def _save_upload(name: str, data: bytes, label: str) -> bool:
    if Path(name).suffix.lower() not in IMAGE_EXTS or not is_valid_image(data):
        return False
    sha = hashlib.sha256(data).hexdigest()
    target = UPLOAD_DIR / label / f"{uuid.uuid4().hex[:12]}{Path(name).suffix.lower()}"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    if not db.add_upload(Path(name).name, target, label, sha):
        target.unlink()  # duplicate
        return False
    return True


def _label_from_zip_path(name: str):
    parts = [p.upper() for p in Path(name).parts]
    for label in CLASS_NAMES:
        if label in parts:
            return label
    return None


@app.post("/upload")
def upload(background: BackgroundTasks, files: List[UploadFile] = File(...), label: str = Form(None)):
    """Bulk upload. Either pick a label for all images, or send a zip with NORMAL/ and PNEUMONIA/ folders."""
    if label is not None:
        label = label.upper()
        if label not in CLASS_NAMES:
            raise HTTPException(400, f"label must be one of {CLASS_NAMES}")
    saved, skipped = {c: 0 for c in CLASS_NAMES}, 0
    for f in files:
        data = f.file.read()
        if f.filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for name in z.namelist():
                    if name.endswith("/") or "__MACOSX" in name:
                        continue
                    lab = _label_from_zip_path(name) or label
                    if lab and _save_upload(name, z.read(name), lab):
                        saved[lab] += 1
                    else:
                        skipped += 1
        elif label and _save_upload(f.filename, data, label):
            saved[label] += 1
        else:
            skipped += 1
    pending = sum(v["pending"] for v in db.upload_summary().values())
    auto = False
    if AUTO_RETRAIN and pending >= AUTO_RETRAIN_THRESHOLD:
        run_id = db.start_run("auto: upload threshold reached")
        if run_id:
            background.add_task(retrain.run, run_id=run_id)
            auto = True
    return {"saved": saved, "skipped": skipped, "pending_for_retraining": pending,
            "auto_retrain_started": auto, "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD}


@app.get("/uploads")
def uploads():
    return {"by_label": db.upload_summary(), "auto_retrain_threshold": AUTO_RETRAIN_THRESHOLD,
            "auto_retrain": AUTO_RETRAIN, "min_images": retrain.MIN_NEW_IMAGES}


@app.post("/retrain", status_code=202)
def start_retrain(background: BackgroundTasks):
    pending = len(db.pending_uploads())
    if pending < retrain.MIN_NEW_IMAGES:
        raise HTTPException(400, f"Upload at least {retrain.MIN_NEW_IMAGES} new images first ({pending} waiting)")
    run_id = db.start_run("button")
    if run_id is None:
        raise HTTPException(409, "A retraining run is already in progress")
    background.add_task(retrain.run, run_id=run_id)
    return {"run_id": run_id, "status": "running"}


@app.get("/retrain/status")
def retrain_status():
    return {"runs": db.recent_runs()}


@app.get("/insights")
def insights():
    return json.loads(INSIGHTS_PATH.read_text())


@app.get("/insights/{name}.png", include_in_schema=False)
def insight_image(name: str):
    path = INSIGHTS_IMG_DIR / f"{Path(name).name}.png"
    if not path.exists():
        raise HTTPException(404)
    return FileResponse(path)


app.mount("/static", StaticFiles(directory=UI_DIR), name="static")
