"""Retraining job.

Steps: take the uploaded images that were not used yet, preprocess them, mix them with a
sample of the original training images (so the model does not forget what it knew), fine-tune
the current saved model, compare old and new on the held-out test set, and only replace the
live model if the new one is not worse.

    python -m src.retrain            # run once from the terminal
"""
import json
import shutil
import traceback

import numpy as np
import pandas as pd

from . import database as db
from .config import (CLASS_NAMES, METADATA_PATH, MODEL_PATH, SEED, STORAGE_DIR, TEST_DIR,
                     TRAIN_DIR, VERSIONS_DIR)
from .model import (callbacks, compile_model, evaluate, load_metadata, load_model, predict_proba,
                    save_model, unfreeze_top)
from .preprocessing import list_images, make_dataset, patient_split, resize_xray, class_weights

REPLAY_PER_CLASS = 300
EPOCHS = 4
LR = 1e-5
F1_TOLERANCE = 0.01  # new model may lose at most this much test F1 and still be promoted
MIN_NEW_IMAGES = 10


def preprocess_uploads(uploads, out_dir):
    """Resize uploaded images to 224px grayscale like the training data. Broken files are skipped."""
    rows = []
    for u in uploads:
        target = out_dir / u["label"] / f"upload_{u['id']}.jpg"
        target.parent.mkdir(parents=True, exist_ok=True)
        try:
            resize_xray(u["stored_path"], target)
        except Exception:
            continue
        rows.append({"path": str(target), "label": u["label"], "y": CLASS_NAMES.index(u["label"]),
                     "patient": f"upload_{u['id']}", "upload_id": u["id"]})
    return pd.DataFrame(rows)


def replay_sample(n_per_class=REPLAY_PER_CLASS, seed=SEED):
    df = list_images(TRAIN_DIR)
    return (df.groupby("label", group_keys=False)
              .apply(lambda g: g.sample(min(n_per_class, len(g)), random_state=seed))
              .reset_index(drop=True))


def run(trigger="manual", run_id=None, epochs=EPOCHS):
    run_id = run_id or db.start_run(trigger)
    if run_id is None:
        return {"status": "busy", "message": "a retraining run is already going"}
    try:
        return _run(run_id, epochs)
    except Exception as e:
        db.update_run(run_id, status="failed", finished_at=db.now(), message=f"{e}\n{traceback.format_exc()[-800:]}")
        raise


def _run(run_id, epochs):
    uploads = db.pending_uploads()
    if len(uploads) < MIN_NEW_IMAGES:
        db.update_run(run_id, status="skipped", finished_at=db.now(), n_new=len(uploads),
                      message=f"need at least {MIN_NEW_IMAGES} new images, have {len(uploads)}")
        return {"status": "skipped", "n_new": len(uploads)}

    db.update_run(run_id, step="preprocessing uploaded images")
    work_dir = STORAGE_DIR / "retrain_work" / str(run_id)
    new = preprocess_uploads(uploads, work_dir)
    replay = replay_sample()
    data = pd.concat([new, replay], ignore_index=True)
    train, val = patient_split(data, val_frac=0.15)
    db.update_run(run_id, n_new=len(new), n_replay=len(replay))

    db.update_run(run_id, step="loading current model as the pretrained starting point")
    old_meta = load_metadata()
    model = load_model(MODEL_PATH)
    test = list_images(TEST_DIR)
    test_ds = make_dataset(test.path, test.y)
    old_metrics = evaluate(test.y, predict_proba(model, test_ds), old_meta.get("threshold", 0.5))

    db.update_run(run_id, step=f"fine-tuning for up to {epochs} epochs")
    unfreeze_top(model, n_layers=40)
    compile_model(model, lr=LR)
    history = model.fit(make_dataset(train.path, train.y, training=True, augment=True),
                        validation_data=make_dataset(val.path, val.y),
                        epochs=epochs, class_weight=class_weights(train.y),
                        callbacks=callbacks(patience=2), verbose=2)

    db.update_run(run_id, step="evaluating new model on the test set")
    new_metrics = evaluate(test.y, predict_proba(model, test_ds), old_meta.get("threshold", 0.5))
    promoted = new_metrics["f1"] >= old_metrics["f1"] - F1_TOLERANCE

    version = int(old_meta.get("version", 1)) + 1
    VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
    meta = {**old_meta, "version": version, "test_metrics": new_metrics, "retrained_from": old_meta.get("version"),
            "retrain_run": run_id, "n_new_images": len(new),
            "history": {k: [float(v) for v in vals] for k, vals in history.history.items()}}
    if promoted:
        # keep the previous model so it can be rolled back by hand
        shutil.copy(MODEL_PATH, VERSIONS_DIR / f"model_v{old_meta.get('version', 1)}.h5")
        shutil.copy(METADATA_PATH, VERSIONS_DIR / f"metadata_v{old_meta.get('version', 1)}.json")
        save_model(model, meta)
        message = f"new model v{version} is live"
    else:
        model.save(VERSIONS_DIR / f"rejected_v{version}_run{run_id}.h5")
        message = "new model was worse on the test set, kept the old one"

    db.mark_used(new["upload_id"].astype(int).tolist(), run_id)
    db.update_run(run_id, status="done", step="finished", finished_at=db.now(), old_metrics=old_metrics,
                  new_metrics=new_metrics, promoted=int(promoted), model_version=version if promoted else None,
                  message=message)
    shutil.rmtree(work_dir, ignore_errors=True)
    return {"status": "done", "promoted": promoted, "old": old_metrics, "new": new_metrics, "message": message}


if __name__ == "__main__":
    print(json.dumps(run("terminal"), indent=2, default=str))
