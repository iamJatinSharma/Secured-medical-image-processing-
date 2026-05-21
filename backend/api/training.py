"""Training API — real PyTorch training on MedMNIST datasets.

Exposes:
- GET  /training/datasets   — list supported MedMNIST datasets
- POST /training/start      — kick off training in a background thread
- GET  /training/status     — current state
- WS   /training/ws         — stream per-epoch metrics
"""
from __future__ import annotations

import asyncio
import logging
import threading
import traceback
from typing import List, Optional

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from ml.trainer import (
    EpochMetrics,
    TrainArgs,
    list_datasets,
    list_trained_models,
    train,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class TrainingConfig(BaseModel):
    # NB: keep `model` for backward compat with existing frontend types
    model: str = "resnet18"  # resnet18 | resnet50 | smallcnn
    dataset: str = "breastmnist"
    epochs: int = 5
    batch_size: int = 32
    learning_rate: float = 0.001
    image_size: int = 64
    max_images: Optional[int] = None


training_state: dict = {"status": "idle", "progress": {}}
connected_clients: List[WebSocket] = []
_training_lock = threading.Lock()


async def _broadcast(message: dict):
    disconnected: List[WebSocket] = []
    for ws in connected_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        if ws in connected_clients:
            connected_clients.remove(ws)


def _run_real_training(config: TrainingConfig, loop: asyncio.AbstractEventLoop):
    global training_state

    with _training_lock:
        training_state = {"status": "training", "progress": {}}

    def on_epoch(m: EpochMetrics):
        payload = {
            "epoch": m.epoch,
            "total_epochs": m.total_epochs,
            "train_loss": m.train_loss,
            "val_loss": m.val_loss,
            "train_acc": m.train_acc,
            "val_acc": m.val_acc,
            "percent_complete": m.percent_complete,
            "eta_seconds": m.eta_seconds,
        }
        with _training_lock:
            training_state = {"status": "training", "progress": payload}  # noqa: F841 (rebinds module global below)
            globals()["training_state"] = training_state
        asyncio.run_coroutine_threadsafe(_broadcast(payload), loop)

    args = TrainArgs(
        dataset=config.dataset,
        architecture=config.model,
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        image_size=config.image_size,
        max_train_samples=config.max_images if config.max_images and config.max_images > 0 else None,
    )

    try:
        result = train(args, on_epoch=on_epoch)
        # Drop the inference model cache so freshly trained weights are loaded next predict()
        try:
            from ml.model_registry import reset_cache
            reset_cache()
        except Exception:
            pass
        completion_msg = {
            "status": "complete",
            "model_name": result["model_name"],
            "best_val_acc": result["best_val_acc"],
            "accuracy": result["accuracy"],
            "f1_score": result["f1_score"],
            "total_time_seconds": result["total_time_seconds"],
            "labels": result["labels"],
            "architecture": result["architecture"],
        }
        with _training_lock:
            globals()["training_state"] = {"status": "complete", "progress": completion_msg}
        asyncio.run_coroutine_threadsafe(_broadcast(completion_msg), loop)
    except Exception as e:
        logger.error("Training failed: %s\n%s", e, traceback.format_exc())
        err_msg = {"status": "error", "error": str(e)}
        with _training_lock:
            globals()["training_state"] = {"status": "error", "progress": err_msg}
        asyncio.run_coroutine_threadsafe(_broadcast(err_msg), loop)


@router.get("/health")
async def health():
    return {"status": "ok", "module": "training"}


@router.get("/datasets")
async def get_datasets():
    """List MedMNIST datasets available for training."""
    return {"datasets": list_datasets()}


@router.get("/trained")
async def get_trained_models():
    """List models that have been trained and have saved weights."""
    return {"models": list_trained_models()}


@router.post("/start")
async def start_training(config: TrainingConfig):
    global training_state

    with _training_lock:
        if training_state.get("status") == "training":
            return {"error": "Training already in progress"}

    loop = asyncio.get_running_loop()
    thread = threading.Thread(
        target=_run_real_training,
        args=(config, loop),
        daemon=True,
    )
    thread.start()

    return {
        "message": "Training started",
        "config": config.model_dump(),
    }


@router.get("/status")
async def get_status():
    with _training_lock:
        return training_state


@router.websocket("/ws")
async def training_websocket(websocket: WebSocket):
    await websocket.accept()
    connected_clients.append(websocket)
    try:
        with _training_lock:
            await websocket.send_json(training_state)
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in connected_clients:
            connected_clients.remove(websocket)
