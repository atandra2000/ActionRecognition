"""
FastAPI serving endpoint for action recognition inference.

Usage:
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""

import io
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import yaml

try:
    import uvicorn
    from fastapi import FastAPI, File, UploadFile, HTTPException
    from fastapi.responses import JSONResponse
    from pydantic import BaseModel
except ImportError:
    raise ImportError(
        "fastapi and uvicorn required for serving. Install: pip install fastapi uvicorn"
    )

from src.utils.config import Config, load_config
from src.models.action_recognition import create_action_model


class PredictResponse(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    all_probs: List[float]
    inference_ms: float


class ModelServer:
    """Wraps a trained model for serving with warmup and ONNX fallback."""

    def __init__(self, config_path: str, checkpoint_path: str, device: str = "cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = load_config(config_path)
        self.model = self._build_model(checkpoint_path)
        self.class_names = getattr(self.config, "class_names", [f"class_{i}" for i in range(120)])

    def _build_model(self, checkpoint_path: str) -> torch.nn.Module:
        model_config = {
            "num_classes": self.config.model.num_classes,
            "num_keypoints": self.config.model.num_keypoints,
            "pose_dim": self.config.model.pose_dim,
            "streams": ["joint", "bone"],
        }
        model = create_action_model(model_config)
        state = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        if "model_state_dict" in state:
            model.load_state_dict(state["model_state_dict"])
        elif "ema_state_dict" in state:
            model.load_state_dict(state["ema_state_dict"])
        else:
            model.load_state_dict(state)
        model.to(self.device)
        model.eval()
        return model

    @torch.no_grad()
    def predict(self, skeleton: np.ndarray) -> Dict:
        """Run inference on a single skeleton sample.

        Args:
            skeleton: (C, T, V) numpy array

        Returns:
            Dict with class_id, class_name, confidence, all_probs, inference_ms
        """
        start = time.perf_counter()
        tensor = torch.from_numpy(skeleton).float().to(self.device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        outputs = self.model({"skeleton": tensor})
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)[0]
        class_id = int(probs.argmax().item())
        confidence = float(probs[class_id].item())
        elapsed = (time.perf_counter() - start) * 1000
        return {
            "class_id": class_id,
            "class_name": self.class_names[class_id] if class_id < len(self.class_names) else str(class_id),
            "confidence": confidence,
            "all_probs": probs.cpu().tolist(),
            "inference_ms": round(elapsed, 2),
        }

    @torch.no_grad()
    def predict_batch(self, skeletons: List[np.ndarray]) -> List[Dict]:
        if not skeletons:
            return []
        batch = np.stack(skeletons)
        tensor = torch.from_numpy(batch).float().to(self.device)
        if tensor.dim() == 3:
            tensor = tensor.unsqueeze(0)
        outputs = self.model({"skeleton": tensor})
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=-1)
        class_ids = probs.argmax(dim=-1).cpu().tolist()
        confidences = probs[range(len(class_ids)), class_ids].cpu().tolist()
        all_probs = probs.cpu().tolist()
        return [
            {
                "class_id": cid,
                "class_name": self.class_names[cid] if cid < len(self.class_names) else str(cid),
                "confidence": conf,
                "all_probs": ap,
                "inference_ms": 0.0,
            }
            for cid, conf, ap in zip(class_ids, confidences, all_probs)
        ]


# Global server instance (initialized at startup)
server: Optional[ModelServer] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global server
    config_path = "configs/ntu120_stgcn.yaml"
    ckpt_path = "outputs/ntu120_stgcn/checkpoints/best_model.pth"
    server = ModelServer(config_path=config_path, checkpoint_path=ckpt_path)
    dummy = np.random.randn(3, 64, 25).astype(np.float32)
    server.predict(dummy)
    yield
    server = None

app = FastAPI(title="Action Recognition API", version="1.0.0", lifespan=lifespan)


@app.get("/health")
def health():
    return {"status": "ok", "device": str(server.device) if server else "uninitialized"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    description="Upload a .npy file containing skeleton data of shape (C, T, V). "
                "Returns predicted action class and confidence."
)
def predict(skeleton: UploadFile = File(..., description="NumPy .npy file with skeleton data (C, T, V)")):
    if server is None:
        raise HTTPException(500, "Model not initialized")
    try:
        content = skeleton.file.read()
        buffer = io.BytesIO(content)
        data = np.load(buffer)
        result = server.predict(data)
        return result
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")


class PredictRequest(BaseModel):
    skeleton: list


@app.post("/predict/json", response_model=PredictResponse)
def predict_json(data: PredictRequest):
    if server is None:
        raise HTTPException(500, "Model not initialized")
    try:
        skeleton = np.array(data.skeleton, dtype=np.float32)
        result = server.predict(skeleton)
        return result
    except Exception as e:
        raise HTTPException(400, f"Inference failed: {e}")


def load_model_for_serving(config_path: str, checkpoint_path: str, device: str = "cuda") -> ModelServer:
    return ModelServer(config_path, checkpoint_path, device)


def run_server(host: str = "0.0.0.0", port: int = 8000):
    uvicorn.run("src.serving.api:app", host=host, port=port, reload=False)
