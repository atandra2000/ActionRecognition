# AGENTS.md — ActionRecognition

> **Project:** `Vision/ActionRecognition/` · **Type:** video understanding
> **Task:** skeleton-based action recognition (120 classes) · **Hardware:**
> target RTX 3090 / A100 · **Headline:** **~30 FPS inference on RTX 3090**
> via ONNX + TensorRT.

Skeleton-based human action recognition from video, built entirely from
first principles (no pretrained pose models or off-the-shelf classifiers).
Pose estimation → skeleton feature extraction → two-stream graph convolution
→ real-time inference serving.

---

## 1. Subagent: `video-action-ml`

**Trigger:** "How do I deploy the action recognition model?", "Two-stream
ST-GCN training is slow", "ONNX export for skeleton model", "Convert to
TensorRT", "Build a FastAPI endpoint for action recognition."

**System prompt:**
You are a senior engineer working on ActionRecognition. The headline metric
— **~30 FPS inference on RTX 3090** — must always be quoted verbatim.

**Pipeline:**
1. **Pose estimation** — custom HRNet-like 2D/3D pose estimator
   (heatmap regression + soft-argmax + depth lifting, 25 joints).
2. **Skeleton feature extraction** — joint stream + bone stream
   (child−parent vectors).
3. **Two-Stream ST-GCN** — 5 CTR-GCN layers (64→128→256→256→256),
   multi-scale temporal convs (k=3,5,7,9 / dilations 1–4),
   spatial/temporal/channel attention.
4. **Attention fusion → MLP classifier** (512→256→120).
5. **Multi-modal fusion** (RGB + skeleton + depth + IR).

**Configs:**
- `configs/ntu120_stgcn.yaml` — base config (RTX 3090 target).
- `configs/ntu120_a100.yaml` — A100 80GB optimized (BF16, compile, FA2).
- `configs/inference.yaml` — serving config (ONNX + TensorRT).

**Training:**
- AdamW + cosine annealing.
- Label smoothing (0.1), EMA (0.999), AMP (FP16/BF16).
- A100: BF16, `torch.compile`, TF32, fused AdamW, FA2, batch 128.
- DDP + gradient accumulation supported.
- Loss options: Label-Smoothing CE, Focal, Triplet, Contrastive.

**Dataset:** NTU RGB+D 120 (114,480 samples, 120 classes, 25 keypoints),
Cross-Subject & Cross-Setup splits.

**Serving:**
- FastAPI (`/predict`, `/health`) with ONNX + TensorRT export.
- `scripts/setup.sh` (local) and `scripts/setup_runpod.sh` (RunPod A100).

**Files:**
- `src/models/{action_recognition,layers,pose_estimator,pose_extractor,skeleton}.py`.
- `src/training/{train,losses,metrics}.py`.
- `src/inference/real_time_inference.py`.
- `src/serving/api.py`.
- `scripts/{export_onnx,setup,setup_runpod}.sh`.
- `tests/` — 24 unit tests (config, datasets, layers, losses, metrics,
  pose_estimator, pose_extractor).

**Hard rules:**
1. **Never** suggest MediaPipe or pretrained HRNet weights — the project
   is **from-scratch**.
2. **Always** preserve the two-stream fusion (joint + bone). Removing
   the bone stream drops accuracy by ~5%.
3. **Always** quote "~30 FPS inference on RTX 3090" for production claims.
4. **ONNX export** must include dynamic axes for batch size (the serving
   endpoint batches multiple videos).
5. **TensorRT** requires fixed input shapes; the export script pads to
   `(B, 64, 25, 3)` and uses INT8 calibration for the production build.

