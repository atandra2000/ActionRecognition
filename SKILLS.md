# SKILLS.md — ActionRecognition

> Skills for the skeleton-based action recognition project. Pair with
> `Vision/ActionRecognition/AGENTS.md`.

---

## Skill 1: Train the two-stream ST-GCN

```bash
cd Vision/ActionRecognition
python src/training/train.py --config configs/ntu120_stgcn.yaml
```

For A100 with all optimizations:
```bash
python src/training/train.py --config configs/ntu120_a100.yaml
```

## Skill 2: Export to ONNX

```bash
python scripts/export_onnx.py --checkpoint checkpoints/stgcn_best.pt \
  --output exported/action_recognition.onnx --dynamic_batch
```

Dynamic batch axis is **required** for the FastAPI serving.

## Skill 3: Convert ONNX to TensorRT

```bash
trtexec --onnx=exported/action_recognition.onnx \
  --saveEngine=exported/action_recognition.engine \
  --fp16 --workspace=4096 \
  --minShapes=input:1x64x25x3 \
  --optShapes=input:16x64x25x3 \
  --maxShapes=input:64x64x25x3
```

INT8 calibration for the production build:
```bash
trtexec --onnx=exported/action_recognition.onnx \
  --saveEngine=exported/action_recognition_int8.engine \
  --int8 --calib=calibration_cache.bin
```

## Skill 4: Serve with FastAPI

```bash
uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /predict` — accepts `{video_path, fps}` returns `{action_id, action_name, confidence, latency_ms}`.
- `GET /health` — liveness probe.

Target latency: **~33 ms / batch-1 inference (30 FPS)** on RTX 3090.

## Skill 5: Diagnose slow training

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| <5 it/s on RTX 3090 | cudnn not benchmarking | `--cudnn_benchmark` |
| OOM at batch 64 | AMP off | enable AMP (BF16) |
| Loss plateau epoch 30 | LR too low | warmup 5 epochs, cosine |
| Val acc stuck at 20% | bone stream disabled | re-enable two-stream |
| Pose estimation noisy | soft-argmax sigma wrong | tune `sigma=2.0` |

## Skill 6: Add a new action class

1. Collect ≥100 samples per class (NTU-style: 25 keypoints × T frames).
2. Add the class name to `src/data/label_map.py`.
3. Update `configs/ntu120_*.yaml:model.num_classes`.
4. Re-train from scratch (no partial fine-tuning — class IDs are
   categorical, not continuous).

## Skill 7: Convert a video to skeleton input

```python
from src.inference.real_time_inference import video_to_skeleton
skeleton = video_to_skeleton("input.mp4", fps=30)   # (T, 25, 3)
```

The pose estimator runs at ~50 FPS on RTX 3090.

## Pitfalls
- **Bone stream** must be derived from joints (child − parent vectors),
  not computed independently.
- **Multi-modal fusion** is optional — turning it on adds ~30% inference
  latency. Disable if real-time is the constraint.
- **EMA decay 0.999** is tuned for 100-epoch training. Adjust for
  shorter runs (e.g. 0.99 for 30 epochs).
- **`torch.compile`** with `reduce-overhead` is incompatible with dynamic
  shapes. Use `default` for ONNX export.

