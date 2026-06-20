"""
Export trained model to ONNX and optionally compile with TensorRT.

Usage:
    python scripts/export_onnx.py --checkpoint outputs/ntu120_stgcn/checkpoints/best_model.pth
    python scripts/export_onnx.py --checkpoint ... --trt
"""

import argparse
from pathlib import Path

import torch
import numpy as np

from src.utils.config import load_config
from src.models.action_recognition import create_action_model


def export_onnx(
    checkpoint_path: str,
    config_path: str = "configs/ntu120_stgcn.yaml",
    output_path: str = None,
    num_keypoints: int = 25,
    num_frames: int = 64,
    pose_dim: int = 3,
    opset_version: int = 17,
):
    """Export PyTorch model to ONNX format.

    Args:
        checkpoint_path: Path to trained checkpoint (.pth)
        config_path: Path to config YAML
        output_path: Output .onnx path (auto-derived if None)
        num_keypoints: Number of skeleton keypoints
        num_frames: Number of input frames
        pose_dim: Skeleton coordinate dimension (2 or 3)
        opset_version: ONNX opset version
    """
    config = load_config(config_path)
    model_config = {
        "num_classes": config.model.num_classes,
        "num_keypoints": num_keypoints,
        "pose_dim": pose_dim,
        "streams": ["joint"],
    }
    model = create_action_model(model_config)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if "model_state_dict" in state:
        model.load_state_dict(state["model_state_dict"])
    elif "ema_state_dict" in state:
        model.load_state_dict(state["ema_state_dict"])
    else:
        model.load_state_dict(state)
    model.eval()

    if output_path is None:
        output_path = str(Path(checkpoint_path).with_suffix(".onnx"))

    # Create dummy input: (1, C, T, V)
    dummy_input = torch.randn(1, pose_dim, num_frames, num_keypoints)

    # Export
    torch.onnx.export(
        model,
        {"skeleton": dummy_input},
        output_path,
        input_names=["skeleton"],
        output_names=["logits", "probabilities"],
        dynamic_axes={
            "skeleton": {0: "batch_size"},
            "logits": {0: "batch_size"},
            "probabilities": {0: "batch_size"},
        },
        opset_version=opset_version,
        do_constant_folding=True,
    )
    print(f"ONNX model exported to: {output_path}")


def export_tensorrt(onnx_path: str, output_path: str = None, fp16: bool = True):
    """Compile ONNX model to TensorRT engine.

    Args:
        onnx_path: Path to .onnx file
        output_path: Output .engine path (auto-derived if None)
        fp16: Enable FP16 inference
    """
    try:
        import tensorrt as trt
    except ImportError:
        print("TensorRT not installed. Install with: pip install tensorrt")
        return

    if output_path is None:
        output_path = str(Path(onnx_path).with_suffix(".engine"))

    logger = trt.Logger(trt.Logger.INFO)
    builder = trt.Builder(logger)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, logger)

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for e in range(parser.num_errors):
                print(parser.get_error(e))
            raise RuntimeError("ONNX parse failed")

    config = builder.create_builder_config()
    if fp16 and builder.platform_has_fast_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)

    serialized = builder.build_serialized_network(network, config)
    with open(output_path, "wb") as f:
        f.write(serialized)
    print(f"TensorRT engine saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX / TensorRT")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to .pth checkpoint")
    parser.add_argument("--config", type=str, default="configs/ntu120_stgcn.yaml")
    parser.add_argument("--output", type=str, default=None, help="Output path")
    parser.add_argument("--trt", action="store_true", help="Also compile to TensorRT")
    parser.add_argument("--fp16", action="store_true", help="Use FP16 for TensorRT")
    parser.add_argument("--num-frames", type=int, default=64, help="Number of temporal frames")
    args = parser.parse_args()

    export_onnx(args.checkpoint, args.config, args.output, num_frames=args.num_frames)
    if args.trt:
        onnx_path = args.output or str(Path(args.checkpoint).with_suffix(".onnx"))
        export_tensorrt(onnx_path, fp16=args.fp16)
