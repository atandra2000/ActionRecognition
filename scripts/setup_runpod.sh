#!/bin/bash
# =============================================================================
# RunPod A100 80GB SXM Setup Script
# Action Recognition Training — Single GPU Optimized
# =============================================================================
# Usage: bash scripts/setup_runpod.sh
#
# This script:
#   1. Detects RunPod environment (A100, CUDA 12.x)
#   2. Installs PyTorch 2.5+ with CUDA 12.4 support
#   3. Installs all project dependencies
#   4. Sets up workspace directories
#   5. Verifies GPU and prints optimization summary
# =============================================================================

set -e

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Action Recognition — RunPod A100 80GB SXM Setup           ║"
echo "╚══════════════════════════════════════════════════════════════╝"

# ── Environment Detection ──────────────────────────────────────────────────

echo ""
echo "[1/6] Detecting environment..."

if command -v nvidia-smi &> /dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    CUDA_DRIVER=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
    echo "  GPU:        $GPU_NAME"
    echo "  Memory:     $GPU_MEM"
    echo "  CUDA Driver: $CUDA_DRIVER"
else
    echo "  ERROR: No NVIDIA GPU detected. This script requires an A100 GPU."
    exit 1
fi

# ── Python Environment ─────────────────────────────────────────────────────

echo ""
echo "[2/6] Setting up Python environment..."

# RunPod typically has conda or system Python
if command -v conda &> /dev/null; then
    echo "  Using conda environment..."
    if ! conda env list | grep -q "action_rec"; then
        conda create -n action_rec python=3.11 -y
    fi
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate action_rec
else
    echo "  Using venv..."
    if [ ! -d "venv" ]; then
        python3 -m venv venv
    fi
    source venv/bin/activate
fi

pip install --upgrade pip setuptools wheel

# ── PyTorch Installation ───────────────────────────────────────────────────

echo ""
echo "[3/6] Installing PyTorch with CUDA 12.4 support..."

# PyTorch 2.5+ with CUDA 12.4 — required for torch.compile + fused AdamW on A100
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Verify PyTorch CUDA
python3 -c "
import torch
print(f'  PyTorch:     {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
print(f'  CUDA version:   {torch.version.cuda}')
print(f'  GPU count:      {torch.cuda.device_count()}')
print(f'  GPU name:       {torch.cuda.get_device_name(0)}')
print(f'  GPU memory:     {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
print(f'  Compute cap:    {torch.cuda.get_device_capability(0)}')
print(f'  BF16 support:   {torch.cuda.is_bf16_supported()}')
print(f'  torch.compile:  {\"available\" if hasattr(torch, \"compile\") else \"NOT available\"}')
"

# ── Project Dependencies ────────────────────────────────────────────────────

echo ""
echo "[4/6] Installing project dependencies..."

pip install -r requirements/requirements.txt

# Install serving deps (optional but useful for RunPod serverless)
pip install fastapi uvicorn pydantic 2>/dev/null || true

# ── Workspace Setup ─────────────────────────────────────────────────────────

echo ""
echo "[5/6] Setting up workspace directories..."

mkdir -p /data/ntu_rgbd120          # Dataset mount point (RunPod network volume)
mkdir -p outputs/ntu120_a100/logs
mkdir -p outputs/ntu120_a100/checkpoints
mkdir -p outputs/ntu120_a100/tensorboard
mkdir -p outputs/ntu120_a100/visualizations

# ── GPU Optimization Verification ──────────────────────────────────────────

echo ""
echo "[6/6] Verifying A100 optimizations..."

python3 -c "
import torch

# Check all A100 features
checks = {
    'BF16 supported': torch.cuda.is_bf16_supported(),
    'TF32 available': hasattr(torch.backends.cuda.matmul, 'allow_tf32'),
    'Flash SDPA': hasattr(torch.backends.cuda, 'enable_flash_sdp'),
    'Mem-efficient SDPA': hasattr(torch.backends.cuda, 'enable_mem_efficient_sdp'),
    'torch.compile': hasattr(torch, 'compile'),
    'Fused AdamW': hasattr(torch.optim.AdamW, '__init__') and 'fused' in torch.optim.AdamW.__init__.__code__.co_varnames,
    'cuDNN benchmark': hasattr(torch.backends.cudnn, 'benchmark'),
}

print('  A100 Feature Support:')
all_ok = True
for feature, supported in checks.items():
    status = '✓' if supported else '✗'
    if not supported:
        all_ok = False
    print(f'    {status} {feature}')

if all_ok:
    print('  All A100 optimizations available!')
else:
    print('  WARNING: Some optimizations unavailable. Check PyTorch version (need 2.5+).')
"

# ── Summary ─────────────────────────────────────────────────────────────────

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║   Setup Complete!                                            ║"
echo "╠══════════════════════════════════════════════════════════════╣"
echo "║                                                              ║"
echo "║   To start training:                                         ║"
echo "║                                                              ║"
echo "║   # 1. Upload/preprocess dataset to /data/ntu_rgbd120       ║"
echo "║   python preprocess_ntu_data.py --data_root /data/ntu_rgbd120║"
echo "║                                                              ║"
echo "║   # 2. Start A100-optimized training:                        ║"
echo "║   python src/training/train.py --config configs/ntu120_a100.yaml║"
echo "║                                                              ║"
echo "║   # 3. Monitor with TensorBoard:                             ║"
echo "║   tensorboard --logdir outputs/ntu120_a100/tensorboard       ║"
echo "║                                                              ║"
echo "║   # 4. Resume from checkpoint:                               ║"
echo "║   python src/training/train.py --config configs/ntu120_a100.yaml \\║"
echo "║       --resume outputs/ntu120_a100/checkpoints/epoch_XX.pth   ║"
echo "║                                                              ║"
echo "║   Expected throughput: 2-3x vs baseline config               ║"
echo "║   Expected GPU memory usage: ~6-8 GB / 80 GB                 ║"
echo "║                                                              ║"
echo "╚══════════════════════════════════════════════════════════════╝"
