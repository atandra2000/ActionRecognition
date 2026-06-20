#!/bin/bash

# Setup script for Action Recognition System

set -e

echo "Setting up Action Recognition System..."

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch (adjust for your CUDA version)
echo "Installing PyTorch..."
if command -v nvidia-smi &> /dev/null; then
    # CUDA available
    CUDA_VERSION=$(nvidia-smi | grep "CUDA Version" | sed 's/.*CUDA Version: \([0-9.]*\).*/\1/')
    echo "Detected CUDA version: $CUDA_VERSION"
    
    # Install PyTorch with CUDA support
    if [[ $CUDA_VERSION == "11.8" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    elif [[ $CUDA_VERSION == "12.1" ]]; then
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
    else
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    fi
else
    # CPU only
    echo "No CUDA detected, installing CPU version of PyTorch..."
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# Install other requirements
echo "Installing other requirements..."
pip install -r requirements/requirements.txt

# Install additional packages for development
echo "Installing development packages..."
pip install pytest black flake8 mypy pre-commit

# Setup pre-commit hooks
echo "Setting up pre-commit hooks..."
pre-commit install

# Create necessary directories
echo "Creating directories..."
mkdir -p data
mkdir -p outputs
mkdir -p outputs/logs
mkdir -p outputs/checkpoints
mkdir -p outputs/inference
mkdir -p outputs/visualizations

# Download NTU RGB+D 120 dataset (placeholder - requires manual download)
echo ""
echo "IMPORTANT: NTU RGB+D 120 Dataset Setup"
echo "======================================"
echo "To use this system, you need to download the NTU RGB+D 120 dataset:"
echo "1. Visit: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp"
echo "2. Request access to the dataset"
echo "3. Download skeleton data files:"
echo "   - nturgbd_skeletons_s001_to_s017.zip (NTU RGB+D 60)"
echo "   - nturgbd_skeletons_s018_to_s032.zip (NTU RGB+D 120 extension)"
echo "4. Extract to: ./data/ntu_rgbd120/"
echo ""

# Create data preprocessing script
echo "Creating data preprocessing script..."
cat > preprocess_ntu_data.py << 'EOF'
#!/usr/bin/env python3
"""
Preprocess NTU RGB+D 120 skeleton data from raw .skeleton files.

Reads raw NTU skeleton files and produces .pkl files expected by
NTURGBD120Dataset and SkeletonDataset.
"""

import os
import sys
import pickle
import argparse
import numpy as np
from pathlib import Path
from collections import defaultdict


def parse_skeleton_file(filepath):
    """Parse a single NTU .skeleton file.

    Returns:
        List of frames, each frame is a list of body skeletons.
        Each body skeleton is a (num_joints, 3) array of (x, y, z).
    """
    with open(filepath, 'r') as f:
        lines = f.readlines()

    num_frames = int(lines[0].strip())
    frames = []
    idx = 1

    for _ in range(num_frames):
        if idx >= len(lines):
            break
        num_bodies = int(lines[idx].strip())
        idx += 1
        frame_bodies = []

        for _ in range(num_bodies):
            if idx >= len(lines):
                break
            body_info = lines[idx].strip().split()
            idx += 1
            num_joints = int(lines[idx].strip())
            idx += 1
            joints = []

            for _ in range(num_joints):
                if idx >= len(lines):
                    break
                values = lines[idx].strip().split()
                idx += 1
                x, y, z = float(values[0]), float(values[1]), float(values[2])
                joints.append([x, y, z])

            if joints:
                frame_bodies.append(np.array(joints, dtype=np.float32))

        frames.append(frame_bodies)

    return frames


def get_subject_id(filename):
    """Extract subject ID from NTU filename.

    Filenames follow pattern: SsssCcccPpppRrrrAaaa.skeleton
    where sss is the setup ID and the subject ID is embedded in the setup.
    For NTU, subject IDs 1-40 are in setups 1-17 (NTU60),
    and 41-106 are in setups 18-32 (NTU120).
    """
    basename = os.path.basename(filename)
    parts = basename.replace('.skeleton', '').split('P')
    if len(parts) < 2:
        return 0
    setup_part = parts[0]
    setup_id = int(setup_part[1:4]) if len(setup_part) >= 4 else 0
    return setup_id


def get_camera_id(filename):
    """Extract camera ID from NTU filename."""
    basename = os.path.basename(filename)
    parts = basename.split('C')
    if len(parts) < 2:
        return 0
    cam_part = parts[1].split('P')[0]
    return int(cam_part) if cam_part.isdigit() else 0


def get_action_label(filename):
    """Extract action class label from NTU filename."""
    basename = os.path.basename(filename)
    parts = basename.split('A')
    if len(parts) < 2:
        return 0
    action_part = parts[1].split('.')[0]
    return int(action_part) - 1  # 0-indexed


def is_xsub_train(subject_id):
    """Cross-subject protocol: subjects 1,2,4,5,8,9,13,14,15,16,17,18,19,25,27,28,31,34,35,38 are training."""
    train_ids = {1, 2, 4, 5, 8, 9, 13, 14, 15, 16, 17, 18, 19, 25, 27, 28, 31, 34, 35, 38}
    return subject_id in train_ids


def is_xset_train(camera_id):
    """Cross-setup protocol: cameras 2,3 are training; camera 1 is testing."""
    return camera_id in {2, 3}


def find_skeleton_files(data_root):
    """Find all .skeleton files recursively."""
    data_root = Path(data_root)
    files = []
    for root, _, filenames in os.walk(data_root):
        for f in filenames:
            if f.endswith('.skeleton'):
                files.append(os.path.join(root, f))
    return sorted(files)


def process_skeleton_files(data_root, protocol='xsub'):
    """Process all skeleton files and split into train/val.

    Returns:
        (train_samples, train_labels), (val_samples, val_labels)
        Each sample is a (C, T, V) numpy array.
    """
    files = find_skeleton_files(data_root)
    print(f"Found {len(files)} skeleton files")

    train_samples = []
    train_labels = []
    val_samples = []
    val_labels = []

    for i, filepath in enumerate(files):
        if (i + 1) % 1000 == 0:
            print(f"  Processed {i + 1}/{len(files)} files...")

        try:
            frames = parse_skeleton_file(filepath)
        except Exception as e:
            print(f"  Warning: Failed to parse {filepath}: {e}")
            continue

        if not frames:
            continue

        subject_id = get_subject_id(filepath)
        camera_id = get_camera_id(filepath)
        action_label = get_action_label(filepath)

        if protocol == 'xsub':
            is_train = is_xsub_train(subject_id)
        elif protocol == 'xset':
            is_train = is_xset_train(camera_id)
        else:
            raise ValueError(f"Unknown protocol: {protocol}")

        # Take first body only, shape: (T, J, 3) -> (3, T, J)
        skeleton_seq = []
        for frame_bodies in frames:
            if frame_bodies:
                skeleton_seq.append(frame_bodies[0])
            else:
                skeleton_seq.append(np.zeros((25, 3), dtype=np.float32))

        if not skeleton_seq:
            continue

        skeleton = np.stack(skeleton_seq, axis=0)  # (T, J, 3)
        skeleton = skeleton.transpose(2, 0, 1)  # (3, T, J)

        if is_train:
            train_samples.append(skeleton)
            train_labels.append(action_label)
        else:
            val_samples.append(skeleton)
            val_labels.append(action_label)

    print(f"  Train: {len(train_samples)} samples")
    print(f"  Val: {len(val_samples)} samples")
    return (train_samples, train_labels), (val_samples, val_labels)


def main():
    parser = argparse.ArgumentParser(description='Preprocess NTU RGB+D 120 data')
    parser.add_argument('--data_root', type=str, required=True, help='Path to NTU RGB+D dataset root')
    parser.add_argument('--protocol', type=str, default='xsub', choices=['xsub', 'xset'],
                        help='Evaluation protocol')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory (defaults to data_root)')

    args = parser.parse_args()
    output_dir = args.output_dir or args.data_root

    print(f"Processing NTU skeleton files from: {args.data_root}")
    print(f"Protocol: {args.protocol}")

    (train_samples, train_labels), (val_samples, val_labels) = process_skeleton_files(
        args.data_root, args.protocol
    )

    # Save as .pkl files
    os.makedirs(output_dir, exist_ok=True)

    train_file = os.path.join(output_dir, f'ntu_rgbd120_{args.protocol}_train.pkl')
    val_file = os.path.join(output_dir, f'ntu_rgbd120_{args.protocol}_val.pkl')
    skel_train_file = os.path.join(output_dir, f'skeleton_{args.protocol}_train.pkl')
    skel_val_file = os.path.join(output_dir, f'skeleton_{args.protocol}_val.pkl')

    for path, samples, labels in [
        (train_file, train_samples, train_labels),
        (val_file, val_samples, val_labels),
        (skel_train_file, train_samples, train_labels),
        (skel_val_file, val_samples, val_labels),
    ]:
        with open(path, 'wb') as f:
            pickle.dump({'samples': samples, 'labels': labels}, f)
        print(f"Saved: {path}")

    print(f"\nData preprocessing complete!")
    print(f"Training samples: {len(train_samples)}")
    print(f"Validation samples: {len(val_samples)}")
    print(f"Number of classes: {len(set(train_labels + val_labels))}")
    print(f"\nYou can now start training with:")
    print(f"python src/training/train.py --config configs/ntu120_stgcn.yaml")


if __name__ == "__main__":
    main()
EOF

chmod +x preprocess_ntu_data.py

# Create training script
echo "Creating training script..."
cat > train_model.sh << 'EOF'
#!/bin/bash

# Training script for Action Recognition

set -e

# Configuration
CONFIG=${1:-"configs/ntu120_stgcn.yaml"}
RESUME=${2:-""}
GPUS=${3:-1}

echo "Starting training with config: $CONFIG"
echo "GPUs: $GPUS"

# Activate virtual environment
source venv/bin/activate

# Set CUDA visible devices
if [ $GPUS -gt 1 ]; then
    export CUDA_VISIBLE_DEVICES=0,1,2,3
    python -m torch.distributed.launch --nproc_per_node=$GPUS src/training/train.py --config $CONFIG
else
    python src/training/train.py --config $CONFIG
fi

echo "Training completed!"
EOF

chmod +x train_model.sh

# Create inference script
echo "Creating inference script..."
cat > run_inference.sh << 'EOF'
#!/bin/bash

# Inference script for Action Recognition

set -e

# Configuration
CONFIG=${1:-"configs/inference.yaml"}
MODEL=${2:-"checkpoints/best_model.pth"}
SOURCE=${3:-"0"}  # Camera index or video file

echo "Starting inference with config: $CONFIG"
echo "Model: $MODEL"
echo "Source: $SOURCE"

# Activate virtual environment
source venv/bin/activate

# Run inference
python src/inference/real_time_inference.py --config $CONFIG --model $MODEL --source $SOURCE

echo "Inference completed!"
EOF

chmod +x run_inference.sh

# Create Docker setup
echo "Creating Docker setup..."
mkdir -p docker
cat > docker/Dockerfile << 'EOF'
FROM pytorch/pytorch:2.0.0-cuda11.8-cudnn8-runtime

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    wget \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create directories
RUN mkdir -p data outputs

# Set environment variables
ENV PYTHONPATH=/app
ENV CUDA_VISIBLE_DEVICES=0

# Default command
CMD ["python", "src/inference/real_time_inference.py", "--help"]
EOF

cat > docker/docker-compose.yml << 'EOF'
version: '3.8'

services:
  action-recognition:
    build: .
    runtime: nvidia
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
    volumes:
      - ./data:/app/data
      - ./outputs:/app/outputs
      - ./configs:/app/configs
    ports:
      - "8888:8888"  # For Jupyter notebook if needed
    command: python src/inference/real_time_inference.py --config configs/inference.yaml --model checkpoints/best_model.pth --source 0
EOF

# Create cloud deployment script
echo "Creating cloud deployment script..."
cat > scripts/deploy_aws.sh << 'EOF'
#!/bin/bash

# AWS deployment script for Action Recognition System

set -e

# Configuration
INSTANCE_TYPE=${1:-"p3.2xlarge"}  # GPU instance
REGION=${2:-"us-west-2"}
KEY_NAME=${3:-"your-key-pair"}
SECURITY_GROUP=${4:-"your-security-group"}

echo "Deploying to AWS EC2..."
echo "Instance type: $INSTANCE_TYPE"
echo "Region: $REGION"

# Launch EC2 instance
INSTANCE_ID=$(aws ec2 run-instances \
    --image-id ami-0c02fb55956c7d316 \
    --instance-type $INSTANCE_TYPE \
    --key-name $KEY_NAME \
    --security-groups $SECURITY_GROUP \
    --user-data file://scripts/aws_userdata.sh \
    --query 'Instances[0].InstanceId' \
    --output text)

echo "Instance launched: $INSTANCE_ID"

# Wait for instance to be running
aws ec2 wait instance-running --instance-ids $INSTANCE_ID

# Get public IP
PUBLIC_IP=$(aws ec2 describe-instances \
    --instance-ids $INSTANCE_ID \
    --query 'Reservations[0].Instances[0].PublicIpAddress' \
    --output text)

echo "Instance IP: $PUBLIC_IP"
echo "SSH into instance: ssh -i $KEY_NAME.pem ubuntu@$PUBLIC_IP"

echo "Deployment complete!"
EOF

chmod +x scripts/deploy_aws.sh

cat > scripts/aws_userdata.sh << 'EOF'
#!/bin/bash

# User data script for AWS EC2 instance

# Update system
apt-get update
apt-get install -y docker.io git

# Install NVIDIA Docker
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-docker2

# Start Docker
systemctl start docker
systemctl enable docker

# Clone repository
cd /home/ubuntu
git clone https://github.com/yourusername/action_recognition_system.git
cd action_recognition_system

# Build and run Docker container
docker-compose -f docker/docker-compose.yml up --build -d
EOF

# Create README for setup
echo "Creating setup documentation..."
cat > SETUP_INSTRUCTIONS.md << 'EOF'
# Action Recognition System - Setup Instructions

## Quick Start

1. **Run setup script:**
   ```bash
   bash scripts/setup.sh
   ```

2. **Activate virtual environment:**
   ```bash
   source venv/bin/activate
   ```

3. **Download NTU RGB+D 120 dataset:**
   - Visit: http://rose1.ntu.edu.sg/Datasets/actionRecognition.asp
   - Request access and download skeleton data
   - Extract to: `./data/ntu_rgbd120/`

4. **Preprocess data:**
   ```bash
   python preprocess_ntu_data.py --data_root ./data/ntu_rgbd120
   ```

5. **Train model:**
   ```bash
   bash train_model.sh configs/ntu120_stgcn.yaml
   ```

6. **Run inference:**
   ```bash
   bash run_inference.sh configs/inference.yaml checkpoints/best_model.pth 0
   ```

## Docker Setup

```bash
docker-compose -f docker/docker-compose.yml up --build
```

## Cloud Deployment (AWS)

```bash
bash scripts/deploy_aws.sh p3.2xlarge us-west-2 your-key-pair your-security-group
```

## Troubleshooting

1. **CUDA out of memory:** Reduce batch size in config file
2. **Slow training:** Use multiple GPUs or reduce model complexity
3. **Poor accuracy:** Check data preprocessing and augmentation settings

## Support

For issues and questions, please check:
1. Logs in `outputs/logs/`
2. Training curves in TensorBoard
3. Model checkpoints in `outputs/checkpoints/`
EOF

echo ""
echo "Setup completed successfully!"
echo ""
echo "Next steps:"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Download NTU RGB+D 120 dataset"
echo "3. Run: python preprocess_ntu_data.py --data_root ./data/ntu_rgbd120"
echo "4. Start training: bash train_model.sh"
echo ""
echo "For detailed instructions, see: SETUP_INSTRUCTIONS.md"