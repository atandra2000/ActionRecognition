"""
Dataset classes for NTU RGB+D 120 action recognition
"""

import pickle
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from collections import Counter
from ..utils.config import DataConfig


class NTURGBD120Dataset(Dataset):
    """NTU RGB+D 120 Dataset for action recognition."""
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 protocol: str = 'xsub',  # 'xsub' or 'xset'
                 modal: str = 'skeleton',  # 'skeleton', 'rgb', 'depth', 'ir'
                 transform=None,
                 max_frames: int = 300,
                 num_person: int = 2,
                 debug: bool = False):
        """
        Args:
            data_root: Path to the dataset root directory
            split: 'train' or 'val'
            protocol: Evaluation protocol ('xsub' or 'xset')
            modal: Data modality to use
            transform: Transform to apply to data
            max_frames: Maximum number of frames per sample
            num_person: Maximum number of people to consider
            debug: Debug mode (load subset of data)
        """
        self.data_root = Path(data_root)
        self.split = split
        self.protocol = protocol
        self.modal = modal
        self.transform = transform
        self.max_frames = max_frames
        self.num_person = num_person
        self.debug = debug
        
        # Load data
        self.samples, self.labels = self._load_data()
        
        # Get number of classes
        self.num_classes = len(set(self.labels))
        
        print(f"Loaded {len(self.samples)} samples for {split} split")
        print(f"Number of classes: {self.num_classes}")
    
    def _get_data_file_path(self) -> Path:
        """Get the path to the preprocessed data file.
        
        Returns:
            Path to preprocessed data file
        
        Raises:
            ValueError: If protocol is invalid
        """
        if self.protocol not in ['xsub', 'xset']:
            raise ValueError(f"Invalid protocol: {self.protocol}. Must be 'xsub' or 'xset'")
        
        filename = f'ntu_rgbd120_{self.protocol}_{self.split}.pkl'
        return self.data_root / filename
    
    def _load_data(self) -> Tuple[List[str], List[int]]:
        """Load dataset samples and labels with validation."""
        data_file = self._get_data_file_path()
    
        if not data_file.exists():
            raise FileNotFoundError(
                f"Dataset file not found: {data_file}\n"
                f"Please preprocess the NTU RGB+D dataset first.\n"
                f"Run: python scripts/preprocess_ntu.py --data_root {self.data_root}"
            )
    
        with open(data_file, 'rb') as f:
            data = pickle.load(f)
    
        # Validate data structure
        self._validate_data_format(data)
    
        return data['samples'], data['labels']

    def _validate_data_format(self, data: Dict) -> None:
        """Validate loaded data format."""
        required_keys = ['samples', 'labels']
        for key in required_keys:
            if key not in data:
                raise ValueError(f"Missing required key: {key}")
    
        if len(data['samples']) != len(data['labels']):
            raise ValueError("Samples and labels length mismatch")
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data sample."""
        sample_path = self.samples[idx]
        label = self.labels[idx]
        
        # Load data based on modality
        if self.modal == 'skeleton':
            data = self._load_skeleton(sample_path)
        elif self.modal == 'rgb':
            data = self._load_rgb_video(sample_path)
        elif self.modal == 'depth':
            data = self._load_depth_video(sample_path)
        elif self.modal == 'ir':
            data = self._load_ir_video(sample_path)
        else:
            raise ValueError(f"Unknown modality: {self.modal}")
        
        # Apply transforms
        if self.transform:
            data = self.transform(data)
        
        return {
            'data': {
                'skeleton': data,
            },
            'label': torch.tensor(label, dtype=torch.long),
            'sample_id': idx
        }
    
    def _load_skeleton(self, sample_path: str) -> torch.Tensor:
        """Load skeleton data or generic data from file.
        
        Args:
            sample_path: Path to data file
        
        Returns:
            Loaded data as tensor
        
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If data format is invalid
        """
        try:
            # Try numpy first (for .npy files) with memory mapping for large files
            if sample_path.endswith('.npy'):
                data = np.load(sample_path, mmap_mode='r')
            # Try pickle (for .pkl files)
            elif sample_path.endswith('.pkl'):
                with open(sample_path, 'rb') as f:
                    data = pickle.load(f)
            else:
                # Default to numpy
                data = np.load(sample_path + '.npy', mmap_mode='r')
            
            if isinstance(data, dict):
                data = data.get('skeleton', data)
            
            if not isinstance(data, np.ndarray):
                raise ValueError(f"Expected numpy array, got {type(data)}")
            
            return torch.from_numpy(data).float()
        except FileNotFoundError as e:
            raise FileNotFoundError(f"Data file not found: {sample_path}") from e
        except Exception as e:
            raise ValueError(f"Error loading data from {sample_path}: {e}") from e
    
    def _load_rgb_video(self, sample_path: str) -> torch.Tensor:
        raise NotImplementedError(
            "RGB video loading is not yet implemented. "
            "Use modal='skeleton' for skeleton-based action recognition."
        )

    def _load_depth_video(self, sample_path: str) -> torch.Tensor:
        raise NotImplementedError(
            "Depth video loading is not yet implemented. "
            "Use modal='skeleton' for skeleton-based action recognition."
        )

    def _load_ir_video(self, sample_path: str) -> torch.Tensor:
        raise NotImplementedError(
            "IR video loading is not yet implemented. "
            "Use modal='skeleton' for skeleton-based action recognition."
        )


class SkeletonDataset(Dataset):
    """Dataset specifically for skeleton data with preprocessing."""
    
    def __init__(self,
                 data_root: str,
                 split: str = 'train',
                 protocol: str = 'xsub',
                 normalize: bool = True,
                 max_frames: int = 300,
                 sampling_strategy: str = 'uniform',
                 **kwargs):
        
        if split not in ['train', 'val', 'test']:
            raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
        
        if protocol not in ['xsub', 'xset']:
            raise ValueError(f"Invalid protocol: {protocol}. Must be 'xsub' or 'xset'")
        
        if sampling_strategy not in ['uniform', 'random', 'first']:
            raise ValueError(f"Invalid sampling strategy: {sampling_strategy}")
        
        self.data_root = Path(data_root)
        self.split = split
        self.protocol = protocol
        self.normalize = normalize
        self.max_frames = max_frames
        self.sampling_strategy = sampling_strategy
        
        # Load skeleton data
        self.samples, self.labels = self._load_skeleton_data()
        self.num_classes = len(set(self.labels))
        
        print(f"Loaded {len(self.samples)} skeleton samples for {split} split")
        print(f"Number of classes: {self.num_classes}")
    
    def _load_skeleton_data(self) -> Tuple[List[np.ndarray], List[int]]:
        """Load skeleton data from files."""
        data_file = self.data_root / f'skeleton_{self.protocol}_{self.split}.pkl'
        
        if not data_file.exists():
            raise FileNotFoundError(
                f"Skeleton data file not found: {data_file}\n"
                f"Please preprocess the skeleton data first."
            )
        
        try:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            raise ValueError(f"Error loading data from {data_file}: {e}") from e
        
        # Validate data structure
        if not isinstance(data, dict) or 'samples' not in data or 'labels' not in data:
            raise ValueError("Data file must contain 'samples' and 'labels' keys")
        
        samples = data['samples']  # List of np.ndarray
        labels = data['labels']    # List of int
        
        if len(samples) != len(labels):
            raise ValueError(f"Length mismatch: {len(samples)} samples vs {len(labels)} labels")
        
        if len(samples) == 0:
            raise ValueError("No samples found in data file")
        
        # Validate sample format
        for i, sample in enumerate(samples[:5]):  # Check first 5
            if not isinstance(sample, np.ndarray):
                raise ValueError(f"Sample {i} is not a numpy array: {type(sample)}")
            if len(sample.shape) < 2:
                raise ValueError(f"Sample {i} has invalid shape: {sample.shape}")
        
        return samples, labels
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range [0, {len(self)})")
        
        skeleton = self.samples[idx]
        label = self.labels[idx]
        
        # Ensure correct shape (C, T, V)
        if len(skeleton.shape) != 3:
            raise ValueError(f"Expected 3D skeleton (C, T, V), got shape {skeleton.shape}")
        
        # Temporal sampling
        skeleton = self._temporal_sampling(skeleton)
        
        # Normalize if required
        if self.normalize:
            skeleton = self._normalize_skeleton(skeleton)
        
        return {
            'data': {
                'skeleton': torch.from_numpy(skeleton).float(),
            },
            'label': torch.tensor(label, dtype=torch.long)
        }
    
    def _temporal_sampling(self, skeleton: np.ndarray) -> np.ndarray:
        """Sample frames to fixed length.
        
        Args:
            skeleton: Input with shape (C, T, V)
        
        Returns:
            Sampled/padded skeleton with shape (C, max_frames, V)
        """
        C, T, V = skeleton.shape
        
        if T == self.max_frames:
            return skeleton.astype(np.float32)
        elif T < self.max_frames:
            # Pad with zeros
            padding = np.zeros((C, self.max_frames - T, V), dtype=np.float32)
            return np.concatenate([skeleton, padding], axis=1)
        else:
            # Sample frames
            if self.sampling_strategy == 'uniform':
                indices = np.linspace(0, T-1, self.max_frames, dtype=int)
            elif self.sampling_strategy == 'random':
                indices = np.sort(np.random.choice(T, self.max_frames, replace=False))
            else:  # 'first'
                indices = np.arange(self.max_frames)
            
            return skeleton[:, indices, :].astype(np.float32)
    
    def _normalize_skeleton(self, skeleton: np.ndarray) -> np.ndarray:
        """Normalize skeleton coordinates preserving 3D spatial structure.
        
        Normalizes by the overall skeleton scale (max distance from center joint)
        rather than per-channel, preserving the relative positions of joints.
        
        Args:
            skeleton: Input with shape (C, T, V)
        
        Returns:
            Normalized skeleton with same shape
        
        Raises:
            ValueError: If input shape is invalid
        """
        if len(skeleton.shape) != 3:
            raise ValueError(f"Expected 3D skeleton (C, T, V), got shape {skeleton.shape}")
        
        C, T, V = skeleton.shape
        
        # Center around the center joint (joint 21 = index 20 for NTU)
        center_idx = 20 if V > 20 else 0
        center = skeleton[:, :, center_idx:center_idx+1]  # (C, T, 1)
        centered = skeleton - center  # (C, T, V)
        
        # Compute scale as max distance from center across all joints and time
        # This preserves the 3D spatial relationships
        distances = np.linalg.norm(centered, axis=0)  # (T, V)
        max_dist = np.max(distances)
        scale = max(max_dist, 1e-6)
        
        # Normalize by scale
        normalized = centered / scale
        
        return normalized.astype(np.float32)


def create_dataset(config: DataConfig, split: str = 'train') -> Dataset:
    """Factory function to create datasets.
    
    Args:
        config: DataConfig object with dataset configuration
        split: 'train', 'val', or 'test'
    
    Returns:
        Initialized dataset object
    
    Raises:
        ValueError: If dataset type is unknown or config is invalid
        FileNotFoundError: If data root doesn't exist
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}. Must be 'train', 'val', or 'test'")
    
    data_root = Path(config.data_root)
    if not data_root.exists():
        raise FileNotFoundError(f"Data root does not exist: {config.data_root}")
    
    if not hasattr(config, 'dataset') or config.dataset is None:
        raise ValueError("config.dataset is not set")
    
    if config.dataset == 'ntu_rgbd120':
        return NTURGBD120Dataset(
            data_root=config.data_root,
            split=split,
            protocol=getattr(config, 'protocol', 'xsub'),
            modal=getattr(config, 'modality', 'skeleton'),
            max_frames=getattr(config, 'max_frame_num', 300),
            debug=getattr(config, 'debug', False)
        )
    elif config.dataset == 'skeleton':
        return SkeletonDataset(
            data_root=config.data_root,
            split=split,
            protocol=getattr(config, 'protocol', 'xsub'),
            normalize=getattr(config, 'skeleton_normalization', True),
            max_frames=getattr(config, 'max_frame_num', 300),
            sampling_strategy=getattr(config, 'sampling_strategy', 'uniform')
        )
    else:
        raise ValueError(f"Unknown dataset: {config.dataset}. Choose from ['ntu_rgbd120', 'skeleton']")


def create_dataloader(dataset: Dataset, config: DataConfig, split: str = 'train') -> DataLoader:
    """Create data loader with proper configuration.
    
    Args:
        dataset: PyTorch Dataset object
        config: DataConfig object with dataloader configuration
        split: 'train', 'val', or 'test'
    
    Returns:
        DataLoader instance
    
    Raises:
        ValueError: If configuration is invalid
    """
    if split not in ['train', 'val', 'test']:
        raise ValueError(f"Invalid split: {split}")
    
    if not hasattr(config, 'batch_size') or config.batch_size < 1:
        raise ValueError("config.batch_size must be >= 1")
    
    if not hasattr(config, 'num_workers'):
        config.num_workers = 0
    
    is_train = split == 'train'
    prefetch_factor = getattr(config, 'prefetch_factor', 2)
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=is_train,
        num_workers=max(0, config.num_workers),
        pin_memory=getattr(config, 'pin_memory', True),
        drop_last=is_train,
        persistent_workers=(config.num_workers > 0),
        prefetch_factor=prefetch_factor if config.num_workers > 0 else None
    )
