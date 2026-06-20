"""
Visualization utilities for action recognition
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple, Optional, Dict, Any
try:
    from torchviz import make_dot
except ImportError:
    make_dot = None
import torch
from dataclasses import dataclass
import warnings


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    # Skeleton drawing
    joint_radius: int = 5
    line_width: int = 2
    confidence_threshold: float = 0.3
    
    # Colors (BGR)
    line_color: Tuple[int, int, int] = (0, 255, 0)
    joint_fill_color: Tuple[int, int, int] = (255, 0, 0)
    joint_outline_color: Tuple[int, int, int] = (255, 255, 255)
    text_color: Tuple[int, int, int] = (255, 255, 255)
    background_color: Tuple[int, int, int] = (0, 0, 0)
    
    # Text rendering
    font_size: float = 0.4
    text_offset: Tuple[int, int] = (5, -5)
    
    # Plot settings
    dpi: int = 300
    figure_size_cm: Tuple[int, int] = (12, 10)
    figure_size_curves: Tuple[int, int] = (15, 5)
    figure_size_attention: Tuple[int, int] = (15, 3)
    
    # Video settings
    video_codec: str = 'mp4v'
    default_fps: int = 30


# Global config instance
_viz_config = VisualizationConfig()


def _validate_keypoints(keypoints: np.ndarray, expected_shape: Optional[Tuple[int, ...]] = None) -> None:
    """Validate keypoints array.
    
    Args:
        keypoints: Keypoints array
        expected_shape: Expected shape (num_joints, num_features)
        
    Raises:
        ValueError: If keypoints are invalid
    """
    if not isinstance(keypoints, np.ndarray):
        raise TypeError(f"keypoints must be numpy array, got {type(keypoints).__name__}")
    if keypoints.ndim < 2:
        raise ValueError(f"keypoints must be 2D+ array, got shape {keypoints.shape}")
    if keypoints.shape[1] not in (2, 3):
        raise ValueError(f"keypoints must have 2 or 3 features, got {keypoints.shape[1]}")
    if expected_shape and keypoints.shape != expected_shape:
        raise ValueError(f"Expected shape {expected_shape}, got {keypoints.shape}")


def _fig_to_array(fig: plt.Figure) -> np.ndarray:
    """Convert matplotlib figure to numpy array.
    
    Args:
        fig: Matplotlib figure object
        
    Returns:
        Image as numpy array (H, W, 3) with uint8 dtype
    """
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    return img


from src.models.skeleton import SKELETON_CONNECTIONS as NTU_SKELETON_CONNECTIONS

# Joint colors (RGB)
JOINT_COLORS = [
    (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0),
    (255, 0, 255), (0, 255, 255), (128, 0, 0), (0, 128, 0),
    (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128),
    (192, 192, 192), (128, 128, 128), (64, 0, 0), (0, 64, 0),
    (0, 0, 64), (64, 64, 0), (64, 0, 64), (0, 64, 64),
    (255, 128, 0), (255, 0, 128), (128, 255, 0), (0, 255, 128),
    (128, 0, 255)
]


def draw_skeleton(image: np.ndarray, keypoints: np.ndarray, 
                 connections: List[Tuple[int, int]] = NTU_SKELETON_CONNECTIONS,
                 confidence_threshold: Optional[float] = None,
                 config: Optional[VisualizationConfig] = None) -> np.ndarray:
    """
    Draw skeleton on image with validation.
    
    Args:
        image: Input image (H, W, C) as uint8
        keypoints: Keypoints array (N, 2) or (N, 3) where N is number of joints
        connections: List of (start_joint, end_joint) connections
        confidence_threshold: Minimum confidence to draw a joint (uses config default if None)
        config: VisualizationConfig instance (uses global if None)
    
    Returns:
        Image with skeleton drawn (H, W, C) as uint8
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError(f"image must be 3D array, got shape {image.shape}")
    
    _validate_keypoints(keypoints)
    
    config = config or _viz_config
    confidence_threshold = confidence_threshold or config.confidence_threshold
    
    image = image.copy()
    height, width = image.shape[:2]
    
    # Convert keypoints to image coordinates
    if keypoints.shape[1] == 2:  # (x, y)
        keypoints_img = keypoints.copy()
        confidences = np.ones(len(keypoints))
    else:  # (x, y, confidence)
        keypoints_img = keypoints[:, :2].copy()
        confidences = keypoints[:, 2]
    
    # Scale keypoints to image size if needed
    if keypoints_img.max() <= 1.0:
        keypoints_img[:, 0] *= width
        keypoints_img[:, 1] *= height
    
    # Draw connections
    for start_idx, end_idx in connections:
        start_idx -= 1  # Convert to 0-indexed
        end_idx -= 1
        
        if start_idx < len(keypoints_img) and end_idx < len(keypoints_img):
            # Check confidence if available
            if confidences[start_idx] < confidence_threshold or confidences[end_idx] < confidence_threshold:
                continue
            
            start_point = tuple(keypoints_img[start_idx].astype(int))
            end_point = tuple(keypoints_img[end_idx].astype(int))
            
            # Draw line
            cv2.line(image, start_point, end_point, config.line_color, config.line_width)
    
    # Draw joints
    for i, (x, y) in enumerate(keypoints_img):
        # Check confidence if available
        if confidences[i] < confidence_threshold:
            continue
        
        point = (int(x), int(y))
        color = JOINT_COLORS[i % len(JOINT_COLORS)]
        
        # Draw circle
        cv2.circle(image, point, config.joint_radius, color, -1)
        cv2.circle(image, point, config.joint_radius, config.joint_outline_color, 1)
        
        # Draw joint index
        cv2.putText(image, str(i + 1), (point[0] + config.text_offset[0], point[1] + config.text_offset[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, config.font_size, config.text_color, 1)
    
    return image


def draw_action_label(image: np.ndarray, action_name: str, 
                     confidence: float, fps: int = 0,
                     position: Tuple[int, int] = (10, 30),
                     config: Optional[VisualizationConfig] = None) -> np.ndarray:
    """
    Draw action label on image.
    
    Args:
        image: Input image
        action_name: Action class name
        confidence: Prediction confidence (0-1)
        fps: Frames per second
        position: Text position (x, y)
        config: VisualizationConfig instance (uses global if None)
    
    Returns:
        Image with action label drawn
        
    Raises:
        ValueError: If inputs are invalid
    """
    if not (0 <= confidence <= 1):
        raise ValueError(f"confidence must be in [0, 1], got {confidence}")
    
    config = config or _viz_config
    image = image.copy()
    x, y = position
    
    # Background rectangle
    text = f"{action_name} ({confidence:.2f})"
    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    
    cv2.rectangle(image, (x - 5, y - text_height - 10), 
                 (x + text_width + 5, y + 5), config.background_color, -1)
    
    # Action text
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 
               0.8, config.line_color, 2)
    
    # FPS counter
    if fps > 0:
        fps_text = f"FPS: {fps}"
        cv2.putText(image, fps_text, (x, y + 30), cv2.FONT_HERSHEY_SIMPLEX,
                   0.6, (255, 255, 0), 2)
    
    return image


def draw_confusion_matrix(cm: np.ndarray, class_names: List[str], 
                         save_path: Optional[str] = None,
                         config: Optional[VisualizationConfig] = None) -> np.ndarray:
    """
    Draw confusion matrix visualization.
    
    Args:
        cm: Confusion matrix (N, N) as integer count
        class_names: List of class names
        save_path: Path to save the visualization
        config: VisualizationConfig instance (uses global if None)
    
    Returns:
        Visualization as numpy array (H, W, 3) with uint8 dtype
        
    Raises:
        ValueError: If cm or class_names are invalid
    """
    if not isinstance(cm, np.ndarray) or cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"cm must be 2D square matrix, got shape {cm.shape}")
    if len(class_names) != cm.shape[0]:
        raise ValueError(f"class_names length {len(class_names)} != cm size {cm.shape[0]}")
    
    config = config or _viz_config
    fig, ax = plt.subplots(figsize=config.figure_size_cm)
    
    try:
        # Normalize confusion matrix with epsilon to avoid division by zero
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)  # Avoid division by zero
        cm_normalized = cm.astype('float') / row_sums
        
        # Create heatmap
        sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names, ax=ax)
        
        ax.set_title('Confusion Matrix')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        
        # Convert to numpy array
        img = _fig_to_array(fig)
        
        return img
    finally:
        plt.close(fig)


def draw_training_curves(train_loss: List[float], val_loss: List[float],
                        train_acc: List[float], val_acc: List[float],
                        save_path: Optional[str] = None,
                        config: Optional[VisualizationConfig] = None) -> np.ndarray:
    """
    Draw training curves.
    
    Args:
        train_loss: Training loss history
        val_loss: Validation loss history
        train_acc: Training accuracy history
        val_acc: Validation accuracy history
        save_path: Path to save the visualization
        config: VisualizationConfig instance (uses global if None)
    
    Returns:
        Visualization as numpy array (H, W, 3) with uint8 dtype
        
    Raises:
        ValueError: If input lists are invalid or have mismatched lengths
    """
    if not all(isinstance(x, list) for x in [train_loss, val_loss, train_acc, val_acc]):
        raise ValueError("All inputs must be lists")
    if not (len(train_loss) == len(val_loss) == len(train_acc) == len(val_acc)):
        raise ValueError(f"All lists must have same length: {len(train_loss)}, {len(val_loss)}, {len(train_acc)}, {len(val_acc)}")
    if len(train_loss) == 0:
        raise ValueError("Input lists cannot be empty")
    
    config = config or _viz_config
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=config.figure_size_curves)
    
    try:
        epochs = range(1, len(train_loss) + 1)
        
        # Loss curves
        ax1.plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
        ax1.plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
        ax1.set_title('Training and Validation Loss', fontsize=14)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Accuracy curves
        ax2.plot(epochs, train_acc, 'b-', label='Training Accuracy', linewidth=2)
        ax2.plot(epochs, val_acc, 'r-', label='Validation Accuracy', linewidth=2)
        ax2.set_title('Training and Validation Accuracy', fontsize=14)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        
        # Convert to numpy array
        img = _fig_to_array(fig)
        
        return img
    finally:
        plt.close(fig)


def draw_attention_map(attention_weights: np.ndarray, 
                      save_path: Optional[str] = None,
                      config: Optional[VisualizationConfig] = None) -> np.ndarray:
    """
    Draw attention map visualization.
    
    Args:
        attention_weights: Attention weights (T, V, V) where T=time, V=vertices
        save_path: Path to save the visualization
        config: VisualizationConfig instance (uses global if None)
    
    Returns:
        Visualization as numpy array (H, W, 3) with uint8 dtype
        
    Raises:
        ValueError: If attention_weights are invalid
    """
    if not isinstance(attention_weights, np.ndarray) or attention_weights.ndim != 3:
        raise ValueError(f"attention_weights must be 3D array, got shape {attention_weights.shape}")
    if attention_weights.shape[1] != attention_weights.shape[2]:
        raise ValueError(f"attention_weights must be (T, V, V), got shape {attention_weights.shape}")
    
    config = config or _viz_config
    T, V, _ = attention_weights.shape
    
    fig, axes = plt.subplots(1, min(T, 5), figsize=config.figure_size_attention)
    if T == 1:
        axes = [axes]
    
    try:
        for t in range(min(T, 5)):
            ax = axes[t]
            sns.heatmap(attention_weights[t], annot=False, cmap='viridis', ax=ax)
            ax.set_title(f'Timestep {t+1}')
            ax.set_xlabel('Joint')
            ax.set_ylabel('Joint')
        
        plt.tight_layout()
        
        # Save if path provided
        if save_path:
            plt.savefig(save_path, dpi=config.dpi, bbox_inches='tight')
        
        # Convert to numpy array
        img = _fig_to_array(fig)
        
        return img
    finally:
        plt.close(fig)


def create_video_visualization(frames: List[np.ndarray], 
                              skeletons: List[np.ndarray],
                              actions: List[str],
                              confidences: List[float],
                              output_path: str,
                              fps: Optional[int] = None,
                              config: Optional[VisualizationConfig] = None) -> None:
    """
    Create video visualization with skeletons and action labels.
    
    Args:
        frames: List of video frames (all same size)
        skeletons: List of skeleton keypoints (same length as frames)
        actions: List of action names (same length as frames)
        confidences: List of prediction confidences (same length as frames)
        output_path: Output video path
        fps: Video FPS (uses config default if None)
        config: VisualizationConfig instance (uses global if None)
        
    Raises:
        ValueError: If inputs are invalid or have mismatched lengths
        IOError: If cannot write video
    """
    if not frames:
        raise ValueError("frames list cannot be empty")
    
    # Validate list lengths
    n_frames = len(frames)
    if not (len(skeletons) == n_frames and len(actions) == n_frames and len(confidences) == n_frames):
        raise ValueError(f"All lists must have same length: frames={n_frames}, skeletons={len(skeletons)}, actions={len(actions)}, confidences={len(confidences)}")
    
    # Validate frame properties
    if not all(isinstance(f, np.ndarray) and f.ndim == 3 for f in frames):
        raise ValueError("All frames must be 3D numpy arrays")
    
    config = config or _viz_config
    fps = fps or config.default_fps
    
    # Get video dimensions from first frame
    height, width = frames[0].shape[:2]
    
    # Validate all frames have same dimensions
    for i, frame in enumerate(frames):
        if frame.shape[:2] != (height, width):
            raise ValueError(f"Frame {i} has shape {frame.shape}, expected ({height}, {width})")
    
    # Create video writer
    try:
        fourcc = cv2.VideoWriter_fourcc(*config.video_codec)
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        if not writer.isOpened():
            raise IOError(f"Failed to open video writer for {output_path}")
        
        for frame, skeleton, action, confidence in zip(frames, skeletons, actions, confidences):
            frame_copy = frame.copy()
            
            # Draw skeleton
            if skeleton is not None:
                try:
                    frame_copy = draw_skeleton(frame_copy, skeleton, config=config)
                except (ValueError, IndexError) as e:
                    warnings.warn(f"Failed to draw skeleton: {e}")
            
            # Draw action label
            if action:
                frame_copy = draw_action_label(frame_copy, action, confidence, config=config)
            
            # Write frame
            writer.write(frame_copy)
        
        writer.release()
    except Exception as e:
        if 'writer' in locals():
            writer.release()
        raise IOError(f"Failed to create video: {e}")


def draw_3d_skeleton(keypoints: np.ndarray, 
                    connections: List[Tuple[int, int]] = NTU_SKELETON_CONNECTIONS,
                    save_path: Optional[str] = None) -> None:
    """
    Draw 3D skeleton using matplotlib.
    
    Args:
        keypoints: 3D keypoints (N, 3)
        connections: Skeleton connections
        save_path: Path to save the visualization
        
    Raises:
        ValueError: If keypoints are invalid
    """
    if not isinstance(keypoints, np.ndarray) or keypoints.ndim != 2 or keypoints.shape[1] != 3:
        raise ValueError(f"keypoints must be (N, 3) array, got shape {keypoints.shape}")
    
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    try:
        # Draw connections
        for start_idx, end_idx in connections:
            start_idx -= 1
            end_idx -= 1
            
            if start_idx < len(keypoints) and end_idx < len(keypoints):
                x = [keypoints[start_idx, 0], keypoints[end_idx, 0]]
                y = [keypoints[start_idx, 1], keypoints[end_idx, 1]]
                z = [keypoints[start_idx, 2], keypoints[end_idx, 2]]
                
                ax.plot(x, y, z, 'b-', linewidth=2, alpha=0.7)
        
        # Draw joints
        ax.scatter(keypoints[:, 0], keypoints[:, 1], keypoints[:, 2], 
                  c='r', s=50, depthshade=True)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title('3D Skeleton')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    finally:
        plt.close(fig)


def visualize_model_architecture(model: torch.nn.Module, 
                                input_shape: Tuple[int, ...],
                                save_path: Optional[str] = None) -> None:
    """
    Visualize model architecture.
    
    Args:
        model: PyTorch model
        input_shape: Input shape (excluding batch dimension)
        save_path: Path to save the visualization
        
    Raises:
        ValueError: If model forward pass fails
    """
    try:
        if make_dot is None:
            raise ImportError(
                "torchviz is required for model architecture visualization. "
                "Install with: pip install torchviz"
            )
        dummy_input = torch.randn(1, *input_shape)
        output = model(dummy_input)
        dot = make_dot(output, params=dict(model.named_parameters()))
        
        if save_path:
            dot.render(save_path, format='png')
        else:
            dot.view()
    except Exception as e:
        raise ValueError(f"Failed to visualize model architecture: {e}")
