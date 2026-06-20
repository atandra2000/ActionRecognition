"""
Real-time inference pipeline for action recognition
"""

import os
import time
import argparse
import cv2
import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from collections import deque
import threading
import queue
from threading import Event
from scipy.ndimage import gaussian_filter1d

from src.models.action_recognition import create_action_model
from src.models.pose_estimator import PoseEstimator3D, heatmaps_to_keypoints
from src.utils.config import load_config, Config
from src.utils.logger import setup_logger
from src.utils.visualization import draw_skeleton, draw_action_label


# NTU RGB+D 120 Action Labels
NTU_ACTION_LABELS = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up (from sitting position)", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses",
    "put on a hat/cap", "take off a hat/cap", "cheer up", "hand waving", "kicking something",
    "reach into pocket", "hopping (one foot jumping)", "jump up", "make a phone call/answer phone",
    "playing with phone/tablet", "typing on a keyboard", "pointing to something with finger",
    "taking a selfie", "check time (from watch)", "rub two hands together",
    "nod head/bow", "shake head", "wipe face", "salute", "put the palms together",
    "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling",
    "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neckache)", "nausea or vomiting condition", "use a fan (with hand or paper)/feeling warm",
    "punching/slapping other person", "kicking other person", "pushing other person",
    "pat on back of other person", "point finger at the other person",
    "hugging other person", "giving something to other person",
    "touch other person's pocket", "handshaking", "walking towards each other",
    "walking apart from each other", "put on headphone", "take off headphone",
    "shoot at the basket", "volleyball spike", "throw bowling ball",
    "sword fighting", "baseball swing", "tennis forehand swing",
    "arm curl", "tennis serve", "push-up", "sit-up", "jump rope",
    "play guitar", "play piano", "shoot with gun", "throw frisbee",
    "hammer", "lift", "throw paper plane", "cut with knife", "stir",
    "sprinkle", "pour", "apply cream on face", "apply cream on hand",
    "apply cream on leg", "apply lotion on back", "wipe face with towel",
    "wipe hands with towel", "wipe body with towel", "blow dry hair",
    "brush hair", "apply lipstick", "apply eyeshadow", "apply mascara",
    "apply blush", "apply foundation", "wear a shoe", "take off a shoe",
    "wear a shirt", "take off a shirt", "wear a coat", "take off a coat",
    "wear a hat", "take off a hat", "wear a backpack", "take off a backpack",
    "wear a scarf", "take off a scarf", "wear a belt", "take off a belt",
    "wear a watch", "take off a watch", "wear a ring", "take off a ring",
    "wear a necklace", "take off a necklace", "wear glasses", "take off glasses",
    "wear a mask", "take off a mask", "wear a glove", "take off a glove",
    "wear a sock", "take off a sock", "wear a pants", "take off a pants"
]


def _apply_temporal_smoothing(predictions: List[Tuple[int, float]], window_size: int = 5) -> Tuple[int, float]:
    """Apply temporal smoothing to predictions using Gaussian kernel.
    
    Args:
        predictions: List of (action_id, confidence) tuples
        window_size: Size of smoothing window
        
    Returns:
        Smoothed (action_id, confidence) tuple
    """
    if not predictions:
        return 0, 0.0
    
    if len(predictions) == 1:
        return predictions[0]
    
    # Extract action IDs and confidences
    action_ids = np.array([p[0] for p in predictions])
    confidences = np.array([p[1] for p in predictions])
    
    # Smooth confidences
    try:
        smoothed_conf = gaussian_filter1d(confidences, sigma=1.0)
    except Exception:
        smoothed_conf = confidences
    
    # Use smoothed confidence for most recent prediction
    smoothed_action_id = action_ids[-1]
    smoothed_confidence = smoothed_conf[-1]
    
    return smoothed_action_id, float(smoothed_confidence)


class RealTimeInference:
    """Real-time action recognition inference pipeline."""
    
    def __init__(self, config: Config, model_path: str):
        self.config = config
        self.model_path = model_path
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        
        # Setup logging
        self.logger = setup_logger(
            name=config.project_name,
            log_file=f"{config.output_dir}/logs/inference.log",
            level=config.log_level
        )
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(f"{config.output_dir}/inference", exist_ok=True)
        
        # Load model
        self.model = self._load_model()
        
        # Pose estimator for real-time processing
        self.pose_estimator = PoseEstimator3D(
            num_keypoints=config.model.num_keypoints,
            use_temporal_refinement=False
        ).to(self.device)
        
        # Buffers for temporal modeling
        self.temporal_window = config.inference.temporal_window
        self.stride = config.inference.stride
        self.skeleton_buffer = deque(maxlen=self.temporal_window)
        self.frame_buffer = deque(maxlen=self.temporal_window)
        
        # Action labels (NTU RGB+D 120)
        self.action_labels = self._load_action_labels()
        
        # Performance tracking
        self.fps_counter = 0
        self.fps_start_time = time.time()
        self.current_fps = 0
        
        # Performance monitoring
        self.inference_times = deque(maxlen=100)
        self.pose_times = deque(maxlen=100)
        
        # Confidence threshold and temporal smoothing
        self.confidence_threshold = getattr(config.inference, 'confidence_threshold', 0.5)
        self.temporal_smoothing = getattr(config.inference, 'temporal_smoothing', True)
        self.smoothing_window = getattr(config.inference, 'smoothing_window', 5)
        self.prediction_history = deque(maxlen=self.smoothing_window)
        
        # Threading control
        self.shutdown_event = Event()
        
        # Threading
        self.frame_queue = queue.Queue(maxsize=10)
        self.result_queue = queue.Queue(maxsize=10)
        
        self.logger.info("Real-time inference initialized")
    
    def _load_model(self) -> torch.nn.Module:
        """Load trained model."""
        # Create model
        model_config = {
            'num_classes': self.config.model.num_classes,
            'num_keypoints': self.config.model.num_keypoints,
            'pose_dim': self.config.model.pose_dim,
            'use_attention': self.config.model.use_attention if hasattr(self.config.model, 'use_attention') else True,
            'multimodal': False,  # Only use skeleton for real-time
        }
        
        model = create_action_model(model_config)
        
        # Load weights
        checkpoint = torch.load(self.model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        model = model.to(self.device)
        model.eval()
        
        self.logger.info(f"Model loaded from {self.model_path}")
        return model
    
    def _load_action_labels(self) -> List[str]:
        """Load action class labels.
        
        Returns:
            List of 120 action labels for NTU RGB+D
        """
        if len(NTU_ACTION_LABELS) != 120:
            self.logger.warning(f"Expected 120 action labels, got {len(NTU_ACTION_LABELS)}")
        return NTU_ACTION_LABELS
    
    def preprocess_frame(self, frame: np.ndarray) -> torch.Tensor:
        """Preprocess frame for pose estimation."""
        # Resize frame
        frame_resized = cv2.resize(frame, (256, 256))
        
        # Convert to tensor
        frame_tensor = torch.from_numpy(frame_resized).float() / 255.0
        frame_tensor = frame_tensor.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        
        return frame_tensor.to(self.device)
    
    def extract_pose(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """Extract pose from frame with error handling.
        
        Args:
            frame: Input video frame
            
        Returns:
            3D keypoints (N, 3) or None if extraction fails
        """
        pose_start = time.time()
        
        try:
            with torch.no_grad():
                frame_tensor = self.preprocess_frame(frame)
                pose_output = self.pose_estimator(frame_tensor)

                if 'keypoints_3d' not in pose_output:
                    self.logger.warning("Pose estimator output missing 'keypoints_3d'")
                    return None

                keypoints_3d = pose_output['keypoints_3d'].squeeze(0).cpu().numpy()

                if keypoints_3d.size == 0 or np.isnan(keypoints_3d).any():
                    self.logger.debug("Invalid keypoints extracted")
                    return None

                self.pose_times.append(time.time() - pose_start)

                return keypoints_3d
        except Exception as e:
            self.logger.debug(f"Pose extraction failed: {e}")
            return None
    
    def recognize_action(self, skeleton_sequence: np.ndarray) -> Tuple[int, float]:
        """Recognize action from skeleton sequence with validation and smoothing.
        
        Args:
            skeleton_sequence: Skeleton sequence (V, T, C)
            
        Returns:
            Tuple of (action_id, confidence)
        """
        inf_start = time.time()
        try:
            with torch.no_grad():
                skeleton_tensor = torch.from_numpy(skeleton_sequence).float()
                skeleton_tensor = skeleton_tensor.permute(2, 0, 1).unsqueeze(0)
                skeleton_tensor = skeleton_tensor.to(self.device)
                
                outputs = self.model({'skeleton': skeleton_tensor})
                
                if 'logits' not in outputs:
                    self.logger.debug("Model output missing 'logits'")
                    self.inference_times.append(time.time() - inf_start)
                    return 0, 0.0
                
                logits = outputs['logits']
                
                if logits.isnan().any() or logits.isinf().any():
                    self.logger.debug("Model logits contain NaN/Inf")
                    self.inference_times.append(time.time() - inf_start)
                    return 0, 0.0
                
                probabilities = F.softmax(logits, dim=1)
                predicted_class = torch.argmax(probabilities, dim=1).item()
                confidence = probabilities[0, predicted_class].item()
                
                if confidence < self.confidence_threshold:
                    self.inference_times.append(time.time() - inf_start)
                    return 0, confidence
                
                self.prediction_history.append((predicted_class, confidence))
                
                if self.temporal_smoothing and len(self.prediction_history) > 1:
                    predicted_class, confidence = _apply_temporal_smoothing(list(self.prediction_history))
                
                self.inference_times.append(time.time() - inf_start)
                return predicted_class, confidence
        except Exception as e:
            self.logger.debug(f"Action recognition failed: {e}")
            self.inference_times.append(time.time() - inf_start)
            return 0, 0.0
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Process single frame."""
        start_time = time.time()
        
        # Extract pose
        pose = self.extract_pose(frame)
        
        if pose is None:
            return {'action': None, 'confidence': 0.0, 'pose': None, 'fps': self.current_fps}
        
        # Add to buffer
        self.skeleton_buffer.append(pose)
        self.frame_buffer.append(frame)
        
        # Check if we have enough frames
        if len(self.skeleton_buffer) < self.temporal_window:
            return {'action': None, 'confidence': 0.0, 'pose': pose, 'fps': self.current_fps}
        
        # Convert to numpy array
        skeleton_sequence = np.stack(self.skeleton_buffer, axis=1)  # (V, T, C)
        
        # Recognize action
        action_id, confidence = self.recognize_action(skeleton_sequence)
        
        # Update FPS
        self.fps_counter += 1
        if time.time() - self.fps_start_time >= 1.0:
            self.current_fps = self.fps_counter
            self.fps_counter = 0
            self.fps_start_time = time.time()
        
        return {
            'action': action_id,
            'action_name': self.action_labels[action_id] if action_id < len(self.action_labels) else f"Action {action_id}",
            'confidence': confidence,
            'pose': pose,
            'fps': self.current_fps,
            'processing_time': time.time() - start_time
        }
    
    def process_video_stream(self, source: Union[int, str] = 0) -> None:
        """Process video stream from camera or file with robust error handling."""
        # Open video source
        if isinstance(source, str) and source.isdigit():
            source = int(source)
        
        cap = cv2.VideoCapture(source)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video source: {source}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        self.logger.info(f"Video source opened: {width}x{height} @ {fps} FPS")
        
        # Video writer for saving output
        writer = None
        if self.config.inference.save_output:
            try:
                output_path = f"{self.config.output_dir}/inference/output.mp4"
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
                if not writer.isOpened():
                    self.logger.warning(f"Failed to create video writer, skipping output save")
                    writer = None
            except Exception as e:
                self.logger.warning(f"Failed to create video writer: {e}")
                writer = None
        
        try:
            frame_count = 0
            
            while not self.shutdown_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                
                # Process frame
                result = self.process_frame(frame)
                
                # Visualize results
                frame = self._visualize_result(frame, result)
                
                # Display frame
                cv2.imshow('Action Recognition', frame)
                
                # Save frame if required
                if writer is not None:
                    try:
                        writer.write(frame)
                    except Exception as e:
                        self.logger.warning(f"Failed to write frame: {e}")
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                should_quit, screenshot_path = self._handle_keyboard_input(key)
                if should_quit:
                    break
                
                if screenshot_path:
                    try:
                        os.makedirs(os.path.dirname(screenshot_path), exist_ok=True)
                        cv2.imwrite(screenshot_path, frame)
                        self.logger.info(f"Screenshot saved: {screenshot_path}")
                    except Exception as e:
                        self.logger.warning(f"Failed to save screenshot: {e}")
                
                # Log processing info
                if frame_count % 100 == 0:
                    avg_pose_time = np.mean(list(self.pose_times)) if self.pose_times else 0
                    avg_inf_time = np.mean(list(self.inference_times)) if self.inference_times else 0
                    self.logger.info(
                        f"Processed {frame_count} frames, "
                        f"FPS: {result['fps']}, "
                        f"Pose: {avg_pose_time*1000:.1f}ms, "
                        f"Inference: {avg_inf_time*1000:.1f}ms, "
                        f"Action: {result['action_name'] if result['action'] else 'None'}"
                    )
        
        except KeyboardInterrupt:
            self.logger.info("Interrupted by user")
        except Exception as e:
            self.logger.error(f"Error in video processing: {e}", exc_info=True)
        finally:
            # Cleanup
            cap.release()
            if writer is not None:
                try:
                    writer.release()
                    self.logger.info(f"Output video saved: {output_path}")
                except Exception as e:
                    self.logger.warning(f"Failed to close video writer: {e}")
            cv2.destroyAllWindows()
            self.logger.info(f"Processed {frame_count} frames")
    
    def _visualize_result(self, frame: np.ndarray, result: Dict[str, Any]) -> np.ndarray:
        """Visualize inference results on frame.
        
        Args:
            frame: Input frame
            result: Result dictionary from process_frame
            
        Returns:
            Frame with visualizations
        """
        # Draw skeleton
        if result['pose'] is not None:
            frame = draw_skeleton(frame, result['pose'])
        
        # Draw action label
        if result['action'] is not None and result['action'] > 0:
            frame = draw_action_label(
                frame,
                result['action_name'],
                result['confidence'],
                result['fps']
            )
        
        # Draw FPS
        if self.config.inference.display_fps:
            cv2.putText(frame, f"FPS: {result['fps']}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def _handle_keyboard_input(self, key: int) -> Tuple[bool, Optional[str]]:
        """Handle keyboard input.
        
        Args:
            key: Keyboard key code
            
        Returns:
            Tuple of (should_quit, screenshot_path or None)
        """
        if key == ord('q'):  # Quit
            return True, None
        elif key == ord('s'):  # Screenshot
            timestamp = int(time.time() * 1000)
            screenshot_path = f"{self.config.output_dir}/inference/frame_{timestamp}.jpg"
            return False, screenshot_path
        return False, None
    
    def process_video_file(self, video_path: str, output_path: Optional[str] = None) -> None:
        """Process video file."""
        if output_path is None:
            output_path = f"{self.config.output_dir}/inference/processed_video.mp4"
        
        self.config.inference.save_output = True
        self.process_video_stream(video_path)
    
    def run_multithreaded(self, source: Union[int, str] = 0) -> None:
        """Run inference with separate threads for capture and processing (thread-safe)."""
        self.shutdown_event.clear()
        
        # Start threads
        threads = [
            threading.Thread(target=self._capture_frames, args=(source,), daemon=False),
            threading.Thread(target=self._process_frames, daemon=False),
            threading.Thread(target=self._display_results, daemon=False)
        ]
        
        for thread in threads:
            thread.start()
        
        try:
            # Keep main thread alive
            for thread in threads:
                thread.join()
        except KeyboardInterrupt:
            self.logger.info("Stopping inference...")
            self.shutdown_event.set()
            
            # Wait for threads to finish
            for thread in threads:
                thread.join(timeout=2.0)
    
    def _capture_frames(self, source: Union[int, str]) -> None:
        """Capture frames from video source with shutdown support."""
        cap = None
        try:
            if isinstance(source, str) and source.isdigit():
                source = int(source)
            
            cap = cv2.VideoCapture(source)
            if not cap.isOpened():
                self.logger.error(f"Failed to open video source: {source}")
                self.shutdown_event.set()
                return
            
            self.logger.info(f"Capture thread started for source: {source}")
            
            while not self.shutdown_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    self.logger.info("End of video stream reached")
                    self.shutdown_event.set()
                    break
                
                try:
                    self.frame_queue.put(frame, timeout=0.1)
                except queue.Full:
                    pass  # Skip frame if queue is full
        except Exception as e:
            self.logger.error(f"Capture thread error: {e}", exc_info=True)
            self.shutdown_event.set()
        finally:
            if cap is not None:
                cap.release()
            self.logger.info("Capture thread stopped")
    
    def _process_frames(self) -> None:
        """Process frames from queue with shutdown support."""
        try:
            self.logger.info("Process thread started")
            
            while not self.shutdown_event.is_set():
                try:
                    frame = self.frame_queue.get(timeout=0.1)
                    result = self.process_frame(frame)
                    
                    # Visualize
                    frame = self._visualize_result(frame, result)
                    
                    try:
                        self.result_queue.put((frame, result), timeout=0.1)
                    except queue.Full:
                        pass  # Skip if display queue is full
                except queue.Empty:
                    pass
        except Exception as e:
            self.logger.error(f"Process thread error: {e}", exc_info=True)
            self.shutdown_event.set()
        finally:
            self.logger.info("Process thread stopped")
    
    def _display_results(self) -> None:
        """Display processed frames with shutdown support."""
        try:
            self.logger.info("Display thread started")
            
            while not self.shutdown_event.is_set():
                try:
                    frame, result = self.result_queue.get(timeout=0.1)
                    cv2.imshow('Action Recognition', frame)
                    
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        self.shutdown_event.set()
                        break
                except queue.Empty:
                    pass
        except Exception as e:
            self.logger.error(f"Display thread error: {e}", exc_info=True)
            self.shutdown_event.set()
        finally:
            cv2.destroyAllWindows()
            self.logger.info("Display thread stopped")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(description='Real-time Action Recognition')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model')
    parser.add_argument('--source', type=str, default='0', help='Video source (camera index or file path)')
    parser.add_argument('--output', type=str, default=None, help='Output video path')
    parser.add_argument('--multithread', action='store_true', help='Use multithreaded processing')
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Create inference pipeline
    inference = RealTimeInference(config, args.model)
    
    # Determine source type
    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    
    # Run inference
    if args.multithread:
        inference.run_multithreaded(source)
    elif args.output:
        inference.process_video_file(source, args.output)
    else:
        inference.process_video_stream(source)


if __name__ == "__main__":
    main()