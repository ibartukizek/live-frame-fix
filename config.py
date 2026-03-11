"""Configuration for the video quality correction pipeline."""

from dataclasses import dataclass


@dataclass
class PipelineConfig:
    """Configuration parameters for the video processing pipeline.
    
    Attributes:
        input_path: Path to the input video file
        output_path: Path where the corrected video will be saved
        fps: Frames per second for the video (default: 30)
        frozen_hash_threshold: Number of identical frame hashes to consider a frame frozen (default: 5)
        frozen_min_frames: Minimum number of consecutive frozen frames to trigger correction (default: 3)
        drop_pts_tolerance: PTS difference tolerance for detecting frame drops in seconds (default: 0.5)
        artifact_block_threshold: Blockiness threshold for artifact detection (default: 1.8)
        drop_short_max_frames: Maximum frames to classify as short drop vs long drop (default: 5)
        film_model_path: Path to the FILM neural network model weights (default: "weights/film_net.pt")
        ifrnet_model_path: Path to the IFRNet model weights (default: "weights/ifrnet_l.pth")
        fbcnn_model_path: Path to the FBCNN model weights (default: "weights/fbcnn_color.pth")
        device: Computation device, either "cuda" or "cpu" (default: "cuda")
        half_precision: Whether to use FP16 for inference (default: True)
    """
    input_path: str
    output_path: str
    fps: int = 30
    frozen_hash_threshold: int = 5
    frozen_min_frames: int = 3
    drop_pts_tolerance: float = 0.5
    artifact_block_threshold: float = 1.8
    drop_short_max_frames: int = 5
    film_model_path: str = "weights/film_net.pt"
    ifrnet_model_path: str = "weights/ifrnet_l.pth"
    fbcnn_model_path: str = "weights/fbcnn_color.pth"
    device: str = "cuda"
    half_precision: bool = True
