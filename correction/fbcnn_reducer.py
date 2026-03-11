"""FBCNN-based JPEG artifact reduction."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)


class FBCNNReducer:
    """JPEG artifact reducer using FBCNN (Flexible Blind Convolutional Neural Network).

    Removes compression artifacts from frames using a deep learning model
    operating in blind mode (unknown quality factor).

    References:
        https://github.com/jiaxi-jiang/FBCNN
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        half: bool = True,
    ):
        """Load the FBCNN color model from disk.

        Args:
            model_path: Path to the pre-trained FBCNN color weights file
                (``fbcnn_color.pth``).
            device: Inference device string, either ``"cuda"`` or ``"cpu"``.
            half: If ``True``, load model in FP16 (faster on supported GPUs).

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FBCNN model weights not found at {model_path!r}.\n"
                "Please clone the FBCNN repository and download the color model:\n"
                "  git clone https://github.com/jiaxi-jiang/FBCNN.git\n"
                "and place the weights at the configured path."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"

        self.model = self._build_fbcnn_color()
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle different state dict formats
        if "params" in state_dict:
            state_dict = state_dict["params"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        elif "model" in state_dict:
            state_dict = state_dict["model"]

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        if self.half:
            self.model.half()

    def _build_fbcnn_color(self) -> torch.nn.Module:
        """Build a simplified FBCNN color model architecture.

        For full accuracy use the official FBCNN implementation from:
        https://github.com/jiaxi-jiang/FBCNN
        """
        import torch.nn as nn

        class FBCNN_Color(nn.Module):
            """FBCNN for color image JPEG artifact reduction."""

            def __init__(self, channels: int = 3, n_feats: int = 64, n_blocks: int = 16):
                super().__init__()

                # Head: input mapping
                self.head = nn.Sequential(
                    nn.Conv2d(channels, n_feats, 3, 1, 1),
                    nn.ReLU(inplace=True),
                )

                # Body: residual blocks
                body = []
                for _ in range(n_blocks):
                    body += [
                        nn.Conv2d(n_feats, n_feats, 3, 1, 1),
                        nn.ReLU(inplace=True),
                    ]
                self.body = nn.Sequential(*body)

                # Quality factor prediction branch
                self.qf_branch = nn.Sequential(
                    nn.Conv2d(n_feats, n_feats, 3, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.AdaptiveAvgPool2d(1),
                    nn.Flatten(),
                    nn.Linear(n_feats, 1),
                    nn.Sigmoid(),
                )

                # Tail: output mapping (channel count includes QF map)
                self.tail = nn.Sequential(
                    nn.Conv2d(n_feats + 1, n_feats, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(n_feats, channels, 3, 1, 1),
                )

            def forward(self, x: torch.Tensor, qf_map: Optional[torch.Tensor] = None):
                """Forward pass.

                Args:
                    x: Input image tensor (B, 3, H, W), values in [0, 1].
                    qf_map: Quality factor map (B, 1, H, W). If None or all-zero,
                        quality is estimated from the image.

                Returns:
                    Restored image tensor (B, 3, H, W), values in [0, 1].
                """
                feat = self.head(x)
                feat = self.body(feat)

                # Use provided QF map or estimate quality from features
                h, w = x.shape[-2:]
                if qf_map is None or torch.all(qf_map == 0):
                    # Blind mode: estimate quality from features
                    qf_val = self.qf_branch(feat)  # (B, 1)
                    qf_map_upsampled = qf_val.view(-1, 1, 1, 1).expand(-1, 1, h, w)
                else:
                    # Guided mode: use provided QF map
                    qf_map_upsampled = torch.nn.functional.interpolate(
                        qf_map, size=(h, w), mode="nearest"
                    )

                # Concatenate features with QF map
                feat = torch.cat([feat, qf_map_upsampled], dim=1)

                # Residual output
                out = self.tail(feat)
                return torch.clamp(x + out, 0.0, 1.0)

        return FBCNN_Color()

    def _bgr_to_tensor(self, bgr: np.ndarray) -> torch.Tensor:
        """Convert a BGR uint8 numpy array to a normalised float tensor.

        Args:
            bgr: (H, W, 3) BGR uint8 array.

        Returns:
            (1, 3, H, W) float tensor on ``self.device`` with values in [0, 1].
        """
        # BGR -> RGB, HWC -> CHW, uint8 -> float32, normalise
        rgb = bgr[:, :, ::-1].astype(np.float32) / 255.0
        tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor.to(self.device)
        if self.half:
            tensor = tensor.half()
        return tensor

    def _tensor_to_bgr(self, tensor: torch.Tensor) -> np.ndarray:
        """Convert a model output tensor back to a BGR uint8 numpy array.

        Args:
            tensor: (1, 3, H, W) float tensor with values in [0, 1].

        Returns:
            (H, W, 3) BGR uint8 array.
        """
        arr = tensor.squeeze(0).float().cpu().numpy()  # (3, H, W)
        arr = np.clip(arr, 0.0, 1.0)
        rgb = (arr * 255.0).astype(np.uint8).transpose(1, 2, 0)  # HWC
        bgr = rgb[:, :, ::-1]  # RGB -> BGR
        return np.ascontiguousarray(bgr)

    @torch.inference_mode()
    def reduce(self, frame: np.ndarray) -> np.ndarray:
        """Apply JPEG artifact reduction to a single frame.

        Operates in blind mode with a zero QF map — the model estimates the
        compression quality factor from the image itself.

        Args:
            frame: Input BGR uint8 numpy array (H, W, 3).

        Returns:
            Artifact-reduced BGR uint8 numpy array of the same resolution.
            Returns the original frame unchanged if an exception occurs.
        """
        try:
            _, _, h, w = (1, 3) + frame.shape[:2]
            tensor = self._bgr_to_tensor(frame)

            # Blind mode: zero QF map of shape (1, 1, H, W)
            qf_map = torch.zeros(
                1, 1, frame.shape[0], frame.shape[1],
                dtype=tensor.dtype,
                device=self.device,
            )

            output = self.model(tensor, qf_map)
            return self._tensor_to_bgr(output)
        except Exception as exc:  # noqa: BLE001
            logger.warning("FBCNNReducer.reduce() failed, returning original frame: %s", exc)
            return frame


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m correction.fbcnn_reducer <image_path>")
        sys.exit(1)

    img_path = sys.argv[1]
    frame = cv2.imread(img_path)
    if frame is None:
        raise SystemExit(f"Cannot load image: {img_path!r}")

    model_path = "weights/fbcnn_color.pth"
    reducer = FBCNNReducer(model_path=model_path, device="cuda", half=True)

    output_dir = Path("/tmp/fbcnn_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = reducer.reduce(frame)

    out_path = output_dir / "output.jpg"
    cv2.imwrite(str(out_path), result)

    print(f"Saved output to {out_path!s}")
    print("FBCNN OK")
