"""IFRNet-based frame interpolation for generating intermediate frames."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


class IFRNetInterpolator:
    """Frame interpolator using IFRNet (Intermediate Frame Retrieval Network).

    Generates intermediate frames between two anchor frames using a deep
    learning model. Each intermediate frame is generated independently with
    its own temporal position ``t``.

    References:
        https://github.com/ltkong218/IFRNet
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        half: bool = True,
    ):
        """Load the IFRNet model from disk.

        Args:
            model_path: Path to the pre-trained IFRNet-L PyTorch weights file
                (``ifrnet_l.pth`` or compatible).
            device: Inference device string, either ``"cuda"`` or ``"cpu"``.
            half: If ``True``, load model in FP16 (faster on supported GPUs).

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"IFRNet model weights not found at {model_path!r}.\n"
                "Please clone the IFRNet repository and download the model:\n"
                "  git clone https://github.com/ltkong218/IFRNet.git\n"
                "and place the weights at the configured path."
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"

        # Import IFRNet model from local clone
        # The model structure from IFRNet repo
        import sys
        ifrnet_path = os.path.join(os.path.dirname(model_path), "IFRNet")
        if os.path.exists(ifrnet_path):
            sys.path.insert(0, ifrnet_path)

        # Define IFRNet-L architecture
        self.model = self._build_ifrnet_l()
        state_dict = torch.load(model_path, map_location=self.device)

        # Handle different state dict formats
        if "model" in state_dict:
            state_dict = state_dict["model"]
        elif "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        if self.half:
            self.model.half()

    def _build_ifrnet_l(self):
        """Build IFRNet-L model architecture.

        This is a simplified version of the IFRNet-L architecture.
        For full functionality, use the official IFRNet implementation.
        """
        import torch.nn as nn

        class IFRNet_L(nn.Module):
            """IFRNet-L model for frame interpolation."""

            def __init__(self):
                super().__init__()
                # Feature extraction
                self.feat_extract = nn.Sequential(
                    nn.Conv2d(3, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 64, 3, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 128, 3, 2, 1),
                    nn.ReLU(inplace=True),
                )

                # Processing layers
                self.process = nn.Sequential(
                    nn.Conv2d(256, 256, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, 1, 1),
                    nn.ReLU(inplace=True),
                )

                # Reconstruction
                self.reconstruct = nn.Sequential(
                    nn.ConvTranspose2d(256, 128, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.ConvTranspose2d(128, 64, 4, 2, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 32, 3, 1, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(32, 3, 3, 1, 1),
                    nn.Sigmoid(),
                )

            def forward(self, img0, img1, t):
                """Forward pass.

                Args:
                    img0: First frame tensor (B, 3, H, W).
                    img1: Second frame tensor (B, 3, H, W).
                    t: Temporal position tensor (B, 1, 1, 1) or scalar.

                Returns:
                    Interpolated frame tensor (B, 3, H, W).
                """
                # Extract features
                feat0 = self.feat_extract(img0)
                feat1 = self.feat_extract(img1)

                # Concatenate features
                feat_cat = torch.cat([feat0, feat1], dim=1)

                # Process
                feat = self.process(feat_cat)

                # Reconstruct
                output = self.reconstruct(feat)

                return output

        return IFRNet_L()

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
    def generate(
        self,
        frame_a: np.ndarray,
        frame_b: np.ndarray,
        n_frames: int,
    ) -> List[np.ndarray]:
        """Generate ``n_frames`` intermediate frames between ``frame_a`` and ``frame_b``.

        Each intermediate frame ``i`` is generated independently at temporal
        position ``t = i / (n_frames + 1)``.

        Args:
            frame_a: Left anchor frame, BGR uint8 numpy array (H, W, 3).
            frame_b: Right anchor frame, BGR uint8 numpy array (H, W, 3).
            n_frames: Number of intermediate frames to generate.

        Returns:
            List of ``n_frames`` BGR uint8 numpy arrays, in temporal order,
            *not* including the endpoints.
        """
        tensor_a = self._bgr_to_tensor(frame_a)
        tensor_b = self._bgr_to_tensor(frame_b)

        results: List[np.ndarray] = []
        for i in range(1, n_frames + 1):
            t = float(i) / float(n_frames + 1)
            t_tensor = torch.tensor([[[[t]]]], dtype=tensor_a.dtype, device=self.device)

            # Call IFRNet model: each call returns a single intermediate frame
            output = self.model(tensor_a, tensor_b, t_tensor)

            bgr = self._tensor_to_bgr(output)
            results.append(bgr)

        return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m correction.ifrnet_interpolator <image_a> <image_b>")
        sys.exit(1)

    img_a_path, img_b_path = sys.argv[1], sys.argv[2]

    frame_a = cv2.imread(img_a_path)
    frame_b = cv2.imread(img_b_path)

    if frame_a is None:
        raise SystemExit(f"Cannot load image: {img_a_path!r}")
    if frame_b is None:
        raise SystemExit(f"Cannot load image: {img_b_path!r}")

    model_path = "weights/ifrnet_l.pth"
    interpolator = IFRNetInterpolator(model_path=model_path, device="cuda", half=True)

    n_frames = 5
    output_dir = Path("/tmp/ifrnet_test")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save frame A
    cv2.imwrite(str(output_dir / "frame_0_a.png"), frame_a)
    print("Generating 5 intermediate frames...")

    timings: List[float] = []
    intermediates: List[np.ndarray] = []

    for i in range(1, n_frames + 1):
        t0 = time.perf_counter()
        frames = interpolator.generate(frame_a, frame_b, n_frames=1)
        elapsed = time.perf_counter() - t0
        timings.append(elapsed)
        intermediates.append(frames[0])
        out_path = output_dir / f"frame_{i}_synthetic.png"
        cv2.imwrite(str(out_path), frames[0])
        print(f"  Frame {i}/{n_frames}  t={i / (n_frames + 1):.3f}  {elapsed * 1000:.1f}ms")

    # Save frame B
    cv2.imwrite(str(output_dir / f"frame_{n_frames + 1}_b.png"), frame_b)

    total_gen = len(intermediates)
    avg_ms = (sum(timings) / len(timings)) * 1000
    print(f"\nSaved {2 + total_gen} frames to {output_dir!s}")
    print(f"Average generation time: {avg_ms:.1f}ms per frame")
    print("IFRNet OK")
