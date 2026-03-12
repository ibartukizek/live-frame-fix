"""FILM-based frame interpolation for generating intermediate frames."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import torch


class FILMInterpolator:
    """Frame interpolator using FILM (Frame Interpolation for Large Motion).

    Generates intermediate frames between two anchor frames using a deep
    learning model. Each intermediate frame is generated independently with
    its own temporal position ``t``, not via recursive halving.

    References:
        https://github.com/dajes/frame-interpolation-pytorch
    """

    def __init__(
        self,
        model_path: str,
        device: str = "cuda",
        half: bool = True,
    ):
        """Load the FILM model from disk.

        Args:
            model_path: Path to the pre-trained FILM TorchScript weights file
                (``film_net.pt`` or compatible TorchScript model).
            device: Inference device string, either ``"cuda"`` or ``"cpu"``.
            half: If ``True``, load model in FP16 (faster on supported GPUs).

        Raises:
            FileNotFoundError: If ``model_path`` does not exist.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"FILM model weights not found at {model_path!r}.\n"
                "Please download a TorchScript model from:\n"
                "  https://github.com/dajes/frame-interpolation-pytorch/releases\n"
                "and place the .pt file at the configured path.\n\n"
                "Example usage:\n"
                "  torch.jit.load('path/to/model.pt')"
            )

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.half = half and self.device.type == "cuda"

        # Load TorchScript model directly
        self.model: torch.jit.ScriptModule = torch.jit.load(model_path, map_location=self.device)
        self.model.eval()
        self.model.to(self.device)

        if self.half:
            self.model.half()

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
            dt_tensor = torch.tensor([[t]], dtype=tensor_a.dtype, device=self.device)

            # Call FILM model: TorchScript model expects (x0, x1, batch_dt)
            output = self.model(tensor_a, tensor_b, dt_tensor)

            bgr = self._tensor_to_bgr(output)
            results.append(bgr)

        return results


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python -m correction.film_interpolator <image_a> <image_b>")
        sys.exit(1)

    img_a_path, img_b_path = sys.argv[1], sys.argv[2]

    frame_a = cv2.imread(img_a_path)
    frame_b = cv2.imread(img_b_path)

    if frame_a is None:
        raise SystemExit(f"Cannot load image: {img_a_path!r}")
    if frame_b is None:
        raise SystemExit(f"Cannot load image: {img_b_path!r}")

    model_path = "weights/film_net.pt"
    interpolator = FILMInterpolator(model_path=model_path, device="cuda", half=True)

    n_frames = 5
    output_dir = Path("/tmp/film_test")
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
