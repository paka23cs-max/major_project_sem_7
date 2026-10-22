"""
SequenceDataset — wraps the existing VideoDataset to return sliding windows of frames.

Why this file exists:
  The original VideoDataset.__getitem__(i) returns one frame.
  The transformer needs a SEQUENCE of frames centered on frame i.
  This class wraps any existing dataset and adds that sliding window logic.

  We do NOT rewrite VideoDataset. We wrap it.
  That means all the existing loading, caching, and preprocessing still works.

Drop this file anywhere in data/ and import it alongside the existing dataset.py.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Tuple, Optional


class SequenceDataset(Dataset):
    """
    Wraps any frame-level dataset and returns sliding windows of frames.

    Given an underlying dataset of N frames, this returns (N - sequence_length + 1)
    sequences. Each sequence is a window of `sequence_length` consecutive frames.

    The TARGET of each sequence is the middle frame's index — used during training
    to tell the model "predict this one."

    Example with sequence_length=9:
        Underlying frames: [f0, f1, f2, f3, f4, f5, f6, f7, f8, f9, ...]
        Sequence at index 0: [f0, f1, f2, f3, f4, f5, f6, f7, f8]  → middle = f4
        Sequence at index 1: [f1, f2, f3, f4, f5, f6, f7, f8, f9]  → middle = f5
        ...
    """

    def __init__(
        self,
        base_dataset: Dataset,
        sequence_length: int = 9,   # Must be odd so there's a clear middle frame
    ):
        assert sequence_length % 2 == 1, "sequence_length must be odd (e.g. 9, 11)"
        assert sequence_length >= 3,     "sequence_length must be at least 3"

        self.base_dataset = base_dataset
        self.sequence_length = sequence_length
        self.middle_idx = sequence_length // 2  # e.g. 4 for sequence_length=9

        # How many complete windows can we form?
        # We lose (sequence_length - 1) frames at the edges
        self.num_sequences = len(base_dataset) - sequence_length + 1

        if self.num_sequences <= 0:
            raise ValueError(
                f"Dataset has {len(base_dataset)} frames but sequence_length={sequence_length}. "
                f"Need at least {sequence_length} frames."
            )

        print(f"SequenceDataset: {len(base_dataset)} frames → {self.num_sequences} sequences "
              f"(window={sequence_length}, middle={self.middle_idx})")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns one sliding window of frames.

        Returns:
            sequence_tensor: (sequence_length, 1, 64, 64)
                             All frames in the window, stacked along dim 0
            middle_frame:    (1, 64, 64)
                             The middle frame — used as the reconstruction target
                             during training
        """
        # Collect frames for this window
        frames = []
        for offset in range(self.sequence_length):
            frame_tensor, _ = self.base_dataset[idx + offset]
            # frame_tensor shape: (1, 64, 64) — single channel grayscale
            frames.append(frame_tensor)

        # Stack into (sequence_length, 1, 64, 64)
        sequence_tensor = torch.stack(frames, dim=0)

        # The middle frame is the one we're trying to predict / reconstruct
        middle_frame = frames[self.middle_idx]  # (1, 64, 64)

        return sequence_tensor, middle_frame


class SequenceDatasetWithLabels(SequenceDataset):
    """
    Same as SequenceDataset but also returns the label of the middle frame.
    Used during evaluation on UCSD Ped2 where ground truth labels exist.
    """

    def __init__(self, base_dataset: Dataset, labels: np.ndarray, sequence_length: int = 9):
        super().__init__(base_dataset, sequence_length)
        self.labels = labels  # One label per frame in base_dataset

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        sequence_tensor, middle_frame = super().__getitem__(idx)
        # Label corresponds to the middle frame of this window
        middle_label = int(self.labels[idx + self.middle_idx])
        return sequence_tensor, middle_frame, middle_label


class BoundaryAwareSequenceDataset(Dataset):
    """
    Sliding-window sequence dataset that never crosses clip/video boundaries.

    Args:
        base_dataset: Frame-level dataset.
        clip_ids: One clip identifier per frame in base_dataset order.
        sequence_length: Odd number of consecutive frames per window.
        frame_indices: Optional subset of frame indices to use. Windows are built
                       only from consecutive indices that share a clip id.
    """

    def __init__(
        self,
        base_dataset: Dataset,
        clip_ids,
        sequence_length: int = 9,
        frame_indices: Optional[list] = None,
    ):
        assert sequence_length % 2 == 1, "sequence_length must be odd (e.g. 9, 11)"
        assert sequence_length >= 3, "sequence_length must be at least 3"

        if len(clip_ids) != len(base_dataset):
            raise ValueError(
                f"clip_ids length ({len(clip_ids)}) must match dataset length ({len(base_dataset)})"
            )

        self.base_dataset = base_dataset
        self.clip_ids = list(clip_ids)
        self.sequence_length = sequence_length
        self.middle_idx = sequence_length // 2

        if frame_indices is None:
            frame_indices = list(range(len(base_dataset)))
        self.frame_indices = sorted(int(i) for i in frame_indices)

        self.windows = self._build_windows()
        if not self.windows:
            raise ValueError(
                f"No boundary-safe windows could be formed with sequence_length={sequence_length}"
            )

        unique_clips = len(set(self.clip_ids[i] for i in self.frame_indices))
        print(
            f"BoundaryAwareSequenceDataset: {len(self.frame_indices)} frames, "
            f"{unique_clips} clips -> {len(self.windows)} windows "
            f"(window={sequence_length}, middle={self.middle_idx})"
        )

    def _build_windows(self):
        windows = []
        if not self.frame_indices:
            return windows

        current_run = []
        previous_idx = None
        previous_clip = None

        for frame_idx in self.frame_indices:
            clip_id = self.clip_ids[frame_idx]
            continues_run = (
                previous_idx is not None
                and frame_idx == previous_idx + 1
                and clip_id == previous_clip
            )

            if not continues_run and current_run:
                windows.extend(self._windows_from_run(current_run))
                current_run = []

            current_run.append(frame_idx)
            previous_idx = frame_idx
            previous_clip = clip_id

        if current_run:
            windows.extend(self._windows_from_run(current_run))

        return windows

    def _windows_from_run(self, run):
        if len(run) < self.sequence_length:
            return []
        return [
            tuple(run[start:start + self.sequence_length])
            for start in range(0, len(run) - self.sequence_length + 1)
        ]

    def __len__(self) -> int:
        return len(self.windows)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        frame_indices = self.windows[idx]
        frames = []
        for frame_idx in frame_indices:
            frame_tensor, _ = self.base_dataset[frame_idx]
            frames.append(frame_tensor)

        sequence_tensor = torch.stack(frames, dim=0)
        middle_frame = frames[self.middle_idx]
        return sequence_tensor, middle_frame

    def get_middle_indices(self) -> np.ndarray:
        return np.array([window[self.middle_idx] for window in self.windows], dtype=np.int64)

    def validate_no_boundary_crossing(self) -> bool:
        for window in self.windows:
            clip = self.clip_ids[window[0]]
            if any(self.clip_ids[i] != clip for i in window):
                return False
        return True


# ─────────────────────────────────────────────
# Sanity check
# ─────────────────────────────────────────────
if __name__ == "__main__":
    from torch.utils.data import TensorDataset

    print("Testing SequenceDataset...")

    # Fake underlying dataset: 100 frames, each (1, 64, 64)
    fake_frames = torch.randn(100, 1, 64, 64)
    fake_targets = fake_frames.clone()
    base_ds = TensorDataset(fake_frames, fake_targets)

    seq_ds = SequenceDataset(base_ds, sequence_length=9)
    print(f"Number of sequences: {len(seq_ds)}")  # Expected: 92

    seq, mid = seq_ds[0]
    print(f"Sequence shape: {seq.shape}")    # Expected: (9, 1, 64, 64)
    print(f"Middle frame shape: {mid.shape}")  # Expected: (1, 64, 64)

    # Test with DataLoader
    loader = DataLoader(seq_ds, batch_size=8, shuffle=True)
    batch_seq, batch_mid = next(iter(loader))
    print(f"Batch sequence shape: {batch_seq.shape}")   # Expected: (8, 9, 1, 64, 64)
    print(f"Batch middle shape:   {batch_mid.shape}")   # Expected: (8, 1, 64, 64)

    print("SequenceDataset tests passed.")
