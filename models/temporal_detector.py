"""
TemporalAnomalyDetector — training and inference pipeline for the transformer model.
This sits ALONGSIDE the existing AnomalyDetector in detector.py.
It does NOT replace it. Both can coexist so you can compare them directly.

PATCH NOTES:
  - Added compute_score_components(loader) — required by train_temporal.py's
    protocol-safe evaluation (evaluate_protocol_scores). Returns a dict of four
    per-window score arrays, computed WITHOUT any threshold/normalization
    (train_temporal.py handles calibration-only normalization itself):

      'cnn_reconstruction'   — pixel-space MSE, CNN reconstructing the middle frame
      'transformer_temporal' — latent-space MSE, masked transformer predicting the
                                middle latent from surrounding context
      'latent_interpolation' — linear-interpolation baseline: predict middle latent
                                as the midpoint of its two neighbors, no learning
      'latent_acceleration'  — curvature/jerk score: ||z[t+1] - 2*z[t] + z[t-1]||^2,
                                measures how abruptly the latent trajectory bends

  - establish_threshold()/detect_anomalies() kept and patched with z-score
    normalization for standalone use (e.g. from a notebook), but train_temporal.py
    does NOT call these — it does its own calibration-split normalization inline.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import os
from typing import Dict, Tuple, Optional
from tqdm import tqdm

from .autoencoder import ConvolutionalAutoencoder
from .temporal_transformer import TemporalTransformerModel


class TemporalAnomalyDetector:
    """
    Trains and runs the temporal transformer for anomaly detection.
    The CNN encoder is kept FROZEN throughout — we only train the transformer.
    """

    def __init__(
        self,
        cnn_encoder: ConvolutionalAutoencoder,
        transformer: TemporalTransformerModel,
        device: torch.device,
    ):
        self.device = device
        self.cnn = cnn_encoder.to(device)
        self.transformer = transformer.to(device)

        # Freeze the CNN
        for param in self.cnn.parameters():
            param.requires_grad = False
        self.cnn.eval()

        self.threshold = None
        self.threshold_stats = {}
        self.is_trained = False

        # Normalization stats for standalone establish_threshold()/detect_anomalies() use
        self.recon_mean = None
        self.recon_std = None
        self.temporal_mean = None
        self.temporal_std = None

    @torch.no_grad()
    def _encode_sequence_batch(self, sequence_batch: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, C, H, W = sequence_batch.shape
        frames_flat = sequence_batch.view(batch_size * seq_len, C, H, W)
        frames_flat = frames_flat.to(self.device)
        latents_flat = self.cnn.encode(frames_flat)
        latent_batch = latents_flat.view(batch_size, seq_len, -1)
        return latent_batch

    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        num_epochs: int = 30,
        learning_rate: float = 1e-4,
        save_path: Optional[str] = None,
        patience: int = 7,
    ) -> Dict:
        print("=" * 60)
        print("TRAINING TEMPORAL TRANSFORMER")
        print(f"  Device:        {self.device}")
        print(f"  Epochs:        {num_epochs}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  CNN:           FROZEN (not being trained)")
        print("=" * 60)

        optimizer = optim.Adam(self.transformer.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', patience=4, factor=0.5, min_lr=1e-6
        )
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_state = None
        no_improve = 0
        history = []

        for epoch in range(num_epochs):
            self.transformer.train()
            train_loss = 0.0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)

            for sequence_batch, middle_frame_batch in pbar:
                with torch.no_grad():
                    latent_seq = self._encode_sequence_batch(sequence_batch)

                predicted = self.transformer(latent_seq)
                actual_middle = latent_seq[:, self.transformer.middle_idx, :]
                loss = criterion(predicted, actual_middle)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), max_norm=1.0)
                optimizer.step()

                train_loss += loss.item()
                pbar.set_postfix({'loss': f"{loss.item():.6f}"})

            train_loss /= len(train_loader)

            val_loss = None
            if val_loader:
                val_loss = self._validate(val_loader, criterion)

            monitor = val_loss if val_loss is not None else train_loss
            scheduler.step(monitor)

            history.append({'epoch': epoch + 1, 'train_loss': train_loss, 'val_loss': val_loss})
            val_str = f" | Val: {val_loss:.6f}" if val_loss else ""
            print(f"Epoch {epoch+1:3d} | Train: {train_loss:.6f}{val_str} | LR: {optimizer.param_groups[0]['lr']:.2e}")

            if monitor < best_loss:
                best_loss = monitor
                best_state = {k: v.clone() for k, v in self.transformer.state_dict().items()}
                no_improve = 0
                if save_path:
                    self._save(save_path, epoch, best_loss)
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        if best_state:
            self.transformer.load_state_dict(best_state)
            print(f"Restored best model (loss={best_loss:.6f})")

        self.is_trained = True
        return {'best_loss': best_loss, 'history': history}

    def _validate(self, val_loader: DataLoader, criterion: nn.Module) -> float:
        self.transformer.eval()
        total = 0.0
        with torch.no_grad():
            for sequence_batch, _ in val_loader:
                latent_seq = self._encode_sequence_batch(sequence_batch)
                predicted = self.transformer(latent_seq)
                actual_middle = latent_seq[:, self.transformer.middle_idx, :]
                total += criterion(predicted, actual_middle).item()
        return total / len(val_loader)

    @torch.no_grad()
    def compute_score_components(self, loader: DataLoader) -> Dict[str, np.ndarray]:
        """
        Compute all four per-window anomaly score components, in loader order
        (loader MUST have shuffle=False for these to align with external labels,
        e.g. test_seq_ds.get_middle_indices()).

        Returns a dict of 1D numpy arrays, all the same length:
            'cnn_reconstruction'   — pixel-space MSE (CNN reconstructing middle frame)
            'transformer_temporal' — latent-space MSE (masked transformer prediction)
            'latent_interpolation' — linear-interpolation baseline (no learning)
            'latent_acceleration'  — curvature/jerk score (no learning)

        No normalization or thresholding is applied here — the caller
        (e.g. train_temporal.py's evaluate_protocol_scores) is responsible for
        calibration-split-only normalization and thresholding.
        """
        self.transformer.eval()
        self.cnn.eval()

        mid = self.transformer.middle_idx

        cnn_recon_scores = []
        transformer_scores = []
        interp_scores = []
        accel_scores = []

        for sequence_batch, middle_frame_batch in tqdm(
            loader, desc="Computing score components", leave=False
        ):
            latent_seq = self._encode_sequence_batch(sequence_batch)  # (B, T, D)
            actual_middle = latent_seq[:, mid, :]

            # 1. CNN reconstruction error (pixel space, middle frame)
            middle_frame_batch = middle_frame_batch.to(self.device)
            recon_error = self.cnn.reconstruction_error(middle_frame_batch)
            cnn_recon_scores.extend(recon_error.cpu().numpy())

            # 2. Transformer temporal prediction error (latent space)
            temporal_score = self.transformer.anomaly_score(latent_seq)
            transformer_scores.extend(temporal_score.cpu().numpy())

            # 3. Linear interpolation baseline: midpoint of immediate neighbors
            prev_latent = latent_seq[:, mid - 1, :]
            next_latent = latent_seq[:, mid + 1, :]
            interp_pred = 0.5 * (prev_latent + next_latent)
            interp_score = torch.mean((interp_pred - actual_middle) ** 2, dim=1)
            interp_scores.extend(interp_score.cpu().numpy())

            # 4. Latent acceleration (curvature / jerk): z[t+1] - 2*z[t] + z[t-1]
            accel = next_latent - 2 * actual_middle + prev_latent
            accel_score = torch.mean(accel ** 2, dim=1)
            accel_scores.extend(accel_score.cpu().numpy())

        return {
            'cnn_reconstruction': np.array(cnn_recon_scores),
            'transformer_temporal': np.array(transformer_scores),
            'latent_interpolation': np.array(interp_scores),
            'latent_acceleration': np.array(accel_scores),
        }

    @torch.no_grad()
    def _compute_raw_scores(self, loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """Collect raw (unnormalized) recon_error and temporal_score arrays over a loader.
        Kept for backward compatibility with establish_threshold()/detect_anomalies()."""
        components = self.compute_score_components(loader)
        return components['cnn_reconstruction'], components['transformer_temporal']

    def establish_threshold(
        self,
        normal_loader: DataLoader,
        percentile: float = 95.0,
        alpha: float = 0.75,
        beta: float = 0.25,
    ) -> Dict:
        """
        Standalone hybrid threshold calibration (z-score normalized).
        NOTE: train_temporal.py does NOT use this — it does its own calibration
        inline in evaluate_protocol_scores(). Use this only if calling the
        detector directly, e.g. from a notebook.
        """
        print("Establishing hybrid threshold on normal data...")

        recon_errors, temporal_scores = self._compute_raw_scores(normal_loader)

        print(f"  [raw] recon_error    — mean: {recon_errors.mean():.6f}, "
              f"std: {recon_errors.std():.6f}, "
              f"min: {recon_errors.min():.6f}, max: {recon_errors.max():.6f}")
        print(f"  [raw] temporal_score — mean: {temporal_scores.mean():.6f}, "
              f"std: {temporal_scores.std():.6f}, "
              f"min: {temporal_scores.min():.6f}, max: {temporal_scores.max():.6f}")

        self.recon_mean = float(recon_errors.mean())
        self.recon_std = float(recon_errors.std() + 1e-8)
        self.temporal_mean = float(temporal_scores.mean())
        self.temporal_std = float(temporal_scores.std() + 1e-8)

        recon_z = (recon_errors - self.recon_mean) / self.recon_std
        temporal_z = (temporal_scores - self.temporal_mean) / self.temporal_std

        scores = alpha * recon_z + beta * temporal_z

        self.threshold = float(np.percentile(scores, percentile))
        self.threshold_stats = {
            'threshold': self.threshold,
            'percentile': percentile,
            'alpha': alpha,
            'beta': beta,
            'recon_mean': self.recon_mean,
            'recon_std': self.recon_std,
            'temporal_mean': self.temporal_mean,
            'temporal_std': self.temporal_std,
            'mean_score': float(np.mean(scores)),
            'std_score': float(np.std(scores)),
            'min_score': float(np.min(scores)),
            'max_score': float(np.max(scores)),
            'fp_rate': float(np.mean(scores > self.threshold)),
        }

        print(f"Hybrid threshold set at {self.threshold:.6f} ({percentile}th percentile)")
        print(f"  alpha={alpha}, beta={beta} (applied to normalized scores)")
        print(f"  Mean normal score: {self.threshold_stats['mean_score']:.6f}")
        print(f"  False positive rate on normal data: {self.threshold_stats['fp_rate']:.4f}")

        return self.threshold_stats

    def detect_anomalies(
        self,
        test_loader: DataLoader,
        alpha: float = 0.75,
        beta: float = 0.25,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Standalone hybrid detection (z-score normalized). Not used by train_temporal.py."""
        if self.threshold is None:
            raise ValueError("Call establish_threshold() before detect_anomalies()")
        if self.recon_mean is None:
            raise ValueError("Normalization stats missing — re-run establish_threshold() first.")

        print("Running hybrid temporal anomaly detection...")

        recon_errors, temporal_scores = self._compute_raw_scores(test_loader)

        recon_z = (recon_errors - self.recon_mean) / self.recon_std
        temporal_z = (temporal_scores - self.temporal_mean) / self.temporal_std

        scores = alpha * recon_z + beta * temporal_z
        flags = scores > self.threshold

        print(f"Detected {flags.sum()} anomalies out of {len(flags)} sequences "
              f"({flags.mean():.2%} anomaly rate)")

        return scores, flags

    def _save(self, path: str, epoch: int, loss: float):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'epoch': epoch,
            'transformer': self.transformer.state_dict(),
            'loss': loss,
            'threshold': self.threshold,
            'threshold_stats': self.threshold_stats,
            'recon_mean': self.recon_mean,
            'recon_std': self.recon_std,
            'temporal_mean': self.temporal_mean,
            'temporal_std': self.temporal_std,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.transformer.load_state_dict(checkpoint['transformer'])
        self.threshold = checkpoint.get('threshold')
        self.threshold_stats = checkpoint.get('threshold_stats', {})
        self.recon_mean = checkpoint.get('recon_mean')
        self.recon_std = checkpoint.get('recon_std')
        self.temporal_mean = checkpoint.get('temporal_mean')
        self.temporal_std = checkpoint.get('temporal_std')
        self.is_trained = True
        print(f"Temporal detector loaded from {path}")
        print(f"  Threshold: {self.threshold}")