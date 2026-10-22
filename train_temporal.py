"""
train_temporal.py — Train and evaluate the temporal transformer.
"""

import argparse
import json
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import sys
import multiprocessing

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from torch.utils.data import DataLoader
from models.autoencoder import ConvolutionalAutoencoder
from models.temporal_transformer import TemporalTransformerModel
from models.temporal_detector import TemporalAnomalyDetector
from data.dataset import SyntheticVideoDataset, VideoDataset
from data.preprocessing import VideoPreprocessor
from data.sequence_dataset import BoundaryAwareSequenceDataset

LATENT_DIM            = 256
SEQ_LEN               = 9
NUM_HEADS             = 4
NUM_LAYERS            = 2
BATCH_SIZE            = 32
LEARNING_RATE         = 1e-4
ORIGINAL_MODEL_PATH   = "outputs/trained_model.pth"
TRANSFORMER_SAVE_PATH = "outputs/temporal_transformer.pth"
PLOT_OUTPUT_PATH      = "outputs/comparison_plot.png"
UCSD_THRESHOLD        = 0.001290  # Calibrated by main.py on UCSD Ped2 training data


def load_or_create_cnn(device):
    cnn = ConvolutionalAutoencoder(latent_dim=LATENT_DIM)
    if os.path.exists(ORIGINAL_MODEL_PATH):
        checkpoint = torch.load(ORIGINAL_MODEL_PATH, map_location=device)
        if 'model_state_dict' in checkpoint:
            cnn.load_state_dict(checkpoint['model_state_dict'])
        else:
            cnn.load_state_dict(checkpoint)
        print(f"Loaded pre-trained CNN from {ORIGINAL_MODEL_PATH}")
    else:
        print(f"WARNING: No pre-trained CNN found at {ORIGINAL_MODEL_PATH}")
    return cnn


def get_image_paths_from_dir(root_dir):
    image_extensions = {'.tif', '.tiff', '.png', '.jpg', '.jpeg'}
    paths = []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in sorted(filenames):
            if os.path.splitext(fname)[1].lower() in image_extensions:
                paths.append(os.path.join(dirpath, fname))
    paths.sort()
    return paths


def get_image_paths_and_clips(root_dir):
    paths = get_image_paths_from_dir(root_dir)
    clip_ids = [os.path.basename(os.path.dirname(path)) for path in paths]
    return paths, clip_ids


def split_ordered_clips(clip_ids, train_ratio=0.70, calibration_ratio=0.15):
    clips = list(dict.fromkeys(clip_ids))
    if len(clips) < 3:
        return split_ordered_frames(len(clip_ids))

    n_clips = len(clips)
    n_train = max(1, int(round(n_clips * train_ratio)))
    n_cal = max(1, int(round(n_clips * calibration_ratio)))
    if n_train + n_cal >= n_clips:
        n_train = max(1, n_clips - 2)
        n_cal = 1

    split_clips = {
        'train': set(clips[:n_train]),
        'calibration': set(clips[n_train:n_train + n_cal]),
        'validation': set(clips[n_train + n_cal:]),
    }
    return {
        name: [i for i, clip_id in enumerate(clip_ids) if clip_id in clip_set]
        for name, clip_set in split_clips.items()
    }


def split_ordered_frames(num_frames, train_ratio=0.70, calibration_ratio=0.15):
    n_train = max(1, int(round(num_frames * train_ratio)))
    n_cal = max(1, int(round(num_frames * calibration_ratio)))
    if n_train + n_cal >= num_frames:
        n_train = max(1, num_frames - 2)
        n_cal = 1
    return {
        'train': list(range(0, n_train)),
        'calibration': list(range(n_train, n_train + n_cal)),
        'validation': list(range(n_train + n_cal, num_frames)),
    }


def summarize_distribution(name, scores):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.size == 0:
        print(f"  {name}: no scores")
        return
    print(
        f"  {name}: n={scores.size} mean={scores.mean():.6f} std={scores.std():.6f} "
        f"min={scores.min():.6f} p50={np.percentile(scores, 50):.6f} "
        f"p90={np.percentile(scores, 90):.6f} p95={np.percentile(scores, 95):.6f} "
        f"p99={np.percentile(scores, 99):.6f} max={scores.max():.6f}"
    )


def normalize_with_calibration(scores, calibration_scores):
    scores = np.asarray(scores, dtype=np.float64)
    calibration_scores = np.asarray(calibration_scores, dtype=np.float64)
    mean = calibration_scores.mean()
    std = calibration_scores.std()
    if std < 1e-12:
        std = 1.0
    return (scores - mean) / std, {'mean': float(mean), 'std': float(std)}


def compute_metrics_for_scores(name, scores, labels, threshold):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    flags = scores > threshold

    precision = precision_score(labels, flags, zero_division=0)
    recall = recall_score(labels, flags, zero_division=0)
    f1 = f1_score(labels, flags, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.0

    print(
        f"  [{name}] threshold={threshold:.6f} | Precision={precision:.4f} "
        f"Recall={recall:.4f} F1={f1:.4f} AUC={auc:.4f}"
    )
    return {
        'name': name,
        'scores': scores,
        'flags': flags,
        'labels': labels,
        'threshold': float(threshold),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'auc': float(auc),
    }


def _build_ucsd_labels(test_paths, test_dir):
    """
    Load frame-level labels from actual ground truth .bmp files,
    exactly like main.py does.
    White pixel in GT = anomaly, black = normal.
    """
    labels = []
    for path in test_paths:
        clip_folder = os.path.basename(os.path.dirname(path))
        fname       = os.path.basename(path)
        gt_fname    = os.path.splitext(fname)[0] + '.bmp'
        gt_path     = os.path.join(test_dir, clip_folder + '_gt', gt_fname)

        if os.path.exists(gt_path):
            import cv2
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            # If any pixel is white (>0), frame contains anomaly
            label = 1 if gt_img is not None and gt_img.max() > 0 else 0
        else:
            label = -1  # no ground truth

        labels.append(label)
    return np.array(labels)


def build_datasets(use_synthetic):
    if use_synthetic:
        print("Using synthetic data...")
        full_ds     = SyntheticVideoDataset(num_normal_frames=800, num_anomaly_frames=80, frame_size=(64, 64))
        labels      = full_ds.get_labels()
        train_ds    = torch.utils.data.Subset(full_ds, list(range(800)))
        test_ds     = full_ds
        test_labels = labels
        train_clip_ids = ['synthetic_normal'] * len(train_ds)
        test_clip_ids = ['synthetic_normal' if label == 0 else 'synthetic_anomaly' for label in labels]
    else:
        ucsd_root = os.path.join("data", "UCSD_Anomaly_Dataset.v1p2", "UCSDped2")
        train_dir = os.path.join(ucsd_root, "Train")
        test_dir  = os.path.join(ucsd_root, "Test")
        if not os.path.exists(train_dir):
            raise FileNotFoundError(f"UCSD Ped2 Train not found at: {train_dir}")
        print(f"Loading UCSD Ped2 from: {ucsd_root}")
        train_paths, train_clip_ids = get_image_paths_and_clips(train_dir)
        test_paths, test_clip_ids = get_image_paths_and_clips(test_dir)
        print(f"Found {len(train_paths)} training frames")
        print(f"Found {len(test_paths)} test frames")
        preprocessor = VideoPreprocessor(
            target_size=(64, 64),
            quality_threshold=0.001,
        )
        train_ds    = VideoDataset(
            video_paths=train_paths, frame_size=(64, 64),
            load_in_memory=True, preprocessor=preprocessor,
        )
        test_ds     = VideoDataset(
            video_paths=test_paths, frame_size=(64, 64),
            load_in_memory=True, preprocessor=preprocessor,
        )
        if len(train_ds) != len(train_clip_ids) or len(test_ds) != len(test_clip_ids):
            raise RuntimeError(
                "Frame loading changed dataset length; clip ids no longer align with frames. "
                "Adjust preprocessing before running temporal experiments."
            )
        test_labels = _build_ucsd_labels(test_paths, test_dir)
        valid = test_labels != -1
        print(f"Test labels: {np.sum(test_labels[valid]==0)} normal, "
              f"{np.sum(test_labels[valid]==1)} anomaly "
              f"({np.sum(~valid)} skipped, no ground truth)")
    return train_ds, test_ds, test_labels, train_clip_ids, test_clip_ids


def evaluate_original_model(cnn, test_ds, test_labels, device, use_synthetic):
    print("\n--- Evaluating ORIGINAL model (per-frame) ---")
    cnn.eval()
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)
    scores = []
    with torch.no_grad():
        for frames, _ in test_loader:
            frames = frames.to(device)
            err    = cnn.reconstruction_error(frames)
            scores.extend(err.cpu().numpy())
    scores = np.array(scores)

    if use_synthetic:
        valid_mask = np.ones(len(scores), dtype=bool)
        labels     = test_labels
    else:
        valid_mask = test_labels != -1
        scores     = scores[valid_mask]
        labels     = test_labels[valid_mask]

    if use_synthetic:
        threshold = float(np.percentile(scores[labels == 0], 95))
    else:
        threshold = UCSD_THRESHOLD
    print(f"  Using threshold: {threshold:.6f}")

    flags = scores > threshold
    return _compute_metrics(scores, flags, labels, threshold, name="Original")


def evaluate_protocol_scores(temporal_detector, calibration_loader, test_loader, test_seq_ds, test_labels, use_synthetic):
    print("\n--- Protocol-safe score evaluation ---")
    print("Computing calibration scores from normal calibration split only...")
    calibration = temporal_detector.compute_score_components(calibration_loader)
    print("Computing test scores...")
    test = temporal_detector.compute_score_components(test_loader)

    middle_indices = test_seq_ds.get_middle_indices()
    seq_labels = test_labels[middle_indices]
    if not use_synthetic:
        valid_mask = seq_labels != -1
        seq_labels = seq_labels[valid_mask]
        for key in test:
            test[key] = test[key][valid_mask]

    print("\nCalibration normal score distributions:")
    summarize_distribution("CNN reconstruction", calibration['cnn_reconstruction'])
    summarize_distribution("Transformer temporal", calibration['transformer_temporal'])
    summarize_distribution("Latent interpolation", calibration['latent_interpolation'])
    summarize_distribution("Latent acceleration", calibration['latent_acceleration'])

    print("\nTest score distributions:")
    summarize_distribution("CNN reconstruction all", test['cnn_reconstruction'])
    summarize_distribution("Transformer temporal all", test['transformer_temporal'])
    summarize_distribution("Latent interpolation all", test['latent_interpolation'])
    summarize_distribution("Latent acceleration all", test['latent_acceleration'])
    if len(np.unique(seq_labels)) > 1:
        normal_mask = seq_labels == 0
        anomaly_mask = seq_labels == 1
        summarize_distribution("CNN reconstruction test normal", test['cnn_reconstruction'][normal_mask])
        summarize_distribution("CNN reconstruction test anomaly", test['cnn_reconstruction'][anomaly_mask])
        summarize_distribution("Transformer temporal test normal", test['transformer_temporal'][normal_mask])
        summarize_distribution("Transformer temporal test anomaly", test['transformer_temporal'][anomaly_mask])
        summarize_distribution("Latent interpolation test normal", test['latent_interpolation'][normal_mask])
        summarize_distribution("Latent interpolation test anomaly", test['latent_interpolation'][anomaly_mask])
        summarize_distribution("Latent acceleration test normal", test['latent_acceleration'][normal_mask])
        summarize_distribution("Latent acceleration test anomaly", test['latent_acceleration'][anomaly_mask])

    cnn_threshold = float(np.percentile(calibration['cnn_reconstruction'], 95.0))
    temporal_threshold = float(np.percentile(calibration['transformer_temporal'], 95.0))
    interpolation_threshold = float(np.percentile(calibration['latent_interpolation'], 95.0))
    acceleration_threshold = float(np.percentile(calibration['latent_acceleration'], 95.0))

    calibration_cnn_norm, cnn_norm_stats = normalize_with_calibration(
        calibration['cnn_reconstruction'], calibration['cnn_reconstruction']
    )
    calibration_temporal_norm, temporal_norm_stats = normalize_with_calibration(
        calibration['transformer_temporal'], calibration['transformer_temporal']
    )
    test_cnn_norm, _ = normalize_with_calibration(test['cnn_reconstruction'], calibration['cnn_reconstruction'])
    test_temporal_norm, _ = normalize_with_calibration(test['transformer_temporal'], calibration['transformer_temporal'])

    hybrid_alpha = 0.5
    hybrid_beta = 0.5
    calibration_hybrid = hybrid_alpha * calibration_cnn_norm + hybrid_beta * calibration_temporal_norm
    test_hybrid = hybrid_alpha * test_cnn_norm + hybrid_beta * test_temporal_norm
    hybrid_threshold = float(np.percentile(calibration_hybrid, 95.0))

    print("\nCalibration normalization:")
    print(f"  CNN mean={cnn_norm_stats['mean']:.6f}, std={cnn_norm_stats['std']:.6f}")
    print(f"  Transformer mean={temporal_norm_stats['mean']:.6f}, std={temporal_norm_stats['std']:.6f}")
    summarize_distribution("Normalized hybrid calibration", calibration_hybrid)
    summarize_distribution("Normalized hybrid test", test_hybrid)

    calibration_stats = {
        'cnn_threshold': cnn_threshold,
        'temporal_threshold': temporal_threshold,
        'interpolation_threshold': interpolation_threshold,
        'acceleration_threshold': acceleration_threshold,
        'hybrid_threshold': hybrid_threshold,
        'cnn_mean': cnn_norm_stats['mean'],
        'cnn_std': cnn_norm_stats['std'],
        'temporal_mean': temporal_norm_stats['mean'],
        'temporal_std': temporal_norm_stats['std'],
        'hybrid_alpha': hybrid_alpha,
        'hybrid_beta': hybrid_beta,
        'sequence_length': SEQ_LEN,
        'latent_dim': LATENT_DIM,
    }
    os.makedirs("outputs", exist_ok=True)
    with open("outputs/calibration_stats.json", "w") as f:
        json.dump(calibration_stats, f, indent=2)
    print("\nSaved calibration stats to outputs/calibration_stats.json")

    print("\nMetrics:")
    results = {
        'cnn': compute_metrics_for_scores(
            "CNN reconstruction-only", test['cnn_reconstruction'], seq_labels, cnn_threshold
        ),
        'temporal': compute_metrics_for_scores(
            "Transformer temporal-only", test['transformer_temporal'], seq_labels, temporal_threshold
        ),
        'interpolation': compute_metrics_for_scores(
            "1-step latent interpolation", test['latent_interpolation'], seq_labels, interpolation_threshold
        ),
        'acceleration': compute_metrics_for_scores(
            "Latent acceleration", test['latent_acceleration'], seq_labels, acceleration_threshold
        ),
        'hybrid': compute_metrics_for_scores(
            "Normalized CNN+Transformer hybrid", test_hybrid, seq_labels, hybrid_threshold
        ),
    }
    return results, {
        'cnn_threshold': cnn_threshold,
        'temporal_threshold': temporal_threshold,
        'interpolation_threshold': interpolation_threshold,
        'acceleration_threshold': acceleration_threshold,
        'hybrid_threshold': hybrid_threshold,
        'cnn_norm_stats': cnn_norm_stats,
        'temporal_norm_stats': temporal_norm_stats,
        'hybrid_alpha': hybrid_alpha,
        'hybrid_beta': hybrid_beta,
    }


def _compute_metrics(scores, flags, labels, threshold, name):
    from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
    min_len   = min(len(scores), len(labels))
    scores    = scores[:min_len]
    flags     = flags[:min_len]
    labels    = labels[:min_len]
    precision = precision_score(labels, flags, zero_division=0)
    recall    = recall_score(labels, flags, zero_division=0)
    f1        = f1_score(labels, flags, zero_division=0)
    try:
        auc = roc_auc_score(labels, scores)
    except ValueError:
        auc = 0.0
    print(f"  [{name}] Precision: {precision:.4f} | Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}")
    return {'name': name, 'scores': scores, 'flags': flags, 'labels': labels,
            'threshold': threshold, 'precision': precision, 'recall': recall, 'f1': f1, 'auc': auc}


def plot_comparison(original_results, temporal_results, save_path):
    fig = plt.figure(figsize=(16, 8))
    fig.suptitle("Original vs Temporal Transformer — Anomaly Detection Comparison",
                 fontsize=14, fontweight='bold')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(original_results['scores'], color='steelblue', linewidth=0.8, label='Score')
    ax1.axhline(original_results['threshold'], color='red', linestyle='--', linewidth=1.2, label='Threshold')
    for i, lbl in enumerate(original_results['labels']):
        if lbl == 1:
            ax1.axvspan(i, i+1, alpha=0.15, color='red')
    ax1.set_title("Original Model (per-frame MSE)")
    ax1.set_xlabel("Frame index")
    ax1.set_ylabel("Reconstruction error")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(temporal_results['scores'], color='darkorange', linewidth=0.8, label='Score')
    ax2.axhline(temporal_results['threshold'], color='red', linestyle='--', linewidth=1.2, label='Threshold')
    for i, lbl in enumerate(temporal_results['labels']):
        if lbl == 1:
            ax2.axvspan(i, i+1, alpha=0.15, color='red')
    ax2.set_title("Temporal Transformer (latent prediction MSE)")
    ax2.set_xlabel("Sequence index")
    ax2.set_ylabel("Prediction error")
    ax2.legend(fontsize=8)

    ax3 = fig.add_subplot(gs[1, :])
    metrics   = ['Precision', 'Recall', 'F1 Score', 'AUC']
    orig_vals = [original_results['precision'], original_results['recall'],
                 original_results['f1'],        original_results['auc']]
    temp_vals = [temporal_results['precision'], temporal_results['recall'],
                 temporal_results['f1'],        temporal_results['auc']]
    x     = np.arange(len(metrics))
    w     = 0.35
    bars1 = ax3.bar(x - w/2, orig_vals, w, label='Original',  color='steelblue',  alpha=0.85)
    bars2 = ax3.bar(x + w/2, temp_vals, w, label='Temporal',  color='darkorange', alpha=0.85)
    for bar in bars1 + bars2:
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha='center', va='bottom', fontsize=9)
    ax3.set_xticks(x)
    ax3.set_xticklabels(metrics)
    ax3.set_ylim(0, 1.1)
    ax3.set_ylabel("Score")
    ax3.set_title("Metric Comparison")
    ax3.legend()
    ax3.grid(axis='y', alpha=0.3)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--seq_len',  type=int, default=SEQ_LEN)
    parser.add_argument('--epochs',   type=int, default=30)
    parser.add_argument('--device',   type=str, default=None)
    args = parser.parse_args()

    device = torch.device(args.device if args.device else
                          ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"Using device: {device}")

    train_ds, test_ds, test_labels, train_clip_ids, test_clip_ids = build_datasets(args.synthetic)
    print(f"Train frames: {len(train_ds)} | Test frames: {len(test_ds)}")

    cnn = load_or_create_cnn(device)

    seq_len = args.seq_len
    split_indices = split_ordered_clips(train_clip_ids)

    train_seq_ds = BoundaryAwareSequenceDataset(
        train_ds, train_clip_ids, sequence_length=seq_len,
        frame_indices=split_indices['train'],
    )
    calibration_seq_ds = BoundaryAwareSequenceDataset(
        train_ds, train_clip_ids, sequence_length=seq_len,
        frame_indices=split_indices['calibration'],
    )
    val_seq_ds = BoundaryAwareSequenceDataset(
        train_ds, train_clip_ids, sequence_length=seq_len,
        frame_indices=split_indices['validation'],
    )
    test_seq_ds = BoundaryAwareSequenceDataset(
        test_ds, test_clip_ids, sequence_length=seq_len,
    )

    print("\nProtocol split summary:")
    print(f"  Train frames:       {len(split_indices['train'])}")
    print(f"  Calibration frames: {len(split_indices['calibration'])}")
    print(f"  Validation frames:  {len(split_indices['validation'])}")
    print(f"  Test frames:        {len(test_ds)}")
    print(f"  Train windows:       {len(train_seq_ds)}")
    print(f"  Calibration windows: {len(calibration_seq_ds)}")
    print(f"  Validation windows:  {len(val_seq_ds)}")
    print(f"  Test windows:        {len(test_seq_ds)}")
    print("  Boundary checks:")
    print(f"    train={train_seq_ds.validate_no_boundary_crossing()}")
    print(f"    calibration={calibration_seq_ds.validate_no_boundary_crossing()}")
    print(f"    validation={val_seq_ds.validate_no_boundary_crossing()}")
    print(f"    test={test_seq_ds.validate_no_boundary_crossing()}")

    train_loader = DataLoader(train_seq_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
    calibration_loader = DataLoader(calibration_seq_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    val_loader = DataLoader(val_seq_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_seq_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    transformer = TemporalTransformerModel(
        latent_dim=LATENT_DIM, sequence_length=seq_len,
        num_heads=NUM_HEADS, num_layers=NUM_LAYERS,
    )
    temporal_detector = TemporalAnomalyDetector(cnn, transformer, device)

    if args.epochs > 0:
        temporal_detector.train(
            train_loader=train_loader, val_loader=val_loader,
            num_epochs=args.epochs, learning_rate=LEARNING_RATE,
            save_path=TRANSFORMER_SAVE_PATH,
        )
    else:
        print("\nSkipping Transformer training because --epochs 0 was requested.")
        print("Scores and metrics will verify protocol plumbing, not trained Transformer quality.")

    protocol_results, threshold_info = evaluate_protocol_scores(
        temporal_detector=temporal_detector,
        calibration_loader=calibration_loader,
        test_loader=test_loader,
        test_seq_ds=test_seq_ds,
        test_labels=test_labels,
        use_synthetic=args.synthetic,
    )

    print("\n" + "=" * 60)
    print("FINAL PROTOCOL-SAFE COMPARISON")
    print("=" * 60)
    print(f"{'Score':<36} {'Threshold':>12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'AUC':>10}")
    print("-" * 92)
    for key in ['cnn', 'temporal', 'interpolation', 'acceleration', 'hybrid']:
        result = protocol_results[key]
        print(
            f"{result['name']:<36} {result['threshold']:>12.6f} "
            f"{result['precision']:>10.4f} {result['recall']:>10.4f} "
            f"{result['f1']:>10.4f} {result['auc']:>10.4f}"
        )

    print("\nThreshold calibration source: calibration split only")
    print(f"  CNN threshold:         {threshold_info['cnn_threshold']:.6f}")
    print(f"  Transformer threshold: {threshold_info['temporal_threshold']:.6f}")
    print(f"  Interpolation threshold: {threshold_info['interpolation_threshold']:.6f}")
    print(f"  Acceleration threshold:  {threshold_info['acceleration_threshold']:.6f}")
    print(f"  Hybrid threshold:      {threshold_info['hybrid_threshold']:.6f}")

    print("\nDone.")
    if args.epochs > 0:
        print(f"  Transformer checkpoint: {TRANSFORMER_SAVE_PATH}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()