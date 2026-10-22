"""
generate_pipeline_visuals.py — Produces a real screenshot for EACH stage of
the pipeline, using your actual trained CNN + transformer, for the
"data flow with intermediate outputs" slide.

Stages produced (matching the requested format):
  1. raw_frame.png              — Raw CCTV frame straight from the dataset
  2. preprocessed_frame.png     — After VideoPreprocessor (grayscale, resized, normalized)
  3. latent_heatmap.png         — The CNN's 256-d compressed latent, visualized as a 16x16 heatmap
  4. reconstruction_compare.png — Original vs. CNN-reconstructed frame, side by side
  5. score_timeline.png         — Hybrid anomaly score over a full clip, with threshold line
  6. final_flagged_frame.png    — The actual frame at the peak anomaly score, with a red border

Run from your project root:
    python generate_pipeline_visuals.py

Requires: outputs/trained_model.pth, outputs/temporal_transformer.pth,
outputs/calibration_stats.json (all already produced by train_temporal.py),
and the UCSD Ped2 dataset in its usual location.

Output: all six PNGs saved to outputs/slide_visuals/
"""

import os
import json
import glob

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from models.autoencoder import ConvolutionalAutoencoder
from models.temporal_transformer import TemporalTransformerModel
from data.preprocessing import VideoPreprocessor

OUT_DIR = os.path.join("outputs", "slide_visuals")
UCSD_TEST_DIR = os.path.join("data", "UCSD_Anomaly_Dataset.v1p2", "UCSDped2", "Test")
CNN_CHECKPOINT = "outputs/trained_model.pth"
TRANSFORMER_CHECKPOINT = "outputs/temporal_transformer.pth"
CALIBRATION_JSON = "outputs/calibration_stats.json"
FRAME_SIZE = (64, 64)


def pick_anomaly_clip():
    """Reuse the same 'most anomalous clip' logic as pick_demo_clip.py."""
    clips = sorted([d for d in os.listdir(UCSD_TEST_DIR)
                     if not d.endswith('_gt') and os.path.isdir(os.path.join(UCSD_TEST_DIR, d))])
    best_clip, best_count = None, -1
    for clip in clips:
        gt_dir = os.path.join(UCSD_TEST_DIR, clip + '_gt')
        if not os.path.isdir(gt_dir):
            continue
        count = 0
        for gt_path in glob.glob(os.path.join(gt_dir, "*.bmp")):
            img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if img is not None and img.max() > 0:
                count += 1
        if count > best_count:
            best_clip, best_count = clip, count
    return best_clip


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(CALIBRATION_JSON, "r") as f:
        cal = json.load(f)

    cnn = ConvolutionalAutoencoder(latent_dim=cal["latent_dim"])
    checkpoint = torch.load(CNN_CHECKPOINT, map_location=device)
    cnn.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    cnn.to(device).eval()

    transformer = TemporalTransformerModel(
        latent_dim=cal["latent_dim"], sequence_length=cal["sequence_length"]
    )
    t_checkpoint = torch.load(TRANSFORMER_CHECKPOINT, map_location=device)
    transformer.load_state_dict(t_checkpoint["transformer"])
    transformer.to(device).eval()

    preprocessor = VideoPreprocessor(
        target_size=FRAME_SIZE, convert_to_grayscale=True,
        normalize_method="minmax", quality_threshold=0.001,
    )

    clip_name = pick_anomaly_clip()
    print(f"Using clip: {clip_name}")
    frame_dir = os.path.join(UCSD_TEST_DIR, clip_name)
    frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.tif")))
    gt_dir = os.path.join(UCSD_TEST_DIR, clip_name + '_gt')

    # Pick a frame that IS labeled anomalous, for a meaningful demo
    anomaly_idx = None
    for i, fp in enumerate(frame_paths):
        fname = os.path.splitext(os.path.basename(fp))[0]
        gt_path = os.path.join(gt_dir, fname + '.bmp')
        if os.path.exists(gt_path):
            gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            if gt_img is not None and gt_img.max() > 0:
                anomaly_idx = i
                break
    if anomaly_idx is None:
        anomaly_idx = len(frame_paths) // 2
    print(f"Using frame index {anomaly_idx} as the featured anomaly frame")

    # ---------- STAGE 1: Raw frame ----------
    raw_bgr = cv2.imread(frame_paths[anomaly_idx])
    cv2.imwrite(os.path.join(OUT_DIR, "1_raw_frame.png"), raw_bgr)
    print("Saved: 1_raw_frame.png")

    # ---------- STAGE 2: Preprocessed frame ----------
    preprocessed = preprocessor.process_frame(raw_bgr, apply_augmentation=False, validate_quality=False)
    preprocessed_display = (np.clip(preprocessed, 0, 1) * 255).astype(np.uint8)
    preprocessed_display = cv2.resize(preprocessed_display, (256, 256), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(os.path.join(OUT_DIR, "2_preprocessed_frame.png"), preprocessed_display)
    print("Saved: 2_preprocessed_frame.png")

    frame_tensor = torch.tensor(preprocessed).unsqueeze(0).unsqueeze(0).float().to(device)

    # ---------- STAGE 3: Latent representation (heatmap) ----------
    with torch.no_grad():
        latent = cnn.encode(frame_tensor).cpu().numpy().flatten()
    latent_grid = latent.reshape(16, 16)  # 256 = 16x16
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(latent_grid, cmap="viridis")
    ax.set_title("CNN latent representation (256-d, shown as 16x16)")
    ax.axis("off")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(os.path.join(OUT_DIR, "3_latent_heatmap.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 3_latent_heatmap.png")

    # ---------- STAGE 4: Reconstruction comparison ----------
    with torch.no_grad():
        reconstruction = cnn(frame_tensor).cpu().numpy()[0, 0]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(preprocessed, cmap="gray")
    axes[0].set_title("Original (preprocessed)")
    axes[0].axis("off")
    axes[1].imshow(reconstruction, cmap="gray")
    axes[1].set_title("CNN reconstruction")
    axes[1].axis("off")
    fig.savefig(os.path.join(OUT_DIR, "4_reconstruction_compare.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 4_reconstruction_compare.png")

    # ---------- STAGE 5: Score timeline over the whole clip ----------
    seq_len = cal["sequence_length"]
    mid = seq_len // 2
    all_frames = []
    for fp in frame_paths:
        bgr = cv2.imread(fp)
        p = preprocessor.process_frame(bgr, apply_augmentation=False, validate_quality=False)
        all_frames.append(p)

    frame_stack = torch.tensor(np.stack(all_frames)).unsqueeze(1).float().to(device)
    with torch.no_grad():
        latents = cnn.encode(frame_stack)

    cnn_scores, trans_scores, centers = [], [], []
    with torch.no_grad():
        for center in range(mid, len(all_frames) - (seq_len - 1 - mid)):
            window = latents[center - mid: center + (seq_len - mid)].unsqueeze(0)
            actual_middle = window[:, mid, :]
            center_frame = frame_stack[center:center + 1]
            cnn_scores.append(cnn.reconstruction_error(center_frame).item())
            trans_scores.append(transformer.anomaly_score(window).item())
            centers.append(center)

    cnn_scores = np.array(cnn_scores)
    trans_scores = np.array(trans_scores)
    cnn_z = (cnn_scores - cal["cnn_mean"]) / cal["cnn_std"]
    trans_z = (trans_scores - cal["temporal_mean"]) / cal["temporal_std"]
    hybrid = cal["hybrid_alpha"] * cnn_z + cal["hybrid_beta"] * trans_z

    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.plot(centers, hybrid, color="#D85A30", linewidth=1.5, label="Hybrid score")
    ax.axhline(cal["hybrid_threshold"], color="gray", linestyle="--", linewidth=1, label="Calibrated threshold")
    ax.axvline(anomaly_idx, color="#1D9E75", linestyle=":", linewidth=1.5, label="Featured frame")
    ax.set_xlabel("Frame index")
    ax.set_ylabel("Hybrid anomaly score")
    ax.set_title(f"Anomaly score over clip: {clip_name}")
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(OUT_DIR, "5_score_timeline.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: 5_score_timeline.png")

    # ---------- STAGE 6: Final flagged output ----------
    peak_idx = centers[int(np.argmax(hybrid))]
    peak_frame = all_frames[peak_idx]
    peak_display = (np.clip(peak_frame, 0, 1) * 255).astype(np.uint8)
    peak_display = cv2.resize(peak_display, (256, 256), interpolation=cv2.INTER_NEAREST)
    peak_bgr = cv2.cvtColor(peak_display, cv2.COLOR_GRAY2BGR)
    bordered = cv2.copyMakeBorder(peak_bgr, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=(0, 0, 220))
    cv2.putText(bordered, "ANOMALY FLAGGED", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (0, 0, 220), 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(OUT_DIR, "6_final_flagged_frame.png"), bordered)
    print(f"Saved: 6_final_flagged_frame.png (peak score frame, index {peak_idx})")

    print(f"\nAll 6 stage screenshots saved to: {OUT_DIR}/")
    print("Drop these directly into your data-flow slide, in order 1 through 6.")


if __name__ == "__main__":
    main()