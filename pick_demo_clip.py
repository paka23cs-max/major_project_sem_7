"""
pick_demo_clip.py — Prepares SHAREABLE sample files for your Streamlit demo,
so other people (guide, classmates, reviewers) can upload something
themselves without needing your local dataset.

Produces TWO ready-to-share samples:
  1. An ANOMALY clip  — the test clip with the most anomaly frames
  2. A NORMAL clip     — a clean clip with no labeled anomalies (from Train)

For each, it writes:
  - outputs/demo_samples/<name>.mp4          (upload via "Video file")
  - outputs/demo_samples/<name>_frames.zip   (upload via "Image sequence" —
                                               unzip first, then select all
                                               images in the file picker)

Run this from your project root:
    python pick_demo_clip.py

Then just send the .mp4 or .zip files in outputs/demo_samples/ to anyone —
e.g. over email, WhatsApp, Google Drive, or a USB stick — and they can drop
them straight into the running Streamlit app on your laptop/screen.
"""

import os
import shutil
import glob
import zipfile
import cv2
import numpy as np

UCSD_ROOT = os.path.join("data", "UCSD_Anomaly_Dataset.v1p2", "UCSDped2")
UCSD_TEST_DIR = os.path.join(UCSD_ROOT, "Test")
UCSD_TRAIN_DIR = os.path.join(UCSD_ROOT, "Train")
OUTPUT_DIR = os.path.join("outputs", "demo_samples")
FPS = 10.0  # UCSD Ped2 doesn't specify FPS; 10 gives a natural-looking walk speed


def find_clip_folders(test_dir):
    """Return list of (clip_name, frame_dir, gt_dir_or_None)."""
    all_entries = sorted(os.listdir(test_dir))
    clip_dirs = [e for e in all_entries if not e.endswith('_gt') and
                 os.path.isdir(os.path.join(test_dir, e))]
    clips = []
    for clip in clip_dirs:
        frame_dir = os.path.join(test_dir, clip)
        gt_dir = os.path.join(test_dir, clip + '_gt')
        gt_dir = gt_dir if os.path.isdir(gt_dir) else None
        clips.append((clip, frame_dir, gt_dir))
    return clips


def count_anomaly_frames(gt_dir):
    """Count frames where the ground-truth mask has any white pixel."""
    if gt_dir is None:
        return -1  # no ground truth available for this clip
    gt_files = sorted(glob.glob(os.path.join(gt_dir, "*.bmp")))
    count = 0
    for gt_path in gt_files:
        img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
        if img is not None and img.max() > 0:
            count += 1
    return count


def package_clip(clip_name, frame_paths, out_name, anomaly_frame_names=None):
    """Write an mp4 and a zip of frames for one clip into OUTPUT_DIR."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- mp4 ---
    first = cv2.imread(frame_paths[0])
    if first is None:
        raise RuntimeError(f"Could not read frame: {frame_paths[0]}")
    height, width = first.shape[:2]
    video_path = os.path.join(OUTPUT_DIR, f"{out_name}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, FPS, (width, height))
    for fp in frame_paths:
        frame = cv2.imread(fp)
        if frame is not None:
            writer.write(frame)
    writer.release()

    # --- zip of frames ---
    zip_path = os.path.join(OUTPUT_DIR, f"{out_name}_frames.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for fp in frame_paths:
            zf.write(fp, arcname=os.path.basename(fp))

    print(f"  [{clip_name}] -> {video_path}")
    print(f"  [{clip_name}] -> {zip_path}")

    if anomaly_frame_names:
        names_only = [os.path.splitext(os.path.basename(p))[0] for p in frame_paths]
        idxs = [names_only.index(n) for n in anomaly_frame_names if n in names_only]
        if idxs:
            print(f"  [{clip_name}] ground-truth anomaly spans frame index "
                  f"{min(idxs)} to {max(idxs)} (0-indexed)")

    return video_path, zip_path


def build_anomaly_samples(top_n=3):
    """Package the top-N test clips ranked by anomaly frame count."""
    if not os.path.isdir(UCSD_TEST_DIR):
        raise FileNotFoundError(f"UCSD Ped2 Test folder not found at: {UCSD_TEST_DIR}")

    clips = find_clip_folders(UCSD_TEST_DIR)
    scored = []
    for clip_name, frame_dir, gt_dir in clips:
        n_anomaly = count_anomaly_frames(gt_dir)
        n_frames = len(glob.glob(os.path.join(frame_dir, "*.tif")))
        scored.append((clip_name, frame_dir, gt_dir, n_anomaly, n_frames))

    with_gt = sorted([c for c in scored if c[3] > 0], key=lambda c: -c[3])
    if not with_gt:
        print("WARNING: no test clips with labeled anomalies found — skipping anomaly samples.")
        return

    chosen = with_gt[:top_n]
    for rank, (clip_name, frame_dir, gt_dir, n_anomaly, n_frames) in enumerate(chosen, start=1):
        print(f"\nAnomaly sample #{rank}: {clip_name} ({n_anomaly}/{n_frames} anomaly frames)")
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.tif")))
        anomaly_frame_names = []
        for fp in frame_paths:
            fname = os.path.splitext(os.path.basename(fp))[0]
            gt_path = os.path.join(gt_dir, fname + ".bmp")
            if os.path.exists(gt_path):
                gt_img = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                if gt_img is not None and gt_img.max() > 0:
                    anomaly_frame_names.append(fname)
        package_clip(clip_name, frame_paths, f"anomaly_sample_{rank}", anomaly_frame_names)


def build_normal_samples(n=3):
    """Package N different clean clips from Train (all normal by construction)."""
    if not os.path.isdir(UCSD_TRAIN_DIR):
        print("WARNING: UCSD Ped2 Train folder not found — skipping normal samples.")
        return

    train_clips = sorted([
        d for d in os.listdir(UCSD_TRAIN_DIR)
        if os.path.isdir(os.path.join(UCSD_TRAIN_DIR, d))
    ])
    if not train_clips:
        print("WARNING: no training clips found — skipping normal samples.")
        return

    # Spread picks across the available clips rather than always the first N,
    # so you get some variety in lighting/crowd density for comparison.
    step = max(1, len(train_clips) // n)
    picks = train_clips[::step][:n]

    for rank, clip_name in enumerate(picks, start=1):
        frame_dir = os.path.join(UCSD_TRAIN_DIR, clip_name)
        frame_paths = sorted(glob.glob(os.path.join(frame_dir, "*.tif")))
        print(f"\nNormal sample #{rank}: {clip_name} ({len(frame_paths)} frames, all normal)")
        package_clip(clip_name, frame_paths, f"normal_sample_{rank}")


def main():
    print("Building shareable demo samples...\n")
    build_anomaly_samples(top_n=3)
    build_normal_samples(n=3)

    print(f"\nAll done. Files are in: {OUTPUT_DIR}/")
    print("Recommended pairings for the dashboard's Clip A / Clip B comparison:")
    print("  Clip A = normal_sample_1   |  Clip B = anomaly_sample_1  (clearest contrast)")
    print("  Clip A = normal_sample_2   |  Clip B = anomaly_sample_2  (a second example)")
    print("  Clip A = normal_sample_1   |  Clip B = normal_sample_3   (normal-vs-normal — ")
    print("    a good sanity check to show scores stay low across DIFFERENT normal clips)")
    print("\nUpload instructions for your users:")
    print("  Video file mode:     upload the .mp4 directly")
    print("  Image sequence mode: unzip the _frames.zip first, then select ALL")
    print("                       extracted image files in the uploader")


if __name__ == "__main__":
    main()