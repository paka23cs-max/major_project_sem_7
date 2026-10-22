"""
streamlit_demo.py — Interactive anomaly detection dashboard.

Upgrades over the basic version:
  - Upload TWO clips (e.g. one normal, one anomalous) and compare their score
    curves on the same interactive chart — this is the strongest demo
    narrative given the calibration-generalization limitation we found.
  - Plotly chart: hover tooltips, click any point to jump the frame viewer
    to that exact frame.
  - Threshold slider: adjust sensitivity live. Flags/counts recompute
    instantly from already-computed scores — NO re-running the model.
  - Frame viewer: red border on anomalous frames, prev/next/first/last
    navigation, synced to whatever point you clicked on the chart.
  - Anomaly thumbnail grid — click any thumbnail to jump straight to it.
  - Export current session's scores as JSON or CSV.

SETUP:
    pip install "streamlit>=1.35" opencv-python-headless plotly pillow

RUN (from your project root):
    streamlit run streamlit_demo.py

REQUIRES:
    outputs/trained_model.pth
    outputs/temporal_transformer.pth
    outputs/calibration_stats.json   (generate via the calibration_stats.json
                                       patch in train_temporal.py, run once)
"""

import os
import json
import tempfile

import cv2
import numpy as np
import pandas as pd
import torch
import streamlit as st
import plotly.graph_objects as go
from PIL import Image

from models.autoencoder import ConvolutionalAutoencoder
from models.temporal_transformer import TemporalTransformerModel
from data.preprocessing import VideoPreprocessor

CNN_CHECKPOINT = "outputs/trained_model.pth"
TRANSFORMER_CHECKPOINT = "outputs/temporal_transformer.pth"
CALIBRATION_JSON = "outputs/calibration_stats.json"
FRAME_SIZE = (64, 64)  # (W, H) — must match training preprocessing

_preprocessor = VideoPreprocessor(
    target_size=FRAME_SIZE,
    convert_to_grayscale=True,
    normalize_method="minmax",
    quality_threshold=0.001,
)

st.set_page_config(page_title="Anomaly Detection Dashboard", page_icon="🎥", layout="wide")


# ----------------------------------------------------------------------
# Model + calibration loading (cached — only runs once per session)
# ----------------------------------------------------------------------
@st.cache_resource
def load_everything():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with open(CALIBRATION_JSON, "r") as f:
        cal = json.load(f)

    cnn = ConvolutionalAutoencoder(latent_dim=cal["latent_dim"])
    checkpoint = torch.load(CNN_CHECKPOINT, map_location=device)
    cnn.load_state_dict(checkpoint.get("model_state_dict", checkpoint))
    cnn.to(device).eval()
    for p in cnn.parameters():
        p.requires_grad = False

    transformer = TemporalTransformerModel(
        latent_dim=cal["latent_dim"], sequence_length=cal["sequence_length"]
    )
    t_checkpoint = torch.load(TRANSFORMER_CHECKPOINT, map_location=device)
    transformer.load_state_dict(t_checkpoint["transformer"])
    transformer.to(device).eval()

    return cnn, transformer, cal, device


# ----------------------------------------------------------------------
# Frame loading (uses the project's real preprocessing pipeline)
# ----------------------------------------------------------------------
def preprocess_frame(frame_bgr: np.ndarray):
    try:
        return _preprocessor.process_frame(frame_bgr, apply_augmentation=False, validate_quality=True)
    except ValueError:
        return None


def frames_from_video(video_bytes: bytes) -> list:
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    cap = cv2.VideoCapture(tmp_path)
    frames, skipped = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        p = preprocess_frame(frame)
        if p is not None:
            frames.append(p)
        else:
            skipped += 1
    cap.release()
    os.unlink(tmp_path)
    if skipped:
        st.warning(f"Skipped {skipped} low-quality frame(s).")
    return frames


def frames_from_images(uploaded_files) -> list:
    frames, skipped = [], 0
    for uf in sorted(uploaded_files, key=lambda f: f.name):
        file_bytes = np.frombuffer(uf.read(), np.uint8)
        frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        p = preprocess_frame(frame_bgr)
        if p is not None:
            frames.append(p)
        else:
            skipped += 1
    if skipped:
        st.warning(f"Skipped {skipped} low-quality frame(s).")
    return frames


# ----------------------------------------------------------------------
# Score computation
# ----------------------------------------------------------------------
@torch.no_grad()
def score_sequence(frames: list, cnn, transformer, device, seq_len: int):
    mid = seq_len // 2
    n = len(frames)
    if n < seq_len:
        st.error(f"Need at least {seq_len} frames, got {n}.")
        return None

    frame_tensor = torch.tensor(np.stack(frames)).unsqueeze(1).float().to(device)
    latents = cnn.encode(frame_tensor)

    cnn_scores, trans_scores, interp_scores, accel_scores, centers = [], [], [], [], []
    for center in range(mid, n - (seq_len - 1 - mid)):
        window_latents = latents[center - mid: center + (seq_len - mid)].unsqueeze(0)
        actual_middle = window_latents[:, mid, :]

        center_frame = frame_tensor[center:center + 1]
        cnn_scores.append(cnn.reconstruction_error(center_frame).item())
        trans_scores.append(transformer.anomaly_score(window_latents).item())

        prev_latent = window_latents[:, mid - 1, :]
        next_latent = window_latents[:, mid + 1, :]
        interp_pred = 0.5 * (prev_latent + next_latent)
        interp_scores.append(torch.mean((interp_pred - actual_middle) ** 2, dim=1).item())

        accel = next_latent - 2 * actual_middle + prev_latent
        accel_scores.append(torch.mean(accel ** 2, dim=1).item())

        centers.append(center)

    return {
        "centers": np.array(centers),
        "cnn": np.array(cnn_scores),
        "transformer": np.array(trans_scores),
        "interpolation": np.array(interp_scores),
        "acceleration": np.array(accel_scores),
    }


def normalize_hybrid(scores: dict, cal: dict) -> np.ndarray:
    cnn_z = (scores["cnn"] - cal["cnn_mean"]) / cal["cnn_std"]
    trans_z = (scores["transformer"] - cal["temporal_mean"]) / cal["temporal_std"]
    return cal["hybrid_alpha"] * cnn_z + cal["hybrid_beta"] * trans_z


def process_upload(frames, clip_label, cnn, transformer, cal, device, seq_len):
    raw = score_sequence(frames, cnn, transformer, device, seq_len)
    if raw is None:
        return None
    hybrid = normalize_hybrid(raw, cal)
    return {
        "label": clip_label,
        "frames": frames,
        "raw": raw,
        "hybrid": hybrid,
        "current_frame": int(raw["centers"][0]),
    }


# ----------------------------------------------------------------------
# UI helpers
# ----------------------------------------------------------------------
def frame_to_display_image(frame_float: np.ndarray, is_anomaly: bool) -> Image.Image:
    arr = (np.clip(frame_float, 0, 1) * 255).astype(np.uint8)
    img = Image.fromarray(arr).convert("RGB").resize((256, 256), Image.NEAREST)
    if is_anomaly:
        bordered = Image.new("RGB", (img.width + 12, img.height + 12), (220, 40, 40))
        bordered.paste(img, (6, 6))
        img = bordered
    return img


def suggest_threshold(clip_data: dict):
    """
    When exactly two clips are loaded, find the threshold that best separates
    them: treat the lower-median-scoring clip as 'normal' (label 0) and the
    higher-median-scoring clip as 'anomalous' (label 1), then pick the
    threshold maximizing TPR - FPR (Youden's J statistic).

    This is an ad-hoc, session-specific suggestion based on whichever two
    clips are currently loaded — NOT a substitute for the calibrated
    training threshold, and it assumes the two clips really do differ in
    kind. Returns None if fewer/more than 2 clips are loaded.
    """
    if len(clip_data) != 2:
        return None

    slots = list(clip_data.keys())
    med0 = np.median(clip_data[slots[0]]["hybrid"])
    med1 = np.median(clip_data[slots[1]]["hybrid"])
    normal_slot, anomaly_slot = (slots[0], slots[1]) if med0 <= med1 else (slots[1], slots[0])

    normal_scores = clip_data[normal_slot]["hybrid"]
    anomaly_scores = clip_data[anomaly_slot]["hybrid"]

    candidates = np.unique(np.concatenate([normal_scores, anomaly_scores]))
    best_j, best_t, best_tpr, best_fpr = -np.inf, candidates[0], 0.0, 0.0
    for t in candidates:
        tpr = np.mean(anomaly_scores > t)
        fpr = np.mean(normal_scores > t)
        j = tpr - fpr
        if j > best_j:
            best_j, best_t, best_tpr, best_fpr = j, t, tpr, fpr

    accuracy = (
        np.mean(anomaly_scores > best_t) * len(anomaly_scores)
        + np.mean(normal_scores <= best_t) * len(normal_scores)
    ) / (len(anomaly_scores) + len(normal_scores))

    return {
        "threshold": float(best_t),
        "normal_slot": normal_slot,
        "anomaly_slot": anomaly_slot,
        "tpr": float(best_tpr),
        "fpr": float(best_fpr),
        "accuracy": float(accuracy),
    }


def build_comparison_chart(clip_data: dict, threshold: float, suggested: dict = None) -> go.Figure:
    fig = go.Figure()
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for i, (key, data) in enumerate(clip_data.items()):
        centers = data["raw"]["centers"]
        hybrid = data["hybrid"]
        fig.add_trace(go.Scatter(
            x=centers, y=hybrid, mode="lines", name=data["label"],
            line=dict(color=colors[i % len(colors)], width=2),
            customdata=[key] * len(centers),
            hovertemplate=f"{data['label']}<br>Frame: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>",
        ))
        flagged = hybrid > threshold
        if flagged.any():
            fig.add_trace(go.Scatter(
                x=centers[flagged], y=hybrid[flagged], mode="markers",
                name=f"{data['label']} — flagged",
                marker=dict(color="red", size=6, symbol="circle"),
                customdata=[key] * int(flagged.sum()),
                hovertemplate=f"Flagged<br>Frame: %{{x}}<br>Score: %{{y:.3f}}<extra></extra>",
            ))
    fig.add_hline(y=threshold, line_dash="dash", line_color="gray",
                  annotation_text=f"Calibrated threshold: {threshold:.3f}", annotation_position="right")
    if suggested is not None:
        fig.add_hline(y=suggested["threshold"], line_dash="dot", line_color="#2ca02c",
                      annotation_text=f"Suggested (best separates loaded clips): {suggested['threshold']:.3f}",
                      annotation_position="left")
    fig.update_layout(
        title="Hybrid anomaly score — click a point to jump the frame viewer",
        xaxis_title="Frame index", yaxis_title="Hybrid anomaly score (z-score)",
        hovermode="x unified", height=450, legend=dict(x=0.01, y=0.99),
    )
    return fig


def export_json(clip_data: dict, threshold: float) -> str:
    payload = {}
    for key, data in clip_data.items():
        flags = (data["hybrid"] > threshold).tolist()
        payload[data["label"]] = {
            "frame_indices": data["raw"]["centers"].tolist(),
            "hybrid_scores": data["hybrid"].tolist(),
            "cnn_scores": data["raw"]["cnn"].tolist(),
            "transformer_scores": data["raw"]["transformer"].tolist(),
            "flags": flags,
            "threshold": threshold,
            "anomaly_count": int(sum(flags)),
        }
    return json.dumps(payload, indent=2)


def export_csv(clip_data: dict, threshold: float) -> str:
    rows = []
    for key, data in clip_data.items():
        for i, center in enumerate(data["raw"]["centers"]):
            rows.append({
                "clip": data["label"],
                "frame_index": int(center),
                "cnn_score": data["raw"]["cnn"][i],
                "transformer_score": data["raw"]["transformer"][i],
                "interpolation_score": data["raw"]["interpolation"][i],
                "acceleration_score": data["raw"]["acceleration"][i],
                "hybrid_score": data["hybrid"][i],
                "flagged": bool(data["hybrid"][i] > threshold),
            })
    return pd.DataFrame(rows).to_csv(index=False)


# ----------------------------------------------------------------------
# Main app
# ----------------------------------------------------------------------
st.title("🎥 CCTV Anomaly Detection — Interactive Dashboard")
st.caption("Upload one or two clips, compare hybrid anomaly scores live, and adjust the "
           "threshold without re-running the model.")

if not (os.path.exists(CNN_CHECKPOINT) and os.path.exists(TRANSFORMER_CHECKPOINT)
        and os.path.exists(CALIBRATION_JSON)):
    st.error("Missing checkpoints or calibration_stats.json — run train_temporal.py first.")
    st.stop()

cnn, transformer, cal, device = load_everything()
seq_len = cal["sequence_length"]

if "clip_data" not in st.session_state:
    st.session_state.clip_data = {}
if "threshold" not in st.session_state:
    st.session_state.threshold = cal["hybrid_threshold"]

with st.sidebar:
    st.header("Upload clips")
    input_mode = st.radio("Input type", ["Video file", "Image sequence"], horizontal=True)

    for slot, label_default in [("A", "Clip A"), ("B", "Clip B (optional, for comparison)")]:
        st.subheader(label_default)
        clip_label = st.text_input(f"Label for {slot}", value=label_default, key=f"label_{slot}")
        if input_mode == "Video file":
            uploaded = st.file_uploader(f"Upload video ({slot})", type=["mp4", "avi", "mov"], key=f"vid_{slot}")
            if uploaded is not None and st.button(f"Analyze {slot}", key=f"btn_{slot}"):
                with st.spinner(f"Processing {slot}..."):
                    frames = frames_from_video(uploaded.read())
                    result = process_upload(frames, clip_label, cnn, transformer, cal, device, seq_len)
                    if result:
                        st.session_state.clip_data[slot] = result
                        st.success(f"{slot} analyzed: {len(frames)} frames")
        else:
            uploaded_files = st.file_uploader(
                f"Upload frame images ({slot})", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
                accept_multiple_files=True, key=f"img_{slot}",
            )
            if uploaded_files and st.button(f"Analyze {slot}", key=f"btn_{slot}"):
                with st.spinner(f"Processing {slot}..."):
                    frames = frames_from_images(uploaded_files)
                    result = process_upload(frames, clip_label, cnn, transformer, cal, device, seq_len)
                    if result:
                        st.session_state.clip_data[slot] = result
                        st.success(f"{slot} analyzed: {len(frames)} frames")

    if st.session_state.clip_data:
        st.divider()
        if st.button("Clear all clips"):
            st.session_state.clip_data = {}
            st.rerun()

if not st.session_state.clip_data:
    st.info("Upload at least one clip in the sidebar to begin.")
    st.stop()

# --- Clear status of what's actually loaded, so a forgotten "Analyze" click
#     on one slot is obvious instead of silently missing ---
status_cols = st.columns(2)
for idx, slot in enumerate(["A", "B"]):
    with status_cols[idx]:
        if slot in st.session_state.clip_data:
            d = st.session_state.clip_data[slot]
            st.success(f"Clip {slot} loaded — \"{d['label']}\" "
                       f"({len(d['raw']['centers'])} scored frames)")
        else:
            st.warning(f"Clip {slot} not loaded yet — upload a file AND click "
                       f"its \"Analyze {slot}\" button in the sidebar.")

st.divider()

# --- Threshold slider (client-side recompute, no re-inference) ---
st.subheader("🎚️ Threshold")

suggested = suggest_threshold(st.session_state.clip_data)
if suggested is not None:
    with st.container(border=True):
        st.markdown(
            f"**Suggested threshold for these two clips: `{suggested['threshold']:.3f}`** "
            f"(session-specific, not the calibrated training threshold)"
        )
        st.caption(
            f"Assumes \"{st.session_state.clip_data[suggested['normal_slot']]['label']}\" is the normal clip "
            f"and \"{st.session_state.clip_data[suggested['anomaly_slot']]['label']}\" is the anomalous one "
            f"(lower vs. higher median score). At this threshold: "
            f"TPR={suggested['tpr']:.1%}, FPR={suggested['fpr']:.1%}, "
            f"accuracy={suggested['accuracy']:.1%} on these two clips only."
        )
        if st.button("Apply suggested threshold"):
            st.session_state.threshold = suggested["threshold"]
            st.rerun()

col1, col2 = st.columns([3, 1])
with col1:
    new_threshold = st.slider(
        "Hybrid anomaly threshold", min_value=-3.0, max_value=20.0,
        value=float(st.session_state.threshold), step=0.05,
    )
    st.session_state.threshold = new_threshold
with col2:
    if st.button("Reset to calibrated"):
        st.session_state.threshold = cal["hybrid_threshold"]
        st.rerun()
st.caption(f"Calibrated threshold from training: {cal['hybrid_threshold']:.3f} "
           f"(see Threats to Validity note — this may not generalize to all clips)")

# --- Metrics row ---
cols = st.columns(len(st.session_state.clip_data) * 3)
i = 0
for slot, data in st.session_state.clip_data.items():
    flags = data["hybrid"] > st.session_state.threshold
    cols[i].metric(f"{data['label']} — frames", len(data["raw"]["centers"]))
    cols[i + 1].metric(f"{data['label']} — flagged", int(flags.sum()))
    cols[i + 2].metric(f"{data['label']} — flag rate", f"{flags.mean() * 100:.1f}%")
    i += 3

st.divider()

# --- Comparison chart ---
fig = build_comparison_chart(st.session_state.clip_data, st.session_state.threshold, suggested)
event = st.plotly_chart(fig, use_container_width=True, on_select="rerun", key="main_chart")

active_slot = None
if event and event.get("selection", {}).get("points"):
    point = event["selection"]["points"][0]
    clicked_frame = int(point["x"])
    clicked_slot = point.get("customdata")
    if isinstance(clicked_slot, list):
        clicked_slot = clicked_slot[0] if clicked_slot else None
    if clicked_slot in st.session_state.clip_data:
        st.session_state.clip_data[clicked_slot]["current_frame"] = clicked_frame
        active_slot = clicked_slot

st.divider()

# --- Frame viewer ---
st.subheader("🖼️ Frame Viewer")
slot_options = list(st.session_state.clip_data.keys())
selected_slot = st.selectbox(
    "Viewing clip", slot_options,
    format_func=lambda s: st.session_state.clip_data[s]["label"],
    index=slot_options.index(active_slot) if active_slot in slot_options else 0,
)
data = st.session_state.clip_data[selected_slot]
centers = data["raw"]["centers"]
frames = data["frames"]

col1, col2, col3, col4, col5 = st.columns([1, 1, 1, 1, 3])
current = data["current_frame"]
if col1.button("⏮️ First"):
    current = int(centers[0])
if col2.button("◀️ Prev"):
    idx = max(0, np.searchsorted(centers, current) - 1)
    current = int(centers[idx])
if col3.button("Next ▶️"):
    idx = min(len(centers) - 1, np.searchsorted(centers, current) + 1)
    current = int(centers[idx])
if col4.button("Last ⏭️"):
    current = int(centers[-1])
data["current_frame"] = current

slider_val = col5.select_slider("Jump to frame (center index)", options=list(centers), value=current)
if slider_val != current:
    data["current_frame"] = int(slider_val)
    current = int(slider_val)

pos = np.searchsorted(centers, current)
pos = min(max(pos, 0), len(centers) - 1)
is_anomaly = bool(data["hybrid"][pos] > st.session_state.threshold)

img_col, info_col = st.columns([1, 1])
with img_col:
    st.image(frame_to_display_image(frames[current], is_anomaly), caption=f"Frame {current}")
with info_col:
    st.metric("Hybrid score", f"{data['hybrid'][pos]:.3f}")
    st.metric("Status", "🚨 ANOMALY" if is_anomaly else "✅ Normal")
    st.metric("CNN reconstruction", f"{data['raw']['cnn'][pos]:.6f}")
    st.metric("Transformer temporal", f"{data['raw']['transformer'][pos]:.6f}")

st.divider()

# --- Anomaly thumbnail grid ---
flags = data["hybrid"] > st.session_state.threshold
flagged_centers = centers[flags]
st.subheader(f"🔴 Flagged frames — {data['label']} ({flags.sum()} / {len(flags)})")
if flagged_centers.size > 0:
    top_k = min(10, len(flagged_centers))
    order = np.argsort(-data["hybrid"][flags])[:top_k]
    thumb_cols = st.columns(5)
    for j, idx in enumerate(order):
        frame_idx = int(flagged_centers[idx])
        with thumb_cols[j % 5]:
            thumb = Image.fromarray((np.clip(frames[frame_idx], 0, 1) * 255).astype(np.uint8))
            thumb.thumbnail((120, 120))
            if st.button(f"Frame {frame_idx}", key=f"thumb_{selected_slot}_{frame_idx}"):
                data["current_frame"] = frame_idx
                st.rerun()
            st.image(thumb, use_container_width=True)
else:
    st.info("No frames exceeded the current threshold.")

st.divider()

# --- Export ---
st.subheader("⬇️ Export")
ecol1, ecol2 = st.columns(2)
with ecol1:
    st.download_button("📄 Download JSON", export_json(st.session_state.clip_data, st.session_state.threshold),
                        file_name="anomaly_analysis.json", mime="application/json", use_container_width=True)
with ecol2:
    st.download_button("📊 Download CSV", export_csv(st.session_state.clip_data, st.session_state.threshold),
                        file_name="anomaly_scores.csv", mime="text/csv", use_container_width=True)