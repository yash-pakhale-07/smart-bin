"""
Smart Bin AI — Streamlit Entry Point
Real-time waste classification, plastic resin-code OCR, and gamified disposal.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import time
from typing import Dict, Optional

import av
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode

from modules.detection import load_model, load_labels, predict_frame
from modules.ocr import extract_resin_code, get_resin_info
from modules.gamification import (
    init_state,
    record_disposal,
    get_badges,
    check_wrong_disposal,
    CORRECT_BIN,
    POINTS_MAP,
)
from modules.utils import (
    CONFIG,
    draw_overlay,
    get_bin_color,
    resize_for_display,
)


# ---------------------------------------------------------------------------
# Page config & custom CSS
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Smart Bin AI — Waste Segregation",
    page_icon="♻️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    /* ---- dark-mode gradient background --------------------------------- */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #e0e0e0;
    }
    /* ---- glass card ------------------------------------------------------ */
    .glass-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 14px;
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
    }
    /* ---- badge pills ----------------------------------------------------- */
    .badge-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        margin: 3px 4px;
        color: #fff;
    }
    .badge-rookie   { background: #4ade80; }
    .badge-warrior  { background: #38bdf8; }
    .badge-saver    { background: #facc15; color: #1a1a1a; }
    .badge-protector{ background: #c084fc; }
    /* ---- header ---------------------------------------------------------- */
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg,#4ade80,#38bdf8,#c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2px;
    }
    .hero-sub {
        font-size: 1rem;
        color: #a0a0c0;
        margin-bottom: 18px;
    }
    /* ---- metrics --------------------------------------------------------- */
    .big-metric {
        font-size: 3rem;
        font-weight: 900;
        line-height: 1;
        margin-bottom: 0;
    }
    .metric-label {
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: #a0a0c0;
    }
    /* ---- detection badge ------------------------------------------------- */
    .det-badge {
        display: inline-block;
        padding: 6px 18px;
        border-radius: 10px;
        font-weight: 700;
        font-size: 1.1rem;
        color: #fff;
        margin-right: 8px;
    }
    .resin-panel {
        background: rgba(50,160,230,0.15);
        border: 1px solid rgba(50,160,230,0.3);
        border-radius: 12px;
        padding: 14px 18px;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ---------------------------------------------------------------------------
# Cached resources
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading TFLite model …")
def _get_model():
    """Load model + labels once and cache across reruns."""
    interpreter = load_model(CONFIG["model_path"])
    labels = load_labels(CONFIG["labels_path"])
    return interpreter, labels


# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------
init_state()

# Extra keys for inter-frame communication
for _key, _default in [
    ("det_label", "Waiting…"),
    ("det_conf", 0.0),
    ("resin_info", None),
    ("webcam_active", False),
]:
    if _key not in st.session_state:
        st.session_state[_key] = _default


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown('<p class="hero-title">♻️ Smart Bin AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Real-time waste detection · Resin-code OCR · Carbon gamification</p>',
    unsafe_allow_html=True,
)

# ---------------------------------------------------------------------------
# Layout: two columns (feed | dashboard)
# ---------------------------------------------------------------------------
col_feed, col_dash = st.columns([2, 1], gap="large")


# ---------------------------------------------------------------------------
# Video processor for streamlit-webrtc
# ---------------------------------------------------------------------------

class WasteDetector(VideoProcessorBase):
    """Process each video frame: classify waste, run OCR on plastics."""

    def __init__(self) -> None:
        self.frame_count: int = 0
        self.label: str = "Waiting…"
        self.confidence: float = 0.0
        self.resin_info: Optional[Dict] = None
        self._interpreter = None
        self._labels = None

    def _ensure_model(self):
        if self._interpreter is None:
            self._interpreter, self._labels = _get_model()

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        self.frame_count += 1

        skip = CONFIG["inference_frame_skip"]
        if self.frame_count % skip == 0:
            self._ensure_model()
            label, conf = predict_frame(
                img,
                self._interpreter,
                self._labels,
                CONFIG["input_size"],
                CONFIG["confidence_threshold"],
            )
            self.label = label
            self.confidence = conf

            # OCR for plastics
            if (
                label == "Plastic"
                and conf >= CONFIG["ocr_confidence_threshold"]
            ):
                code = extract_resin_code(img)
                if code is not None:
                    self.resin_info = get_resin_info(code)
                else:
                    self.resin_info = None
            else:
                self.resin_info = None

        # Draw overlay
        annotated = draw_overlay(img, self.label, self.confidence, self.resin_info)
        return av.VideoFrame.from_ndarray(annotated, format="bgr24")


# ---------------------------------------------------------------------------
# Feed column
# ---------------------------------------------------------------------------

with col_feed:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)

    # Try streamlit-webrtc first
    model_ready = True
    try:
        _get_model()
    except FileNotFoundError:
        model_ready = False

    if not model_ready:
        st.warning(
            "⚠️ **Model not found.**  Place `model.tflite` and `labels.txt` "
            "in the `model/` folder. See README for instructions.",
            icon="📦",
        )

    ctx = webrtc_streamer(
        key="smart-bin-feed",
        mode=WebRtcMode.SENDRECV,
        video_processor_factory=WasteDetector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
        rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    )

    # If the processor is running, pull live state for the sidebar
    if ctx.video_processor:
        st.session_state.det_label = ctx.video_processor.label
        st.session_state.det_conf = ctx.video_processor.confidence
        st.session_state.resin_info = ctx.video_processor.resin_info
        st.session_state.webcam_active = True

    # ---- Detection badge below the feed ----
    det_label = st.session_state.det_label
    det_conf = st.session_state.det_conf
    resin_info = st.session_state.resin_info

    if det_label and det_label != "Waiting…" and det_label != "Unknown":
        hex_col = get_bin_color(det_label)
        st.markdown(
            f'<span class="det-badge" style="background:{hex_col}">'
            f'{det_label}</span>'
            f'<span style="font-size:1.1rem;color:#ccc">'
            f'{det_conf * 100:.1f}% confidence</span>',
            unsafe_allow_html=True,
        )

        correct_bin = CORRECT_BIN.get(det_label, "Unknown")
        st.info(f"🗑️ **Suggested bin:** {correct_bin}", icon="🗑️")

        # Resin code  panel (Plastic only)
        if resin_info is not None:
            st.markdown('<div class="resin-panel">', unsafe_allow_html=True)
            r_icon = "♻️" if resin_info["recyclable"] else "⚠️"
            st.markdown(
                f"**Resin Code #{resin_info['code']}** — "
                f"**{resin_info['type']}** &nbsp; {r_icon}"
            )
            st.markdown(f"_{resin_info['instruction']}_")
            st.markdown("</div>", unsafe_allow_html=True)
    elif det_label == "Unknown":
        st.markdown(
            '<span class="det-badge" style="background:#555">Unknown</span>'
            '<span style="color:#999"> Low confidence — move item closer</span>',
            unsafe_allow_html=True,
        )
    else:
        st.caption("Start the webcam above to begin detecting waste.")

    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Disposal confirmation ---------------------------------------------
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.subheader("Confirm Disposal")
    bin_options = ["Blue", "Green", "Grey", "Brown"]
    sel_bin = st.selectbox(
        "Which bin did you use?",
        bin_options,
        index=0,
        key="sel_bin",
    )
    if st.button("✅ Confirm Disposal", use_container_width=True, type="primary"):
        if det_label and det_label not in ("Waiting…", "Unknown"):
            wrong = check_wrong_disposal(det_label, sel_bin)
            result = record_disposal(det_label, correct=not wrong)
            if result["correct"]:
                st.success(result["message"])
                if result["badge_unlocked"]:
                    st.balloons()
                    st.success(f"🏆 Badge unlocked: **{result['badge_unlocked']}**")
            else:
                st.error(result["message"])
        else:
            st.warning("Detect an item first before confirming disposal.")
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Dashboard column
# ---------------------------------------------------------------------------

with col_dash:
    # ---- Carbon Points ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Carbon Points</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="big-metric" style="color:#4ade80">'
        f'{st.session_state.total_points}</p>',
        unsafe_allow_html=True,
    )
    last_fb = st.session_state.last_feedback
    if last_fb and last_fb.get("points_delta"):
        st.metric(
            label="Last disposal",
            value=f"+{last_fb['points_delta']}",
            delta=f"+{last_fb['points_delta']} pts",
        )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Streak ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    streak = st.session_state.streak
    fire = "🔥" if streak >= 3 else ""
    st.markdown('<p class="metric-label">Disposal Streak</p>', unsafe_allow_html=True)
    st.markdown(
        f'<p class="big-metric" style="color:#facc15">'
        f'{streak} {fire}</p>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Badges ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Badges Earned</p>', unsafe_allow_html=True)
    earned = st.session_state.badges_earned
    if earned:
        badge_html = ""
        badge_classes = {
            "Recycling Rookie": "badge-rookie",
            "Eco Warrior": "badge-warrior",
            "Carbon Saver": "badge-saver",
            "Planet Protector": "badge-protector",
        }
        for b in earned:
            cls = badge_classes.get(b, "badge-rookie")
            badge_html += f'<span class="badge-pill {cls}">🏅 {b}</span>'
        st.markdown(badge_html, unsafe_allow_html=True)
    else:
        st.caption("No badges yet — keep recycling!")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Last Feedback ----
    if last_fb:
        if last_fb["correct"]:
            st.success(last_fb["message"])
        else:
            st.error(last_fb["message"])

    # ---- Disposal History ---
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Recent Disposals</p>', unsafe_allow_html=True)
    log = st.session_state.disposal_log
    if log:
        import pandas as pd

        df = pd.DataFrame(log[-5:])
        df.columns = ["Time", "Item", "Result", "Points", "Streak"]
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("No disposals recorded yet.")
    st.markdown("</div>", unsafe_allow_html=True)

    # ---- Bin Guide ----
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<p class="metric-label">Bin Guide</p>', unsafe_allow_html=True)
    guide_cols = st.columns(4)
    bin_emojis = {"Plastic": "🔵", "Paper": "🟢", "Metal": "⚪", "Glass": "🟤"}
    for i, (cat, colour) in enumerate(CORRECT_BIN.items()):
        with guide_cols[i]:
            st.markdown(
                f"**{bin_emojis[cat]} {cat}**\n\n{colour} bin\n\n+{POINTS_MAP[cat]} pts"
            )
    st.markdown("</div>", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------
st.markdown("---")
st.caption(
    "Smart Bin AI · Built with Streamlit, TFLite & Tesseract · "
    "♻️ Every item sorted is a step toward a greener planet"
)
