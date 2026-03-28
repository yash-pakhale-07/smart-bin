"""
Smart Bin AI — Utility Module
Drawing overlays, colour helpers, frame resizing, and central configuration.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Central configuration — single source of truth for paths & thresholds
# ---------------------------------------------------------------------------

CONFIG: Dict = {
    "model_path": "model/model.tflite",
    "labels_path": "model/labels.txt",
    "input_size": (224, 224),
    "confidence_threshold": 0.55,
    "ocr_confidence_threshold": 0.70,
    "inference_frame_skip": 5,
}


# ---------------------------------------------------------------------------
# Colour palette (BGR for OpenCV, hex for Streamlit)
# ---------------------------------------------------------------------------

_LABEL_COLOURS_BGR: Dict[str, Tuple[int, int, int]] = {
    "Plastic": (230, 160, 50),   # cerulean blue
    "Paper": (80, 190, 80),      # green
    "Metal": (160, 160, 160),    # grey
    "Glass": (50, 120, 200),     # amber / brown
    "Unknown": (100, 100, 100),
}

_LABEL_COLOURS_HEX: Dict[str, str] = {
    "Plastic": "#32A0E6",
    "Paper": "#50BE50",
    "Metal": "#A0A0A0",
    "Glass": "#C87832",
    "Unknown": "#646464",
}


def get_bin_color(label: str) -> str:
    """Return a hex colour string for the UI badge associated with *label*."""
    return _LABEL_COLOURS_HEX.get(label, "#646464")


# ---------------------------------------------------------------------------
# Frame helpers
# ---------------------------------------------------------------------------

def resize_for_display(frame: np.ndarray, max_width: int = 640) -> np.ndarray:
    """
    Proportionally resize *frame* so its width does not exceed *max_width*.
    """
    h, w = frame.shape[:2]
    if w <= max_width:
        return frame
    scale = max_width / w
    new_dim = (max_width, int(h * scale))
    return cv2.resize(frame, new_dim, interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------------------
# Overlay drawing
# ---------------------------------------------------------------------------

def draw_overlay(
    frame: np.ndarray,
    label: str,
    confidence: float,
    resin_info: Optional[Dict] = None,
) -> np.ndarray:
    """
    Draw classification info directly onto *frame* (mutates in place).

    Elements rendered:
    * **Label badge** — coloured rectangle with the class name.
    * **Confidence bar** — horizontal bar showing prediction confidence.
    * **Resin code badge** — only when *resin_info* is provided.

    Returns the annotated frame (same object).
    """
    h, w = frame.shape[:2]
    colour = _LABEL_COLOURS_BGR.get(label, (100, 100, 100))

    # ---- label badge --------------------------------------------------------
    badge_text = f"{label}  {confidence * 100:.1f}%"
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(badge_text, font, font_scale, thickness)

    pad = 10
    badge_x, badge_y = 15, 15
    cv2.rectangle(
        frame,
        (badge_x, badge_y),
        (badge_x + tw + pad * 2, badge_y + th + pad * 2),
        colour,
        cv2.FILLED,
    )
    cv2.putText(
        frame,
        badge_text,
        (badge_x + pad, badge_y + th + pad),
        font,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )

    # ---- confidence bar -----------------------------------------------------
    bar_y = badge_y + th + pad * 2 + 12
    bar_w = int((w - 60) * confidence)
    cv2.rectangle(frame, (15, bar_y), (15 + bar_w, bar_y + 10), colour, cv2.FILLED)
    cv2.rectangle(frame, (15, bar_y), (w - 45, bar_y + 10), colour, 1)

    # ---- resin code badge (bottom-left) -------------------------------------
    if resin_info is not None:
        resin_text = (
            f"Resin #{resin_info['code']}  {resin_info['type']}  "
            f"{'Recyclable' if resin_info['recyclable'] else 'Non-recyclable'}"
        )
        (rw, rh), _ = cv2.getTextSize(resin_text, font, 0.6, 1)
        ry = h - 25
        cv2.rectangle(
            frame,
            (15, ry - rh - pad),
            (15 + rw + pad * 2, ry + pad),
            (0, 0, 0),
            cv2.FILLED,
        )
        cv2.putText(
            frame,
            resin_text,
            (15 + pad, ry),
            font,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return frame
