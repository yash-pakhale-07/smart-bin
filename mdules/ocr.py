"""
Smart Bin AI — OCR Module
Extracts plastic resin identification codes (♳ 1–7) from frames using
Tesseract OCR.  Only invoked when the detected label is "Plastic" and
confidence exceeds the OCR threshold.
"""

from __future__ import annotations

import re
from typing import Dict, Optional

import cv2
import numpy as np
import pytesseract


# ---------------------------------------------------------------------------
# Resin-code knowledge base
# ---------------------------------------------------------------------------
_RESIN_TABLE: Dict[int, Dict] = {
    1: {
        "code": 1,
        "type": "PET",
        "instruction": "Blue bin — curbside recyclable",
        "recyclable": True,
    },
    2: {
        "code": 2,
        "type": "HDPE",
        "instruction": "Blue bin — recyclable",
        "recyclable": True,
    },
    3: {
        "code": 3,
        "type": "PVC",
        "instruction": "Special disposal facility",
        "recyclable": False,
    },
    4: {
        "code": 4,
        "type": "LDPE",
        "instruction": "Blue bin — check locally",
        "recyclable": True,
    },
    5: {
        "code": 5,
        "type": "PP",
        "instruction": "Blue bin — recyclable",
        "recyclable": True,
    },
    6: {
        "code": 6,
        "type": "PS",
        "instruction": "Avoid — special disposal",
        "recyclable": False,
    },
    7: {
        "code": 7,
        "type": "Other",
        "instruction": "Special handling required",
        "recyclable": False,
    },
}


def extract_resin_code(frame: np.ndarray) -> Optional[int]:
    """
    Attempt to read a plastic resin code (1-7) from the centre of *frame*.

    Pipeline
    --------
    1. Crop the centre 40 % of the frame.
    2. Convert to grayscale.
    3. Apply adaptive threshold + morphological opening to clean noise.
    4. Run Tesseract OCR (``--psm 6`` — assume a single uniform block).
    5. Regex-extract the first digit in the 1–7 range.

    Returns
    -------
    int | None
        The detected resin code, or ``None`` if nothing was found.
    """
    h, w = frame.shape[:2]

    # ---- centre crop (40 %) -------------------------------------------------
    cx, cy = w // 2, h // 2
    crop_w, crop_h = int(w * 0.4), int(h * 0.4)
    x1, y1 = max(cx - crop_w // 2, 0), max(cy - crop_h // 2, 0)
    x2, y2 = min(cx + crop_w // 2, w), min(cy + crop_h // 2, h)
    roi = frame[y1:y2, x1:x2]

    # ---- pre-process --------------------------------------------------------
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    # OTSU binarisation works well for recycling symbols
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # morphological opening to remove speckle noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

    # ---- OCR ----------------------------------------------------------------
    custom_config = r"--oem 3 --psm 6 -c tessedit_char_whitelist=1234567"
    text = pytesseract.image_to_string(cleaned, config=custom_config)

    # ---- extract first valid digit 1-7 --------------------------------------
    match = re.search(r"[1-7]", text)
    if match:
        return int(match.group())
    return None


def get_resin_info(code: int) -> Dict:
    """
    Return a dict with resin-code metadata.

    Keys: ``code``, ``type``, ``instruction``, ``recyclable``.

    Falls back to code-7 (Other) for out-of-range values.
    """
    return _RESIN_TABLE.get(code, _RESIN_TABLE[7])
