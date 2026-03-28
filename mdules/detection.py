"""
Smart Bin AI — Detection Module
Loads a TFLite model exported from Google Teachable Machine and runs
real-time inference on webcam frames to classify waste into 4 categories:
Plastic, Paper, Metal, Glass.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Tuple

import numpy as np

# ---------------------------------------------------------------------------
# TFLite interpreter import — works with both the standalone tflite-runtime
# package (Linux / macOS) and the full tensorflow package (Windows fallback).
# ---------------------------------------------------------------------------
try:
    from tflite_runtime.interpreter import Interpreter  # type: ignore[import-untyped]
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter  # type: ignore[import-untyped]


def load_labels(labels_path: str) -> list[str]:
    """
    Read the labels.txt file produced by Teachable Machine.
    Each line is one class name.  Returns a list of stripped label strings.
    """
    path = Path(labels_path)
    if not path.exists():
        raise FileNotFoundError(
            f"Labels file not found at {labels_path!r}.  "
            "Export your Teachable Machine model and place labels.txt in model/."
        )
    with open(path, "r", encoding="utf-8") as fh:
        labels = [line.strip() for line in fh if line.strip()]
    return labels


def load_model(model_path: str) -> Interpreter:
    """
    Load and allocate tensors for a TFLite model.
    Returns the ready-to-use Interpreter instance.
    """
    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            f"Model file not found at {model_path!r}.  "
            "Export your Teachable Machine model and place model.tflite in model/."
        )
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter


def predict_frame(
    frame: np.ndarray,
    interpreter: Interpreter,
    labels: list[str],
    input_size: Tuple[int, int] = (224, 224),
    confidence_threshold: float = 0.55,
) -> Tuple[str, float]:
    """
    Run inference on a single BGR frame from OpenCV.

    Parameters
    ----------
    frame : np.ndarray
        BGR image (H × W × 3) from ``cv2.VideoCapture``.
    interpreter : Interpreter
        Pre-loaded TFLite interpreter.
    labels : list[str]
        Ordered list of class names matching the model's output.
    input_size : tuple[int, int]
        Spatial dimensions expected by the model (default 224×224).
    confidence_threshold : float
        Minimum softmax confidence to accept a prediction.

    Returns
    -------
    tuple[str, float]
        ``(label, confidence)`` — returns ``("Unknown", 0.0)`` when the
        top prediction falls below *confidence_threshold*.
    """
    import cv2  # local import to avoid top-level opencv dependency in tests

    # ---- pre-process --------------------------------------------------------
    resized = cv2.resize(frame, input_size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_dtype = input_details[0]["dtype"]
    input_data = np.expand_dims(rgb, axis=0)

    # Teachable Machine float32 models expect [0, 1] normalised input
    if input_dtype == np.float32:
        input_data = (input_data / 255.0).astype(np.float32)
    else:
        input_data = input_data.astype(input_dtype)

    # ---- inference ----------------------------------------------------------
    interpreter.set_tensor(input_details[0]["index"], input_data)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]["index"])[0]

    # ---- post-process -------------------------------------------------------
    # Apply softmax if output looks like raw logits (values outside 0-1)
    if output_data.max() > 1.0 or output_data.min() < 0.0:
        exp_vals = np.exp(output_data - np.max(output_data))
        output_data = exp_vals / exp_vals.sum()

    top_index = int(np.argmax(output_data))
    confidence = float(output_data[top_index])

    if confidence < confidence_threshold:
        return ("Unknown", 0.0)

    if top_index >= len(labels):
        return ("Unknown", 0.0)

    return (labels[top_index], round(confidence, 4))
