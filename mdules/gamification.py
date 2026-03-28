"""
Smart Bin AI — Gamification Module
Tracks carbon points, disposal streaks, badges, and disposal history
entirely through ``streamlit.session_state``.
"""

from __future__ import annotations

import datetime
from typing import Dict, List, Optional

import streamlit as st


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POINTS_MAP: Dict[str, int] = {
    "Plastic": 5,
    "Paper": 3,
    "Metal": 6,
    "Glass": 4,
}

CORRECT_BIN: Dict[str, str] = {
    "Plastic": "Blue",
    "Paper": "Green",
    "Metal": "Grey",
    "Glass": "Brown",
}

BADGE_THRESHOLDS: List[tuple[int, str]] = [
    (5, "Recycling Rookie"),
    (10, "Eco Warrior"),
    (20, "Carbon Saver"),
    (50, "Planet Protector"),
]


# ---------------------------------------------------------------------------
# State management
# ---------------------------------------------------------------------------

def init_state() -> None:
    """
    Initialise all gamification keys in ``st.session_state``.
    Safe to call on every rerun — existing values are preserved.
    """
    defaults = {
        "total_points": 0,
        "streak": 0,
        "badges_earned": [],
        "last_label": None,
        "disposal_log": [],
        "last_feedback": None,
        "frame_count": 0,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def record_disposal(label: str, correct: bool) -> Dict:
    """
    Record a disposal event and update points / streak / badges.

    Parameters
    ----------
    label : str
        Detected waste category (Plastic / Paper / Metal / Glass).
    correct : bool
        Whether the user selected the correct disposal bin.

    Returns
    -------
    dict
        ``{points_delta, streak, badge_unlocked, message, correct}``
    """
    result: Dict = {
        "points_delta": 0,
        "streak": 0,
        "badge_unlocked": None,
        "message": "",
        "correct": correct,
    }

    if correct:
        delta = POINTS_MAP.get(label, 0)
        st.session_state.total_points += delta
        st.session_state.streak += 1
        result["points_delta"] = delta
        result["streak"] = st.session_state.streak
        result["message"] = f"✅ Carbon Saved +{delta}"

        # Check for new badge unlock
        new_badge = _check_new_badge()
        if new_badge:
            result["badge_unlocked"] = new_badge
            result["message"] += f"  |  🏆 Badge unlocked: {new_badge}"
    else:
        st.session_state.streak = 0
        result["streak"] = 0
        result["message"] = "❌ Wrong Bin — Please dispose correctly"

    st.session_state.last_label = label
    st.session_state.last_feedback = result

    # Append to disposal log (keep last 20 internally, display last 5)
    log_entry = {
        "time": datetime.datetime.now().strftime("%H:%M:%S"),
        "label": label,
        "correct": "✅" if correct else "❌",
        "points": result["points_delta"],
        "streak": result["streak"],
    }
    st.session_state.disposal_log.append(log_entry)
    if len(st.session_state.disposal_log) > 20:
        st.session_state.disposal_log = st.session_state.disposal_log[-20:]

    return result


def get_badges(total_points: int) -> List[str]:
    """
    Return all badges earned at the given *total_points* level.
    """
    return [name for threshold, name in BADGE_THRESHOLDS if total_points >= threshold]


def check_wrong_disposal(label: str, selected_bin: str) -> bool:
    """
    Return ``True`` if *selected_bin* is **wrong** for *label*.

    Correct bins:
        Plastic → Blue, Paper → Green, Metal → Grey, Glass → Brown
    """
    expected = CORRECT_BIN.get(label)
    if expected is None:
        return True  # unknown label → always wrong
    return selected_bin != expected


# ---------------------------------------------------------------------------
# Helpers (private)
# ---------------------------------------------------------------------------

def _check_new_badge() -> Optional[str]:
    """
    Compare current points against badge thresholds and return the name
    of any *newly* unlocked badge, or ``None``.
    """
    earned = get_badges(st.session_state.total_points)
    for badge in earned:
        if badge not in st.session_state.badges_earned:
            st.session_state.badges_earned.append(badge)
            return badge
    return None
