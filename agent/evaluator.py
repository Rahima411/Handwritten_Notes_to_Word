"""Heuristic evaluation for agent review and human-in-the-loop prompts."""

import re
from typing import List

from .state import PageObservation


UNCLEAR_PATTERNS = (
    "[unclear",
    "?",
    "illegible",
)


def evaluate_output(raw_text: str, observation: PageObservation) -> List[str]:
    """Return review items the UI should show to the user."""

    feedback: List[str] = []
    lowered = raw_text.lower()

    if any(pattern in lowered for pattern in UNCLEAR_PATTERNS):
        feedback.append("Review unclear words or symbols marked by the OCR model.")

    if observation.likely_chemistry:
        chemistry_tokens = re.findall(r"\b(?:CH\d?|HCl|Cu|Cl|C=O|C=C|C#C)\b", raw_text)
        if len(chemistry_tokens) < 2:
            feedback.append("Chemistry notation may be under-detected; compare formulas against the image.")
        feedback.append("Confirm reaction arrows, temperatures, and compound labels before final use.")

    if observation.likely_table and "<table" not in lowered:
        feedback.append("A table-like layout was detected, but no structured table was produced.")

    if len(raw_text.strip()) < 40:
        feedback.append("The extracted text is very short; the page may need a clearer image or manual correction.")

    return feedback
