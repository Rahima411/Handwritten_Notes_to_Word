"""Planning and observation logic for the handwritten-notes agent."""

from __future__ import annotations

import re
from typing import List

import numpy as np
from PIL import Image

from .prompts import build_ocr_prompt
from .state import AgentConfig, AgentDecision, PageObservation


CHEMISTRY_HINTS = re.compile(
    r"(benzene|chloro|acetaldehyde|butadiene|alkyne|alkene|ch\d?|hcl|c#c|c=c|c=o)",
    re.IGNORECASE,
)


class AgentPlanner:
    """Observes a page and decides which OCR strategy to use."""

    def observe(self, image: Image.Image, filename: str = "uploaded_image") -> PageObservation:
        rgb = image.convert("RGB")
        width, height = rgb.size
        arr = np.array(rgb)
        aspect_ratio = width / height if height else 1.0
        orientation = "portrait" if height >= width else "landscape"

        likely_lined_paper = self._detect_lined_paper(arr)
        likely_table = self._detect_table_like_structure(arr)
        likely_chemistry = self._detect_chemistry_like_page(arr, filename)

        notes: List[str] = []
        if likely_lined_paper:
            notes.append("lined notebook paper detected")
        if likely_chemistry:
            notes.append("chemistry-style notation or diagrams likely")
        if likely_table:
            notes.append("table or grid-like lines likely")
        if orientation == "portrait":
            notes.append("portrait page, read top-to-bottom")

        return PageObservation(
            filename=filename,
            width=width,
            height=height,
            orientation=orientation,
            aspect_ratio=round(aspect_ratio, 3),
            likely_lined_paper=likely_lined_paper,
            likely_chemistry=likely_chemistry,
            likely_table=likely_table,
            notes=notes,
        )

    def decide(self, observation: PageObservation, config: AgentConfig) -> AgentDecision:
        strategy = "structured_note_transcription"
        actions = ["observe_page", "select_prompt", "run_ocr", "evaluate_output", "generate_docx"]
        rationale = ["Goal: produce a structured editable Word document from handwritten notes."]

        if observation.likely_chemistry and config.preserve_chemistry:
            strategy = "chemistry_preserving_transcription"
            rationale.append("Detected chemistry-like symbols/diagrams, so formulas and reaction arrows are preserved.")

        if observation.likely_table:
            rationale.append("Detected grid-like layout, so table-preserving OCR instructions are included.")

        if config.autonomy_level == "semi":
            rationale.append("Semi-autonomy selected: the agent will surface review items before final trust.")

        prompt = build_ocr_prompt(
            likely_chemistry=observation.likely_chemistry and config.preserve_chemistry,
            likely_table=observation.likely_table,
            output_mode=config.output_mode,
            prefer_exact_transcription=config.prefer_exact_transcription,
        )

        return AgentDecision(
            strategy=strategy,
            prompt=prompt,
            actions=actions,
            requires_review=config.autonomy_level == "semi",
            rationale=rationale,
        )

    def _detect_lined_paper(self, arr: np.ndarray) -> bool:
        gray = arr.mean(axis=2)
        row_darkness = 255 - gray.mean(axis=1)
        threshold = row_darkness.mean() + row_darkness.std()
        prominent_rows = np.where(row_darkness > threshold)[0]
        if len(prominent_rows) < 8:
            return False

        groups = []
        current = [int(prominent_rows[0])]
        for row in prominent_rows[1:]:
            if int(row) - current[-1] <= 2:
                current.append(int(row))
            else:
                groups.append(current)
                current = [int(row)]
        groups.append(current)
        return len(groups) >= 8

    def _detect_table_like_structure(self, arr: np.ndarray) -> bool:
        gray = arr.mean(axis=2)
        col_darkness = 255 - gray.mean(axis=0)
        row_darkness = 255 - gray.mean(axis=1)
        col_peaks = np.sum(col_darkness > col_darkness.mean() + 1.7 * col_darkness.std())
        row_peaks = np.sum(row_darkness > row_darkness.mean() + 1.7 * row_darkness.std())
        return bool(col_peaks > arr.shape[1] * 0.03 and row_peaks > arr.shape[0] * 0.03)

    def _detect_chemistry_like_page(self, arr: np.ndarray, filename: str) -> bool:
        # Image-only heuristic: chemistry notes often have many arrow-length strokes and formula spacing.
        # Filename and OCR text are not available yet, so this is intentionally conservative.
        red_channel = arr[:, :, 0].astype(int)
        green_channel = arr[:, :, 1].astype(int)
        blue_channel = arr[:, :, 2].astype(int)
        darker_pixels = arr.mean(axis=2) < 210
        blue_ink = (
            (blue_channel > red_channel + 8)
            & (blue_channel > green_channel + 2)
            & darker_pixels
        )
        blue_ink_ratio = np.mean(blue_ink)
        return bool(blue_ink_ratio > 0.02 or CHEMISTRY_HINTS.search(filename))
