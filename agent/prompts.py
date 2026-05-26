"""Prompt templates for the handwritten-notes agent."""

BASE_TRANSCRIPTION_RULES = """
You are converting a handwritten notebook page into structured editable content for a Word document.
Transcribe the page faithfully. Preserve headings, numbering, bullets, tables, formulas, arrows, labels, and line breaks when they carry meaning.
Use Markdown for headings and lists. Use HTML <table> markup only when the page contains a real table.
Do not invent missing content. If a word or symbol is unclear, write [unclear: best guess] instead of silently guessing.
Do not summarize unless the user explicitly requested a summary.
""".strip()

CHEMISTRY_RULES = """
This page may contain organic chemistry notes, reaction mechanisms, structural formulas, benzene rings, arrows, reagents, temperatures, and compound names.
Preserve chemical notation exactly: CH, CH2, CH3, HCl, Cu, C=C, C#C or triple-bond notation, C=O, Cl, arrows, charges, labels, temperatures, and reaction conditions.
For ring structures or drawings that cannot be represented cleanly as text, describe them in bracketed form, for example [benzene ring diagram with alternating double bonds].
Keep compound names near the reaction or structure they label.
""".strip()

STRUCTURED_OUTPUT_RULES = """
Format the output for a clean Word document:
- Use # for the page title if visible.
- Use ## for numbered sections such as (ii) Benzene.
- Keep reactions on separate lines.
- Keep explanatory labels directly below the formula or reaction they belong to.
""".strip()

EXACT_OUTPUT_RULES = """
Prioritize exact transcription over cleanup:
- Preserve the visible reading order.
- Keep rough line breaks.
- Avoid rewriting grammar or terminology.
""".strip()


def build_ocr_prompt(
    *,
    likely_chemistry: bool,
    likely_table: bool,
    output_mode: str,
    prefer_exact_transcription: bool,
) -> str:
    """Build the prompt the OCR vision model receives."""

    sections = [BASE_TRANSCRIPTION_RULES]

    if likely_chemistry:
        sections.append(CHEMISTRY_RULES)

    if likely_table:
        sections.append(
            "The page may include a table or grid. Preserve rows and columns using HTML table markup."
        )

    if prefer_exact_transcription or output_mode == "exact":
        sections.append(EXACT_OUTPUT_RULES)
    else:
        sections.append(STRUCTURED_OUTPUT_RULES)

    return "\n\n".join(sections)
