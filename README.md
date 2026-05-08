---
title: handwriting-to-word-agent
app_file: app.py
sdk: streamlit
---

# Agentic Handwriting to Word Converter

This project converts handwritten notebook images into editable Microsoft Word
documents. It now includes a lightweight agentic layer that turns the original
reactive pipeline into a semi-autonomous conversion workflow.

## What Changed

The original flow was:

```text
Upload image -> OCR -> Word document -> Download
```

The agentic flow is:

```text
Observe -> Interpret -> Decide -> Act -> Review -> Learn
```

The agent observes page properties, selects an OCR strategy, runs the existing
OCR and Word-generation tools, evaluates the output, and surfaces review items
before the user trusts the final document.

## Agent Architecture

```text
Input
  handwritten page images and user settings

Processing
  image observation, layout hints, OCR prompt selection

Decision
  strategy selection, review requirement, prompt construction

Action
  OCR transcription, document generation, batch combination

Feedback
  extracted text preview, warnings, audit log, human review

Memory
  opt-in local preferences only; raw notes are not stored by default
```

## Agent Type

The system uses a goal-based agent with light learning memory.

The goal is to produce a clean, structured Word document from handwritten notes.
The agent chooses actions based on that goal, while optional local memory stores
non-sensitive formatting preferences for future runs. It is intentionally
semi-autonomous because OCR errors can change meaning and handwritten notes may
contain private information.

## Intelligence Layer

- ML/Vision model: Qwen3-VL through Hugging Face Transformers.
- Rules: page observation, layout hints, review warnings, output policies.
- LLM-style prompting: dynamic OCR prompts for structured notes, exact
  transcription, tables, and chemistry-heavy pages.

For chemistry notes like `input/image 5.jpeg`, the agent uses a prompt that
preserves formulas, reaction arrows, compound labels, benzene diagrams,
temperatures, and reagents.

## Human-in-the-Loop Controls

The Streamlit UI includes:

- agentic workflow toggle
- structured vs exact transcription mode
- semi vs full autonomy setting
- chemistry notation preservation
- opt-in local memory
- "Forget Agent Memory" control
- agent review panel with observations, rationale, warnings, prompt, and audit log

## Ethical Design

- Privacy: raw uploaded notes are not stored by the agent memory.
- Transparency: the app shows the selected strategy, prompt, and audit log.
- User control: memory is opt-in and can be deleted.
- Bias awareness: uncertain OCR output is surfaced for review instead of hidden.
- Realistic autonomy: semi-autonomy is the recommended default.

## Usage

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Run the Streamlit app:

```bash
streamlit run app.py
```

If Streamlit shows a `torch.classes` file-watcher error on Windows, run:

```bash
streamlit run app.py --server.fileWatcherType none
```

3. Upload handwritten notes, review the agent output, and download the `.docx`.

## Supported Formats

- Input: JPEG, PNG, BMP, TIFF, WebP
- Output: Microsoft Word `.docx`
