"""Agent orchestrator for handwritten notes to Word conversion."""

from __future__ import annotations

import os
import tempfile
from typing import Iterable, List

from PIL import Image

from utils.ocr_processor import OCRProcessor

from .evaluator import evaluate_output
from .memory import AgentMemory
from .planner import AgentPlanner
from .state import AgentConfig, AgentResult, BatchAgentResult
from .tools import AgentTools


class HandwritingAgent:
    """Goal-based, semi-autonomous conversion agent."""

    def __init__(self, ocr_processor: OCRProcessor, memory: AgentMemory | None = None):
        self.planner = AgentPlanner()
        self.tools = AgentTools(ocr_processor)
        self.memory = memory or AgentMemory()

    def process_image(
        self,
        image: Image.Image,
        *,
        filename: str = "converted.docx",
        config: AgentConfig | None = None,
    ) -> AgentResult:
        config = config or AgentConfig()
        observation = self.planner.observe(image, filename=filename)
        decision = self.planner.decide(observation, config)

        audit_log = [
            f"Observed {observation.width}x{observation.height} {observation.orientation} page.",
            f"Selected strategy: {decision.strategy}.",
            "Ran OCR with agent-selected prompt.",
        ]

        raw_text = self.tools.transcribe_image(image, decision.prompt)
        feedback_items = evaluate_output(raw_text, observation)
        document_bytes = self.tools.generate_docx(raw_text)

        if config.allow_memory:
            self.memory.save_preferences(
                {
                    "preferred_output_mode": config.output_mode,
                    "preserve_chemistry": config.preserve_chemistry,
                }
            )
            audit_log.append("Saved opt-in formatting preferences to local memory.")
        else:
            audit_log.append("Memory disabled; no preferences or note content were stored.")

        output_name = os.path.splitext(filename)[0] + ".docx"
        return AgentResult(
            raw_text=raw_text,
            document_bytes=document_bytes,
            filename=output_name,
            observation=observation,
            decision=decision,
            feedback_items=feedback_items,
            audit_log=audit_log,
            metadata={"autonomy_level": config.autonomy_level},
        )

    def process_batch(
        self,
        files: Iterable,
        *,
        combine_output: bool,
        config: AgentConfig | None = None,
    ) -> BatchAgentResult:
        config = config or AgentConfig()
        results: List[AgentResult] = []
        audit_log: List[str] = []
        feedback_items: List[str] = []

        with tempfile.TemporaryDirectory() as temp_dir:
            doc_paths: List[str] = []

            for index, file in enumerate(files, start=1):
                image = Image.open(file)
                result = self.process_image(image, filename=file.name, config=config)
                results.append(result)
                audit_log.extend([f"Page {index}: {entry}" for entry in result.audit_log])
                feedback_items.extend([f"{file.name}: {item}" for item in result.feedback_items])

                doc_path = os.path.join(temp_dir, result.filename)
                with open(doc_path, "wb") as f:
                    f.write(result.document_bytes)
                doc_paths.append(doc_path)

            if combine_output:
                output_path = os.path.join(temp_dir, "agent_combined_output.docx")
                self.tools.combine_documents(doc_paths, output_path)
                with open(output_path, "rb") as f:
                    output_bytes = f.read()
                output_name = "agent_combined_output.docx"
                output_type = "single"
            else:
                import io
                import zipfile

                zip_buffer = io.BytesIO()
                with zipfile.ZipFile(zip_buffer, "w") as zf:
                    for doc_path in doc_paths:
                        zf.write(doc_path, os.path.basename(doc_path))
                zip_buffer.seek(0)
                output_bytes = zip_buffer.getvalue()
                output_name = "agent_batch_output.zip"
                output_type = "zip"

        return BatchAgentResult(
            results=results,
            output_bytes=output_bytes,
            filename=output_name,
            output_type=output_type,
            audit_log=audit_log,
            feedback_items=feedback_items,
            metadata={"pages": len(results), "combine_output": combine_output},
        )
