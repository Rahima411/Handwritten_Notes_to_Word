"""Tool wrappers used by the agent."""

from typing import Iterable, List

from PIL import Image

from utils.ocr_processor import OCRProcessor
from utils.word_generator import WordGenerator


class AgentTools:
    """Adapters around existing project utilities."""

    def __init__(self, ocr_processor: OCRProcessor):
        self.ocr_processor = ocr_processor

    def transcribe_image(self, image: Image.Image, prompt: str) -> str:
        return self.ocr_processor.process_image(image, prompt=prompt)

    def generate_docx(self, text: str, title: str | None = None) -> bytes:
        generator = WordGenerator()
        generator.generate_from_qwen_output(text, title=title)
        return generator.save_to_bytes()

    def save_docx(self, text: str, path: str, title: str | None = None) -> str:
        generator = WordGenerator()
        generator.generate_from_qwen_output(text, title=title)
        return generator.save(path)

    def combine_documents(self, doc_paths: Iterable[str], output_path: str) -> str:
        return WordGenerator().combine_documents(list(doc_paths), output_path)
