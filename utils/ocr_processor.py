import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
from typing import List, Dict, Union
import importlib.util


class _TorchClassesPath:
    """Prevents Streamlit's source watcher from probing torch custom classes."""

    _path = []


try:
    torch.classes.__path__ = _TorchClassesPath()
except Exception:
    pass

class OCRProcessor:
    
    def __init__(self, model_id: str = "JackChew/Qwen2-VL-2B-OCR", gpu: bool = True):
        """
        Initialize the Qwen2-VL OCR processor.
        
        Args:
            model_id: Hugging Face model ID
            gpu: Whether to use GPU acceleration
        """
        self.model_id = model_id
        self.device = "cuda" if gpu and torch.cuda.is_available() else "cpu"
        self._processor = None
        self._model = None
        
    @property
    def pipeline(self):
        if self._model is None:
            print(f"Loading Qwen2-VL OCR model ({self.model_id})...")
            print("Loading Qwen2-VL OCR processor...")
            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                trust_remote_code=True,
            )
            model_kwargs = {
                "trust_remote_code": True,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
            }
            has_accelerate = importlib.util.find_spec("accelerate") is not None
            if has_accelerate:
                model_kwargs["device_map"] = self.device

            print(f"Loading Qwen2-VL OCR model weights on {self.device}...")
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                **model_kwargs,
            ).eval()
            if not has_accelerate:
                print(f"Moving Qwen2-VL OCR model to {self.device}...")
                self._model = self._model.to(self.device)
            print("Qwen2-VL OCR model is ready.")
        return self._processor, self._model

    def process_image(self, image: Union[Image.Image, np.ndarray], prompt: str = None) -> str:
        """
        Extract text and structure from an image using Qwen2-VL OCR.
        
        Args:
            image: PIL Image or numpy array
            prompt: Optional task prompt for the vision-language model
            
        Returns:
            Extracted text (Markdown/HTML format)
        """
        processor, model = self.pipeline
        
        # Ensure image is PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        task_prompt = prompt or "extract all data from these handwritten notes without miss anything"
        
        # Prepare conversation prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image,
                    },
                    {
                        "type": "text",
                        "text": task_prompt
                    }
                ]
            }
        ]
        
        # Preprocess
        inputs = processor.apply_chat_template(
            conversation,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        inputs.pop("token_type_ids", None)
        inputs = inputs.to(self.device)
        
        # Generate
        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=4096)
        
        # Decode
        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids, 
            skip_special_tokens=True, 
            clean_up_tokenization_spaces=True
        )[0]
        
        return output_text
