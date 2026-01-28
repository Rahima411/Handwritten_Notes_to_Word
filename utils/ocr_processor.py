import torch
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image
import numpy as np
from typing import List, Dict, Union

class OCRProcessor:
    
    def __init__(self, model_id: str = "JackChew/Qwen2-VL-2B-OCR", gpu: bool = True):
        """
        Initialize the Qwen2-VL processor.
        
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
            print(f"Loading Qwen2-VL model ({self.model_id})...")
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                device_map=self.device, 
                trust_remote_code=True
            ).eval()
        return self._processor, self._model

    def process_image(self, image: Union[Image.Image, np.ndarray]) -> str:
        """
        Extract text and structure from an image using Qwen2-VL.
        
        Args:
            image: PIL Image or numpy array
            
        Returns:
            Extracted text (Markdown/HTML format)
        """
        processor, model = self.pipeline
        
        # Ensure image is PIL
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        # Prepare conversation prompt
        conversation = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                    },
                    {
                        "type": "text",
                        "text": "extract all data from these handwritten notes without miss anything"
                    }
                ]
            }
        ]
        
        # Preprocess
        text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
        inputs = processor(
            text=[text_prompt], 
            images=[image], 
            padding=True, 
            return_tensors="pt"
        )
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
