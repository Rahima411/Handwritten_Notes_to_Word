"""
OCR Processor Module
Performs OCR on handwritten document images using a hybrid approach:
- EasyOCR for text detection (finding bounding boxes)
- TrOCR (Transformer-based OCR) for text recognition (reading handwriting)
"""

import easyocr
import numpy as np
from PIL import Image
import cv2
import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from typing import List, Tuple, Optional, Dict

class OCRProcessor:
    """Processes images using EasyOCR for detection and TrOCR for recognition."""
    
    def __init__(self, languages: List[str] = ['en'], gpu: bool = False):
        """
        Initialize the OCR processor.
        
        Args:
            languages: List of language codes (kept for compatibility, though TrOCR is mostly EN)
            gpu: Whether to use GPU acceleration
        """
        self.languages = languages
        self.gpu = gpu
        self.device = 'cuda' if gpu and torch.cuda.is_available() else 'cpu'
        
        self._reader = None
        self._trocr_processor = None
        self._trocr_model = None

    @property
    def reader(self):
        """Lazy initialization of EasyOCR reader (used for detection)."""
        if self._reader is None:
            # We only need it for detection, but EasyOCR API bundles them.
            # Initializing with basic English is fine.
            self._reader = easyocr.Reader(['en'], gpu=self.gpu)
        return self._reader

    @property
    def trocr_pipeline(self):
        """Lazy initialization of TrOCR model and processor."""
        if self._trocr_model is None:
            # Switch to LARGE model for better accuracy
            model_name = 'microsoft/trocr-large-handwritten' 
            print(f"Loading TrOCR model ({model_name})... this may take a moment.")
            self._trocr_processor = TrOCRProcessor.from_pretrained(model_name)
            self._trocr_model = VisionEncoderDecoderModel.from_pretrained(model_name).to(self.device)
        return self._trocr_processor, self._trocr_model

    def preprocess_for_ocr(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better detection results.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Preprocessed image
        """
        # TrOCR is robust, but clean images help detection.
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Simple denoising often helps
        denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        
        # CLAHE for contrast - slightly stronger clipLimit for detection
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)
        
        return enhanced

    def recognize_text_trocr(self, image_crop: np.ndarray) -> str:
        """
        Recognize text from a cropped image using TrOCR.
        """
        processor, model = self.trocr_pipeline
        
        try:
            # Convert CV2 numpy array (BGR/GRAY) to PIL Image (RGB)
            if len(image_crop.shape) == 2:
                pil_image = Image.fromarray(image_crop).convert("RGB")
            else:
                pil_image = Image.fromarray(cv2.cvtColor(image_crop, cv2.COLOR_BGR2RGB))

            pixel_values = processor(images=pil_image, return_tensors="pt").pixel_values.to(self.device)
            # Use beam search for better quality
            generated_ids = model.generate(pixel_values, num_beams=4, early_stopping=True)
            generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
            
            return generated_text
        except Exception as e:
            print(f"TrOCR Error: {e}")
            return ""

    def extract_text(self, image: np.ndarray, preprocess: bool = True) -> List[Dict]:
        """
        Extract text from an image.
        
        Args:
            image: Input image as numpy array
            preprocess: Whether to preprocess the image (for detection mostly)
            
        Returns:
            List of dictionaries with 'text', 'bbox', and 'confidence' keys
        """
        # Step 1: Detect Text Boxes using EasyOCR
        
        detection_image = image
        if preprocess:
            # Use preprocessed image for detection to find faint lines better
            detection_image = self.preprocess_for_ocr(image)

        # Tuned parameters for handwriting detection:
        # link_threshold: higher val merges words into lines better (def: 0.4)
        # low_text: lower val finds fainter text (def: 0.4)
        # width_ths: merges horizontally (def: 0.5)
        # add_margin: expands box slightly (def: 0.1)
        results = self.reader.readtext(
            detection_image,
            link_threshold=0.6,
            low_text=0.3,
            width_ths=0.7,
            add_margin=0.15
        )
        
        extracted = []
        for bbox, easy_text, easy_conf in results:
            # bbox is list of 4 points: [[x,y], [x,y], [x,y], [x,y]]
            # Crop the original image (not preprocessed) for TrOCR to preserve details
            
            # Get coords
            top_left = bbox[0]
            bottom_right = bbox[2]
            
            x_min = max(0, int(top_left[0]))
            y_min = max(0, int(top_left[1]))
            x_max = min(image.shape[1], int(bottom_right[0]))
            y_max = min(image.shape[0], int(bottom_right[1]))
            
            # Safety check
            if x_max <= x_min or y_max <= y_min:
                continue
                
            crop = image[y_min:y_max, x_min:x_max]
            
            # Run TrOCR on the crop
            trocr_text = self.recognize_text_trocr(crop)
            
            # Fallback to EasyOCR text if TrOCR yields nothing (rare)
            final_text = trocr_text if trocr_text.strip() else easy_text
            
            # Standardize bbox (x, y, w, h)
            w = x_max - x_min
            h = y_max - y_min
            
            extracted.append({
                'text': final_text,
                'bbox': (x_min, y_min, w, h),
                'confidence': 1.0, # TrOCR doesn't give simple confidence score easily
                'original_bbox': bbox
            })
        
        return extracted
    
    def extract_text_from_region(self, image: np.ndarray, 
                                  region: Tuple[int, int, int, int]) -> List[Dict]:
        """
        Extract text from a specific region of the image.
        """
        x, y, w, h = region
        
        # Ensure region is within image bounds
        img_h, img_w = image.shape[:2]
        x = max(0, x)
        y = max(0, y)
        w = min(w, img_w - x)
        h = min(h, img_h - y)
        
        # Extract region
        roi = image[y:y+h, x:x+w]
        
        # Run OCR on region
        results = self.extract_text(roi)
        
        # Adjust coordinates to original image space
        for result in results:
            bx, by, bw, bh = result['bbox']
            result['bbox'] = (bx + x, by + y, bw, bh)
            result['region_offset'] = (x, y)
        
        return results
    
    def extract_text_columns(self, image: np.ndarray, 
                              columns: List[Tuple[int, int, int, int]]) -> Dict[int, List[Dict]]:
        """
        Extract text from multiple columns.
        """
        column_texts = {}
        
        for col_idx, column in enumerate(columns):
            texts = self.extract_text_from_region(image, column)
            # Sort by vertical position within column
            texts.sort(key=lambda t: t['bbox'][1])
            column_texts[col_idx] = texts
        
        return column_texts
    
    def group_into_lines(self, texts: List[Dict], 
                          line_threshold: int = 20) -> List[List[Dict]]:
        """
        Group text elements into lines based on vertical position.
        """
        if not texts:
            return []
        
        # Sort by vertical position
        sorted_texts = sorted(texts, key=lambda t: t['bbox'][1])
        
        lines = []
        current_line = [sorted_texts[0]]
        current_y = sorted_texts[0]['bbox'][1]
        
        for text in sorted_texts[1:]:
            y = text['bbox'][1]
            if abs(y - current_y) <= line_threshold:
                current_line.append(text)
            else:
                # Sort current line by horizontal position
                current_line.sort(key=lambda t: t['bbox'][0])
                lines.append(current_line)
                current_line = [text]
                current_y = y
        
        # Don't forget the last line
        if current_line:
            current_line.sort(key=lambda t: t['bbox'][0])
            lines.append(current_line)
        
        return lines
    
    def get_formatted_text(self, texts: List[Dict]) -> str:
        """
        Convert text dictionaries to formatted string.
        """
        lines = self.group_into_lines(texts)
        formatted_lines = []
        
        for line in lines:
            line_text = ' '.join([t['text'] for t in line])
            formatted_lines.append(line_text)
        
        return '\n'.join(formatted_lines)


def process_image_file(image_path: str) -> str:
    """
    Convenience function to process an image file and return text.
    """
    from .layout_detector import load_image, LayoutDetector
    
    image = load_image(image_path)
    detector = LayoutDetector()
    processor = OCRProcessor()
    
    columns = detector.detect_columns(image)
    column_texts = processor.extract_text_columns(image, columns)
    
    all_text = []
    for col_idx in sorted(column_texts.keys()):
        texts = column_texts[col_idx]
        formatted = processor.get_formatted_text(texts)
        all_text.append(formatted)
    
    return '\n\n---\n\n'.join(all_text)
