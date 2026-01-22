"""
Layout Detector Module
Detects columns and dividers in handwritten document images using OpenCV.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional


class LayoutDetector:
    """Detects layout structure (columns, dividers) in document images."""
    
    def __init__(self, min_line_length: int = 100, line_threshold: int = 50):
        """
        Initialize the layout detector.
        
        Args:
            min_line_length: Minimum length for a line to be considered a divider
            line_threshold: Threshold for line detection sensitivity
        """
        self.min_line_length = min_line_length
        self.line_threshold = line_threshold
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for better layout detection.
        
        Args:
            image: Input image as numpy array (BGR or RGB)
            
        Returns:
            Preprocessed grayscale image
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply adaptive thresholding for better binarization
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        return binary
    
    def detect_lines(self, image: np.ndarray) -> Tuple[List[Tuple], List[Tuple]]:
        """
        Detect horizontal and vertical lines in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple of (vertical_lines, horizontal_lines)
        """
        binary = self.preprocess_image(image)
        height, width = binary.shape[:2]
        
        # Detect vertical lines
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, height // 10))
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
        vertical_lines = cv2.HoughLinesP(
            vertical, 1, np.pi/180, self.line_threshold,
            minLineLength=self.min_line_length, maxLineGap=20
        )
        
        # Detect horizontal lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width // 10, 1))
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_lines = cv2.HoughLinesP(
            horizontal, 1, np.pi/180, self.line_threshold,
            minLineLength=self.min_line_length, maxLineGap=20
        )
        
        v_lines = vertical_lines.tolist() if vertical_lines is not None else []
        h_lines = horizontal_lines.tolist() if horizontal_lines is not None else []
        
        return v_lines, h_lines
    
    def detect_columns(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect column regions in the image.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of column bounding boxes as (x, y, width, height)
        """
        height, width = image.shape[:2]
        v_lines, h_lines = self.detect_lines(image)
        
        # Find significant vertical dividers (lines that span most of the height)
        significant_v_lines = []
        for line in v_lines:
            x1, y1, x2, y2 = line[0]
            line_height = abs(y2 - y1)
            if line_height > height * 0.3:  # Line spans at least 30% of image height
                x_pos = (x1 + x2) // 2
                significant_v_lines.append(x_pos)
        
        # Remove duplicate/close lines
        significant_v_lines = sorted(set(significant_v_lines))
        filtered_v_lines = []
        for x in significant_v_lines:
            if not filtered_v_lines or x - filtered_v_lines[-1] > width * 0.1:
                filtered_v_lines.append(x)
        
        # Create column regions
        columns = []
        if filtered_v_lines:
            # Add left margin
            prev_x = 0
            for x in filtered_v_lines:
                if x - prev_x > width * 0.1:  # Minimum column width
                    columns.append((prev_x, 0, x - prev_x, height))
                prev_x = x
            # Add right column
            if width - prev_x > width * 0.1:
                columns.append((prev_x, 0, width - prev_x, height))
        else:
            # No clear column dividers - try to detect based on content density
            columns = self._detect_columns_by_density(image)
        
        # If still no columns detected, assume single column or split in middle
        if not columns:
            # Check if image is wide enough for 2 columns
            if width > height * 0.8:  # Landscape-ish or square
                mid = width // 2
                columns = [
                    (0, 0, mid, height),
                    (mid, 0, width - mid, height)
                ]
            else:
                columns = [(0, 0, width, height)]
        
        return columns
    
    def _detect_columns_by_density(self, image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Detect columns based on vertical whitespace/content density.
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of column bounding boxes
        """
        binary = self.preprocess_image(image)
        height, width = binary.shape
        
        # Calculate vertical projection (sum of pixels in each column)
        v_projection = np.sum(binary, axis=0)
        
        # Smooth the projection
        kernel_size = width // 50
        if kernel_size % 2 == 0:
            kernel_size += 1
        if kernel_size < 3:
            kernel_size = 3
        smoothed = np.convolve(v_projection, np.ones(kernel_size)/kernel_size, mode='same')
        
        # Find valleys (potential column separators)
        threshold = np.mean(smoothed) * 0.3
        valleys = []
        in_valley = False
        valley_start = 0
        
        for i, val in enumerate(smoothed):
            if val < threshold and not in_valley:
                in_valley = True
                valley_start = i
            elif val >= threshold and in_valley:
                in_valley = False
                valley_center = (valley_start + i) // 2
                valley_width = i - valley_start
                if valley_width > width * 0.02:  # Minimum valley width
                    valleys.append(valley_center)
        
        # Filter valleys to find the most significant one near the center
        center = width // 2
        significant_valleys = [v for v in valleys if abs(v - center) < width * 0.2]
        
        if significant_valleys:
            divider = min(significant_valleys, key=lambda x: abs(x - center))
            return [
                (0, 0, divider, height),
                (divider, 0, width - divider, height)
            ]
        
        return []
    
    def get_reading_order_regions(self, image: np.ndarray) -> List[dict]:
        """
        Get regions in reading order (top-to-bottom, left-to-right within columns).
        
        Args:
            image: Input image as numpy array
            
        Returns:
            List of region dictionaries with 'bbox' and 'column' keys
        """
        columns = self.detect_columns(image)
        regions = []
        
        for col_idx, (x, y, w, h) in enumerate(columns):
            regions.append({
                'bbox': (x, y, w, h),
                'column': col_idx,
                'order': col_idx  # For now, simple left-to-right ordering
            })
        
        return sorted(regions, key=lambda r: r['order'])


def load_image(image_path: str) -> np.ndarray:
    """
    Load an image from file path.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Image as numpy array (BGR format)
    """
    image = cv2.imread(image_path)
    if image is None:
        # Try with PIL for broader format support
        pil_image = Image.open(image_path)
        image = np.array(pil_image)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def pil_to_cv2(pil_image: Image.Image) -> np.ndarray:
    """Convert PIL Image to OpenCV format."""
    numpy_image = np.array(pil_image)
    if len(numpy_image.shape) == 3 and numpy_image.shape[2] == 3:
        return cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return numpy_image
