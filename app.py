"""
Handwriting to Word Converter
A Gradio application that converts handwritten images to editable Word documents.
Supports multi-column layouts and various document structures.
"""

import gradio as gr
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
import io

# Import our utility modules
from utils.layout_detector import LayoutDetector, pil_to_cv2
from utils.ocr_processor import OCRProcessor
from utils.word_generator import WordGenerator


# Initialize processors (lazy loading)
layout_detector = None
ocr_processor = None
word_generator = None


def get_layout_detector():
    global layout_detector
    if layout_detector is None:
        layout_detector = LayoutDetector()
    return layout_detector


def get_ocr_processor():
    global ocr_processor
    if ocr_processor is None:
        ocr_processor = OCRProcessor(languages=['en'], gpu=True)
    return ocr_processor


def get_word_generator():
    global word_generator
    if word_generator is None:
        word_generator = WordGenerator()
    return word_generator


def process_image(image: Image.Image, show_columns: bool = False) -> Tuple[np.ndarray, dict]:
    """
    Process an image and detect its layout.
    
    Args:
        image: PIL Image
        show_columns: Whether to visualize detected columns
        
    Returns:
        Tuple of (visualization image, column info dict)
    """
    # Convert PIL to OpenCV format
    cv_image = pil_to_cv2(image)
    
    # Detect columns
    detector = get_layout_detector()
    columns = detector.detect_columns(cv_image)
    
    if show_columns:
        # Draw columns on image for visualization
        vis_image = cv_image.copy()
        colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0)]
        
        for idx, (x, y, w, h) in enumerate(columns):
            color = colors[idx % len(colors)]
            cv2.rectangle(vis_image, (x, y), (x + w, y + h), color, 3)
            cv2.putText(vis_image, f"Column {idx + 1}", (x + 10, y + 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
        
        # Convert back to RGB for display
        vis_image = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
        return vis_image, {"num_columns": len(columns), "columns": columns}
    
    return cv_image, {"num_columns": len(columns), "columns": columns}


def extract_text_from_image(image: Image.Image) -> Tuple[str, dict]:
    """
    Extract text from an image using OCR.
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (extracted text, metadata dict)
    """
    # Convert PIL to OpenCV format
    cv_image = pil_to_cv2(image)
    
    # Detect layout
    detector = get_layout_detector()
    columns = detector.detect_columns(cv_image)
    
    # Extract text from each column
    processor = get_ocr_processor()
    column_texts = processor.extract_text_columns(cv_image, columns)
    
    # Format text output
    all_text_parts = []
    total_confidence = 0
    total_items = 0
    
    for col_idx in sorted(column_texts.keys()):
        texts = column_texts[col_idx]
        formatted = processor.get_formatted_text(texts)
        all_text_parts.append(f"=== Column {col_idx + 1} ===\n{formatted}")
        
        # Calculate average confidence
        for t in texts:
            total_confidence += t['confidence']
            total_items += 1
    
    combined_text = '\n\n'.join(all_text_parts)
    avg_confidence = (total_confidence / total_items * 100) if total_items > 0 else 0
    
    metadata = {
        "num_columns": len(columns),
        "total_text_items": total_items,
        "average_confidence": f"{avg_confidence:.1f}%"
    }
    
    return combined_text, metadata


def convert_to_word(image: Image.Image, filename: str = "converted_document") -> Tuple[str, str, str]:
    """
    Convert handwritten image to Word document.
    
    Args:
        image: PIL Image
        filename: Base filename for the output document
        
    Returns:
        Tuple of (file path, extracted text, status message)
    """
    if image is None:
        return None, "", "❌ Please upload an image first."
    
    try:
        # Convert PIL to OpenCV format
        cv_image = pil_to_cv2(image)
        
        # Detect layout
        detector = get_layout_detector()
        columns = detector.detect_columns(cv_image)
        
        # Extract text
        processor = get_ocr_processor()
        column_texts = processor.extract_text_columns(cv_image, columns)
        
        # Generate formatted text for preview
        all_text_parts = []
        for col_idx in sorted(column_texts.keys()):
            texts = column_texts[col_idx]
            formatted = processor.get_formatted_text(texts)
            all_text_parts.append(f"=== Column {col_idx + 1} ===\n{formatted}")
        combined_text = '\n\n'.join(all_text_parts)
        
        # Generate Word document
        generator = WordGenerator()
        doc = generator.generate_from_column_texts(column_texts, title=filename)
        
        # Save to temporary file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{filename}.docx")
        generator.save(output_path)
        
        status = f"✅ Successfully converted! Detected {len(columns)} column(s)."
        
        return output_path, combined_text, status
        
    except Exception as e:
        return None, "", f"❌ Error during conversion: {str(e)}"


def batch_convert(images: list, base_filename: str = "document") -> Tuple[list, str]:
    """
    Convert multiple images to Word documents.
    
    Args:
        images: List of PIL Images
        base_filename: Base filename for output documents
        
    Returns:
        Tuple of (list of file paths, status message)
    """
    if not images:
        return [], "❌ Please upload at least one image."
    
    output_files = []
    results = []
    
    for idx, image in enumerate(images):
        filename = f"{base_filename}_{idx + 1}"
        file_path, text, status = convert_to_word(image, filename)
        
        if file_path:
            output_files.append(file_path)
            results.append(f"📄 {filename}.docx: {status}")
        else:
            results.append(f"📄 {filename}: {status}")
    
    status_text = "\n".join(results)
    return output_files, status_text


# Create the Gradio interface
def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(
        title="Handwriting to Word Converter",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="indigo",
        ),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 20px;
        }
        .output-text {
            font-family: 'Courier New', monospace;
            background-color: #f5f5f5;
            padding: 15px;
            border-radius: 8px;
        }
        .status-box {
            padding: 10px;
            border-radius: 8px;
            margin-top: 10px;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # Handwriting to Word Converter
            
            Convert your handwritten notes to editable Word documents with AI-powered OCR.
            Supports multi-column layouts and various document structures.
            
            ---
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # Single Image Tab
            with gr.TabItem("📝 Single Image", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload Handwritten Image",
                            type="pil",
                            height=400
                        )
                        
                        with gr.Row():
                            filename_input = gr.Textbox(
                                label="Output Filename",
                                value="converted_document",
                                placeholder="Enter filename (without extension)"
                            )
                        
                        with gr.Row():
                            convert_btn = gr.Button(
                                "🔄 Convert to Word",
                                variant="primary",
                                size="lg"
                            )
                            preview_btn = gr.Button(
                                "👁️ Preview Layout",
                                variant="secondary"
                            )
                    
                    with gr.Column(scale=1):
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False,
                            lines=2
                        )
                        
                        output_file = gr.File(
                            label="📥 Download Word Document",
                            file_types=[".docx"]
                        )
                        
                        extracted_text = gr.Textbox(
                            label="📋 Extracted Text Preview",
                            lines=15,
                            interactive=False,
                            elem_classes=["output-text"]
                        )
                
                # Layout preview section
                with gr.Accordion("🔍 Column Detection Preview", open=False):
                    layout_preview = gr.Image(
                        label="Detected Columns",
                        height=400
                    )
                    layout_info = gr.JSON(
                        label="Layout Information"
                    )
                
                # Event handlers for single image
                convert_btn.click(
                    fn=convert_to_word,
                    inputs=[input_image, filename_input],
                    outputs=[output_file, extracted_text, status_output]
                )
                
                preview_btn.click(
                    fn=lambda img: process_image(img, show_columns=True) if img else (None, {}),
                    inputs=[input_image],
                    outputs=[layout_preview, layout_info]
                )
            
            # Batch Processing Tab
            with gr.TabItem("📚 Batch Processing", id=2):
                gr.Markdown(
                    """
                    ### Batch Convert Multiple Images
                    Upload multiple handwritten images and convert them all to Word documents.
                    """
                )
                
                with gr.Row():
                    with gr.Column():
                        batch_images = gr.Gallery(
                            label="Upload Multiple Images",
                            height=300,
                            object_fit="contain"
                        )
                        
                        batch_files = gr.File(
                            label="Or upload files here",
                            file_count="multiple",
                            file_types=["image"]
                        )
                        
                        batch_filename = gr.Textbox(
                            label="Base Filename",
                            value="document",
                            placeholder="Base name for output files"
                        )
                        
                        batch_btn = gr.Button(
                            "🔄 Convert All",
                            variant="primary",
                            size="lg"
                        )
                    
                    with gr.Column():
                        batch_status = gr.Textbox(
                            label="Conversion Status",
                            lines=10,
                            interactive=False
                        )
                        
                        batch_output = gr.File(
                            label="📥 Download Converted Documents",
                            file_count="multiple"
                        )
                
                def process_batch_files(files, base_name):
                    if not files:
                        return [], "❌ Please upload images first."
                    
                    images = []
                    for f in files:
                        try:
                            img = Image.open(f.name)
                            images.append(img)
                        except Exception as e:
                            pass
                    
                    return batch_convert(images, base_name)
                
                batch_btn.click(
                    fn=process_batch_files,
                    inputs=[batch_files, batch_filename],
                    outputs=[batch_output, batch_status]
                )
            
            # # About Tab
            # with gr.TabItem("ℹ️ About", id=3):
            #     gr.Markdown(
            #         """
            #         ## About This Application
                    
            #         This application uses AI-powered Optical Character Recognition (OCR) 
            #         to convert handwritten documents into editable Word files.
                    
            #         ### Features
                    
            #         - **🔍 Smart Layout Detection**: Automatically detects multi-column layouts
            #         - **✍️ Handwriting Recognition**: Uses EasyOCR for accurate text extraction
            #         - **📄 Word Document Generation**: Creates properly formatted .docx files
            #         - **📚 Batch Processing**: Convert multiple images at once
                    
            #         ### Supported Formats
                    
            #         - **Input**: JPEG, PNG, BMP, TIFF, WebP
            #         - **Output**: Microsoft Word (.docx)
                    
            #         ### Tips for Best Results
                    
            #         1. Use clear, high-resolution images
            #         2. Ensure good lighting and contrast
            #         3. Avoid shadows and glare
            #         4. Keep the document flat and aligned
                    
            #         ### Technology Stack
                    
            #         - **OCR Engine**: EasyOCR
            #         - **Layout Detection**: OpenCV
            #         - **Document Generation**: python-docx
            #         - **Web Interface**: Gradio
                    
            #         ---
                    
            #         *Built with ❤️ for converting handwritten notes to digital format*
            #         """
            #     )
        
        gr.Markdown(
            """
            ---
            <center>
            <p style="color: gray; font-size: 12px;">
            Handwriting to Word Converter | Powered by EasyOCR & Gradio
            </p>
            </center>
            """
        )
    
    return demo


# Create and launch the app
demo = create_interface()

if __name__ == "__main__":
    demo.launch(
        share=True,  # Set to True for public link
        server_name="0.0.0.0",
        server_port=7861
    )
