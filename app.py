import gradio as gr
import numpy as np
from PIL import Image
import cv2
import tempfile
import os
from pathlib import Path
from typing import Tuple, Optional
import io

from utils.layout_detector import LayoutDetector, pil_to_cv2
from utils.ocr_processor import OCRProcessor
from utils.word_generator import WordGenerator

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
        ocr_processor = OCRProcessor(gpu=True)
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
    Extract text from an image using OCR (Qwen2-VL).
    
    Args:
        image: PIL Image
        
    Returns:
        Tuple of (extracted text, metadata dict)
    """
    # Get processor
    processor = get_ocr_processor()
    
    # Run Qwen2-VL
    extracted_text = processor.process_image(image)
    
    metadata = {
        "model": "Qwen2-VL-2B-OCR",
        "length": len(extracted_text)
    }
    
    return extracted_text, metadata


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
        return None, "", "Please upload an image first."
    
    try:
        # Run OCR (Qwen2-VL)
        processor = get_ocr_processor()
        extracted_text = processor.process_image(image)
        
        # Generate Word document
        generator = WordGenerator()
        doc = generator.generate_from_qwen_output(extracted_text, title=filename)
        
        # Save to temporary file
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, f"{filename}.docx")
        generator.save(output_path)
        
        status = "Successfully converted using Qwen2-VL-2B-OCR!"
        
        return output_path, extracted_text, status
        
    except Exception as e:
        return None, "", f"Error during conversion: {str(e)}"


def batch_convert(images: list, combine_output: bool = False, base_filename: str = "document") -> Tuple[list, str]:
    """
    Convert multiple images to Word documents.
    
    Args:
        images: List of PIL Images
        combine_output: Whether to combine all outputs into one file
        base_filename: Base filename for output documents
        
    Returns:
        Tuple of (list of file paths, status message)
    """
    if not images:
        return [], "Please upload at least one image."
    
    output_files = []
    results = []
    
    for idx, image in enumerate(images):
        filename = f"{base_filename}_{idx + 1}"
        file_path, text, status = convert_to_word(image, filename)
        
        if file_path:
            output_files.append(file_path)
            results.append(f"{filename}.docx: {status}")
        else:
            results.append(f"{filename}: {status}")
            
    # Combine if requested
    if combine_output and output_files:
        try:
            generator = WordGenerator()
            # unique output path
            temp_dir = os.path.dirname(output_files[0])
            combined_path = os.path.join(temp_dir, f"{base_filename}_combined.docx")
            generator.combine_documents(output_files, combined_path)
            
            # Return only the combined file logic or append it?
            # Let's return just the combined file as the primary output if merged
            output_files = [combined_path]
            results.append(f"\n✅ Merged {len(images)} documents into {base_filename}_combined.docx")
        except Exception as e:
            results.append(f"\n❌ Merge failed: {str(e)}")
    
    status_text = "\n".join(results)
    return output_files, status_text


# Create the Gradio interface
def create_interface():
    """Create and return the Gradio interface."""
    
    with gr.Blocks(
        title="Handwriting-to-Word-Converter",
        theme=gr.themes.Soft(
            primary_hue="indigo",
            secondary_hue="blue",
            neutral_hue="slate",
            text_size=gr.themes.sizes.text_lg
        ).set(
            button_primary_background_fill="*primary_500",
            button_primary_background_fill_hover="*primary_600",
        ),
        css="""
        .main-title {
            text-align: center;
            margin-bottom: 2rem;
            color: #4F46E5;
        }
        .main-title h1 {
            font-size: 3rem;
            font-weight: 800;
        }
        .output-text {
            font-family: 'Courier New', monospace;
            background-color: #f8fafc;
            padding: 1.5rem;
            border-radius: 0.5rem;
            border: 1px solid #e2e8f0;
        }
        button.primary-btn {
            font-weight: bold;
            font-size: 1.1em;
        }
        """
    ) as demo:
        
        gr.Markdown(
            """
            # ✍️ Handwriting to Word Converter
            
            ### Convert handwritten notes to editable Word documents with Qwen2-VL AI
            
            This tool uses the state-of-the-art **Qwen2-VL-2B-OCR** model to transcribe handwriting and smart tables directly into Word format (DOCX).
            
            ---
            """,
            elem_classes=["main-title"]
        )
        
        with gr.Tabs():
            # Single Image Tab
            with gr.TabItem("📄 Single Image", id=1):
                with gr.Row():
                    with gr.Column(scale=1):
                        input_image = gr.Image(
                            label="Upload Handwritten Image",
                            type="pil",
                            height=450,
                            sources=["upload", "clipboard"]
                        )
                        
                        with gr.Row():
                            filename_input = gr.Textbox(
                                label="Output Filename",
                                value="converted_document",
                                placeholder="Enter filename (without extension)",
                                scale=2
                            )
                        
                        convert_btn = gr.Button(
                            "✨ Convert to Word",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                    
                    with gr.Column(scale=1):
                        status_output = gr.Textbox(
                            label="Status",
                            interactive=False
                        )
                        
                        output_file = gr.File(
                            label="📥 Download Word Document",
                            file_types=[".docx"]
                        )
                        
                        extracted_text = gr.Textbox(
                            label="📝 Extracted Text Preview",
                            lines=20,
                            interactive=False,
                            elem_classes=["output-text"]
                        )            
                
                # Event handlers for single image
                convert_btn.click(
                    fn=convert_to_word,
                    inputs=[input_image, filename_input],
                    outputs=[output_file, extracted_text, status_output]
                )
            
            # Batch Processing Tab
            with gr.TabItem("📚 Batch Processing", id=2):
                gr.Markdown(
                    """
                    ### Batch Convert Multiple Images
                    Upload multiple handwritten images to convert them all at once. Check **Combine Output** to merge them into a single Word file.
                    """
                )
                
                with gr.Row():
                    with gr.Column(scale=1):
                        batch_files = gr.File(
                            label="Upload Images",
                            file_count="multiple",
                            file_types=["image"],
                            height=300
                        )
                        
                        batch_filename = gr.Textbox(
                            label="Base Filename",
                            value="document",
                            placeholder="Base name for output files"
                        )
                        
                        combine_chk = gr.Checkbox(
                            label="🧩 Combine all into one single DOCX file",
                            value=False,
                            info="If checked, you will get one merged document with page breaks."
                        )
                        
                        batch_btn = gr.Button(
                            "🚀 Convert All",
                            variant="primary",
                            size="lg",
                            elem_classes=["primary-btn"]
                        )
                    
                    with gr.Column(scale=1):
                        batch_status = gr.Textbox(
                            label="Conversion Log",
                            lines=15,
                            interactive=False
                        )
                        
                        batch_output = gr.File(
                            label="📥 Download Converted Documents",
                            file_count="multiple"
                        )
                
                def process_batch_files(files, combine, base_name):
                    if not files:
                        return [], "⚠️ Please upload images first."
                    
                    images = []
                    for f in files:
                        try:
                            img = Image.open(f.name)
                            images.append(img)
                        except Exception as e:
                            pass
                    
                    return batch_convert(images, combine, base_name)
                
                batch_btn.click(
                    fn=process_batch_files,
                    inputs=[batch_files, combine_chk, batch_filename],
                    outputs=[batch_output, batch_status]
                )
            
        
        gr.Markdown(
            """
            ---
            <center>
            <p style="color: gray; font-size: 14px;">
            Handwriting to Word Converter | Powered by <b>Qwen2-VL</b> & <b>Gradio</b>
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
