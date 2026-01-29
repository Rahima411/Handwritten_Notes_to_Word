import streamlit as st
import os
import tempfile
from PIL import Image
from utils.ocr_processor import OCRProcessor
from utils.word_generator import WordGenerator
import zipfile
import io
import shutil

# Page Config
st.set_page_config(
    page_title="Handwriting to Word Converter",
    page_icon="📝",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .reportview-container {
        background: #f0f2f6;
    }
    .main-header {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
        font-size: 2.5rem;
        color: #4B0082;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .stButton>button {
        background-color: #4B0082;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #6931a1;
        border-color: #6931a1;
        color: white;
    }
    .stTextInput>div>div>input {
        border-radius: 8px;
    }
    .css-1aumxhk {
        padding: 2rem;
        border-radius: 10px;
        background-color: white;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">📝 Handwriting to Word Converter</h1>', unsafe_allow_html=True)

# Initialize Processors with Caching
@st.cache_resource
def get_ocr_processor():
    """
    Lazy load the OCR processor. This will execute only once.
    """
    return OCRProcessor()

try:
    ocr_processor = get_ocr_processor()
except Exception as e:
    st.error(f"Failed to load OCR Model: {e}")
    st.stop()

# Helper function to clear session state
def clear_state():
    if 'single_result_doc' in st.session_state:
        del st.session_state['single_result_doc']
    if 'single_result_text' in st.session_state:
        del st.session_state['single_result_text']
    if 'batch_results' in st.session_state:
        del st.session_state['batch_results']

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/000000/document.png", width=150)
    st.title("Settings & Info")
    st.info(
        """
        **Model:** Qwen2-VL-2B-OCR
        **Capabilities:**
        - Handwriting Recognition
        - Table Structure Extraction
        - Layout Preservation
        """
    )
    if st.button("Clear History"):
        clear_state()
        st.rerun()

# Tabs
tab1, tab2 = st.tabs(["📄 Single Image Conversion", "📚 Batch Analysis"])

# --- Tab 1: Single Image ---
with tab1:
    st.markdown("### Convert a Single Handwritten Page")
    uploaded_file = st.file_uploader("Upload an Image", type=["png", "jpg", "jpeg", "bmp", "tiff"], key="single_upload")

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
            
        with col2:
            if st.button("Convert to Word", key="convert_single"):
                with st.spinner("Processing... The model is analyzing your image..."):
                    try:
                        # Process Image
                        raw_text = ocr_processor.process_image(image)
                        
                        # Generate Word Doc
                        # Instantiate new WordGenerator for each conversion
                        wg = WordGenerator() 
                        doc = wg.generate_from_qwen_output(raw_text)
                        
                        # Save to bytes for download
                        doc_bytes = wg.save_to_bytes()
                        
                        # Store in session state
                        st.session_state['single_result_doc'] = doc_bytes
                        st.session_state['single_result_text'] = raw_text
                        st.session_state['single_filename'] = os.path.splitext(uploaded_file.name)[0] + ".docx"
                        
                    except Exception as e:
                        st.error(f"An error occurred: {str(e)}")

    # Display Results if available
    if 'single_result_text' in st.session_state and uploaded_file is not None:
        st.divider()
        st.success("Conversion Successful! 🎉")
        
        col_res1, col_res2 = st.columns(2)
        
        with col_res1:
            st.subheader("Extracted Text Preview")
            st.markdown("The underlying structured text extracted from the image:")
            st.text_area("Extracted Text", value=st.session_state['single_result_text'], height=400, label_visibility="collapsed")
            
        with col_res2:
            st.subheader("Download Result")
            st.markdown("Download the fully formatted Word document:")
            
            st.download_button(
                label="📥 Download Word Document",
                data=st.session_state['single_result_doc'],
                file_name=st.session_state.get('single_filename', 'converted.docx'),
                mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document"
            )

# --- Tab 2: Batch Processing ---
with tab2:
    st.markdown("### Batch Process Multiple Images")
    uploaded_files = st.file_uploader("Upload Multiple Images", type=["png", "jpg", "jpeg", "bmp", "tiff"], accept_multiple_files=True, key="batch_upload")
    combine_output = st.checkbox("Combine all outputs into a single Word document?", value=True)
    
    if uploaded_files:
        if st.button("Start Batch Conversion", key="convert_batch"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            converted_paths = []
            temp_dir = tempfile.mkdtemp()
            
            try:
                total_files = len(uploaded_files)
                for i, file in enumerate(uploaded_files):
                    status_text.text(f"Processing {i+1}/{total_files}: {file.name}...")
                    
                    image = Image.open(file)
                    raw_text = ocr_processor.process_image(image)
                    
                    wg = WordGenerator()
                    wg.generate_from_qwen_output(raw_text)
                    
                    # Save individual doc to temp file
                    temp_doc_path = os.path.join(temp_dir, f"{os.path.splitext(file.name)[0]}.docx")
                    wg.save(temp_doc_path)
                    converted_paths.append(temp_doc_path)
                    
                    progress_bar.progress((i + 1) / total_files)
                
                status_text.text("Processing complete! Preparing download...")
                
                # Handle Output
                final_files = {} # filename: bytes
                
                if combine_output and len(converted_paths) > 0:
                    combined_path = os.path.join(temp_dir, "combined_batch_output.docx")
                    WordGenerator().combine_documents(converted_paths, combined_path)
                    
                    with open(combined_path, "rb") as f:
                        final_files["combined_batch_output.docx"] = f.read()
                        
                    st.session_state['batch_results'] = {
                        "type": "single",
                        "data": final_files["combined_batch_output.docx"],
                        "name": "combined_batch_output.docx"
                    }
                    
                elif len(converted_paths) > 0:
                    # Zip individual files
                    zip_buffer = io.BytesIO()
                    with zipfile.ZipFile(zip_buffer, "w") as zf:
                        for doc_path in converted_paths:
                            zf.write(doc_path, os.path.basename(doc_path))
                    zip_buffer.seek(0)
                    
                    st.session_state['batch_results'] = {
                        "type": "zip",
                        "data": zip_buffer.getvalue(),
                        "name": "batch_output.zip"
                    }
                
                st.success("Batch Processing Finished! 🎉")
                
            except Exception as e:
                st.error(f"An error occurred during batch processing: {str(e)}")
            finally:
                # Cleanup
                try:
                    shutil.rmtree(temp_dir)
                except:
                    pass

    # Display Batch Results
    if 'batch_results' in st.session_state and uploaded_files:
        st.divider()
        res = st.session_state['batch_results']
        mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document" if res["type"] == "single" else "application/zip"
        
        st.download_button(
            label=f"📥 Download {res['name']}",
            data=res['data'],
            file_name=res['name'],
            mime=mime_type
        )
