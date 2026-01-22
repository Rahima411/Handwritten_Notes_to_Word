# Deployment Guide - Hugging Face Spaces

This guide walks you through deploying the Handwriting to Word Converter to Hugging Face Spaces.

## Prerequisites

- A free [Hugging Face](https://huggingface.co/) account

## Deployment Steps

### Option 1: Deploy via Hugging Face Web Interface (Easiest)

1. **Go to Hugging Face Spaces**: https://huggingface.co/spaces

2. **Create a new Space**:
   - Click "Create new Space"
   - Enter a name (e.g., `handwriting-to-word`)
   - Select **Gradio** as the SDK
   - Choose **Public** for visibility (required for free tier)
   - Click "Create Space"

3. **Upload Files**: Use the "Files" tab to upload these files:
   ```
   app.py
   requirements.txt
   README.md
   utils/__init__.py
   utils/layout_detector.py
   utils/ocr_processor.py
   utils/word_generator.py
   ```

4. **Wait for Build**: The Space will automatically build and deploy (usually 2-5 minutes)

5. **Get Your Link**: Your app will be available at:
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/handwriting-to-word
   ```

### Option 2: Deploy via Git

1. **Install Git LFS** (if not already):
   ```bash
   git lfs install
   ```

2. **Clone your new Space**:
   ```bash
   git clone https://huggingface.co/spaces/YOUR_USERNAME/handwriting-to-word
   cd handwriting-to-word
   ```

3. **Copy project files**:
   ```bash
   # Copy all files from the PPIT folder
   cp -r /path/to/PPIT/* .
   ```

4. **Push to Hugging Face**:
   ```bash
   git add .
   git commit -m "Initial deployment"
   git push
   ```

## File Structure for Deployment

Make sure your Space contains:

```
├── app.py                    # Main Gradio application
├── requirements.txt          # Python dependencies
├── README.md                 # Space configuration (with frontmatter)
└── utils/
    ├── __init__.py
    ├── layout_detector.py
    ├── ocr_processor.py
    └── word_generator.py
```

## Troubleshooting

### Build Fails
- Check the "Logs" tab in your Space for error messages
- Ensure all required packages are in `requirements.txt`

### Application Crashes
- The first load may take 1-2 minutes as EasyOCR downloads models
- Check memory limits (free tier has 16GB RAM)

### OCR Quality Issues
- Ensure images are clear and well-lit
- Higher resolution images generally work better

## Share Your App

Once deployed, share your app link:
```
https://huggingface.co/spaces/YOUR_USERNAME/handwriting-to-word
```

Anyone with this link can use your handwriting converter!
