
# AI Document Summarizer

An AI-powered desktop application built with Python and PyQt5 that automatically summarizes PDF documents and text using a locally running AI model.

## Overview

AI Document Summarizer helps users understand lengthy documents more efficiently by generating concise summaries, extracting important key points, and identifying relevant keywords.

The application uses a pre-trained DistilBART model from Hugging Face Transformers and performs AI inference locally.

## Features

- PDF Document Upload
- Text and Paragraph Summarization
- AI-Powered Text Summarization
- Key Points Extraction
- Keyword Extraction using TF-IDF
- Adjustable Summary Length (Short, Medium, Long)
- Export Summaries to PDF
- GPU Acceleration with CUDA Support
- User-Friendly PyQt5 Desktop Interface

## Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| PyQt5 | Desktop GUI |
| PyTorch | Deep learning framework |
| Hugging Face Transformers | AI summarization |
| DistilBART | Pre-trained summarization model |
| PyMuPDF | PDF text extraction |
| Scikit-learn | Keyword extraction |
| ReportLab | PDF generation |

## AI Model

Model: sshleifer/distilbart-cnn-12-6

The application uses a pre-trained DistilBART model for automatic text summarization.

The model is downloaded on first use and cached locally for subsequent execution.

## Installation

### 1. Clone the repository

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd AI_Document_Summarizer
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate it on Windows:

```powershell
.\venv\Scripts\activate
```

### 3. Install PyTorch

For CUDA 12.4:

```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```

For CPU-only installation, use the appropriate PyTorch CPU build instead.

### 4. Install other dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the application

```bash
python main.py
```

## How to Use

1. Launch the application.
2. Upload a PDF document or paste text.
3. Select the desired summary length.
4. Click Generate Summary.
5. View the summary, key points, and keywords.
6. Export the generated results as a PDF document.

## Hardware Requirements

- Python 3.10 or newer
- Minimum 8GB RAM recommended
- NVIDIA GPU with CUDA support recommended
- Internet connection for initial model download

CPU inference is also supported.

## Limitations

- Scanned PDFs require OCR, which is not currently implemented.
- Summary quality depends on the input document and AI model.
- Keyword extraction currently uses TF-IDF.
- Long documents are processed using text chunking.

## Future Improvements

- OCR support for scanned PDF documents
- DOCX document support
- AI-powered document question answering
- Document history and management
- Improved key point extraction
- Customizable PDF report templates

## Author

Developed as a personal AI and Python software development project.

## License

MIT License
