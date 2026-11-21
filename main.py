
import sys
import os
import re
import torch
import fitz

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QPushButton,
    QTextEdit, QLabel, QFileDialog,
    QComboBox, QMessageBox, QTabWidget,
    QProgressBar
)

from PyQt5.QtCore import QThread, pyqtSignal
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.feature_extraction.text import TfidfVectorizer

from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer
)
from reportlab.lib.units import inch
from xml.sax.saxutils import escape


# ==========================================
# CONFIGURATION
# ==========================================

MODEL_NAME = "sshleifer/distilbart-cnn-12-6"

MODEL = None
TOKENIZER = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==========================================
# LOAD AI MODEL
# ==========================================

def load_model():

    global MODEL, TOKENIZER

    if MODEL is None:

        print("Loading AI model...")

        TOKENIZER = AutoTokenizer.from_pretrained(
            MODEL_NAME
        )

        MODEL = AutoModelForSeq2SeqLM.from_pretrained(
            MODEL_NAME
        )

        MODEL.to(DEVICE)
        MODEL.eval()

        print("AI model loaded successfully!")

    return MODEL, TOKENIZER


# ==========================================
# PDF TEXT EXTRACTION
# ==========================================

def extract_pdf_text(file_path):

    document = fitz.open(file_path)

    pages = []

    for page in document:

        text = page.get_text("text")

        if text.strip():
            pages.append(text)

    document.close()

    return "\n".join(pages)


# ==========================================
# TEXT PREPROCESSING
# ==========================================

def clean_text(text):

    text = re.sub(r"\s+", " ", text)

    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    return text.strip()


# ==========================================
# SPLIT LONG DOCUMENTS
# ==========================================

def split_into_chunks(text, tokenizer):

    token_ids = tokenizer.encode(
        text,
        add_special_tokens=False
    )

    max_tokens = 850
    overlap = 60

    chunks = []

    start = 0

    while start < len(token_ids):

        end = min(start + max_tokens, len(token_ids))

        chunk_ids = token_ids[start:end]

        chunk = tokenizer.decode(
            chunk_ids,
            skip_special_tokens=True
        )

        if chunk.strip():
            chunks.append(chunk)

        if end == len(token_ids):
            break

        start = end - overlap

    return chunks


# ==========================================
# GENERATE SUMMARY
# ==========================================

def generate_summary(text, length):

    model, tokenizer = load_model()

    settings = {

        "Short": {
            "max_length": 90,
            "min_length": 25
        },

        "Medium": {
            "max_length": 150,
            "min_length": 50
        },

        "Long": {
            "max_length": 220,
            "min_length": 90
        }
    }

    config = settings[length]

    chunks = split_into_chunks(text, tokenizer)

    summaries = []

    for chunk in chunks:

        inputs = tokenizer(
            chunk,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        with torch.inference_mode():

            output = model.generate(
                **inputs,
                max_length=config["max_length"],
                min_length=config["min_length"],
                num_beams=4,
                do_sample=False,
                early_stopping=True
            )

        summary = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        summaries.append(summary)

    # Combine individual chunk summaries
    combined_summary = " ".join(summaries)

    # If the combined summary is still too long,
    # summarize it again in smaller chunks.
    if len(chunks) > 1:

        final_chunks = split_into_chunks(
            combined_summary,
            tokenizer
        )

        if len(final_chunks) < len(chunks):

            final_summaries = []

            for chunk in final_chunks:

                inputs = tokenizer(
                    chunk,
                    return_tensors="pt",
                    truncation=True,
                    max_length=1024
                )

                inputs = {
                    key: value.to(DEVICE)
                    for key, value in inputs.items()
                }

                with torch.inference_mode():

                    output = model.generate(
                        **inputs,
                        max_length=config["max_length"],
                        min_length=config["min_length"],
                        num_beams=4,
                        do_sample=False
                    )

                final_summaries.append(
                    tokenizer.decode(
                        output[0],
                        skip_special_tokens=True
                    )
                )

            combined_summary = " ".join(final_summaries)

    return combined_summary


# ==========================================
# KEY POINT EXTRACTION
# ==========================================

def extract_key_points(summary):

    sentences = re.split(
        r"(?<=[.!?])\s+",
        summary
    )

    points = []

    for sentence in sentences:

        sentence = sentence.strip()

        if len(sentence) > 15:
            points.append(sentence)

    return points[:10]


# ==========================================
# KEYWORD EXTRACTION
# ==========================================

def extract_keywords(text, count=10):

    try:

        vectorizer = TfidfVectorizer(
            stop_words="english",
            ngram_range=(1, 2),
            max_features=1000
        )

        matrix = vectorizer.fit_transform([text])

        scores = matrix.toarray()[0]

        words = vectorizer.get_feature_names_out()

        ranked = sorted(
            zip(words, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [
            word for word, score in ranked[:count]
            if score > 0
        ]

    except ValueError:

        return []


# ==========================================
# BACKGROUND AI WORKER
# ==========================================

class SummaryWorker(QThread):

    finished = pyqtSignal(str, list, list)
    error = pyqtSignal(str)

    def __init__(self, text, length):

        super().__init__()

        self.text = text
        self.length = length

    def run(self):

        try:

            cleaned = clean_text(self.text)

            if not cleaned:
                raise ValueError("No text was provided.")

            summary = generate_summary(
                cleaned,
                self.length
            )

            points = extract_key_points(summary)

            keywords = extract_keywords(cleaned)

            self.finished.emit(
                summary,
                points,
                keywords
            )

        except Exception as e:

            self.error.emit(str(e))


# ==========================================
# EXPORT SUMMARY TO PDF
# ==========================================

def export_to_pdf(summary, points, keywords):

    file_path, _ = QFileDialog.getSaveFileName(
        None,
        "Save Summary",
        "AI_Summary.pdf",
        "PDF Files (*.pdf)"
    )

    if not file_path:
        return

    if not file_path.lower().endswith(".pdf"):
        file_path += ".pdf"

    try:

        document = SimpleDocTemplate(
            file_path,
            pagesize=A4,
            rightMargin=50,
            leftMargin=50,
            topMargin=50,
            bottomMargin=50
        )

        styles = getSampleStyleSheet()

        content = []

        content.append(
            Paragraph(
                "AI DOCUMENT SUMMARY",
                styles["Title"]
            )
        )

        content.append(Spacer(1, 20))

        content.append(
            Paragraph(
                "Summary",
                styles["Heading2"]
            )
        )

        content.append(
            Paragraph(
                escape(summary),
                styles["BodyText"]
            )
        )

        content.append(Spacer(1, 20))

        content.append(
            Paragraph(
                "Key Points",
                styles["Heading2"]
            )
        )

        for point in points:

            content.append(
                Paragraph(
                    "&#8226; " + escape(point),
                    styles["BodyText"]
                )
            )

            content.append(Spacer(1, 6))

        content.append(Spacer(1, 15))

        content.append(
            Paragraph(
                "Keywords",
                styles["Heading2"]
            )
        )

        content.append(
            Paragraph(
                escape(", ".join(keywords)),
                styles["BodyText"]
            )
        )

        document.build(content)

        QMessageBox.information(
            None,
            "Success",
            "Summary exported successfully!"
        )

    except Exception as e:

        QMessageBox.critical(
            None,
            "Export Error",
            str(e)
        )


# ==========================================
# MAIN APPLICATION
# ==========================================

class SummarizerApp(QMainWindow):

    def __init__(self):

        super().__init__()

        self.setWindowTitle(
            "AI Document Summarizer"
        )

        self.setGeometry(100, 100, 1100, 750)

        self.summary = ""
        self.key_points = []
        self.keywords = []

        self.worker = None

        self.setup_ui()

    # --------------------------------------

    
    def setup_ui(self):

        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(35, 25, 35, 25)
        main_layout.setSpacing(18)

        # =================================
        # GLOBAL STYLESHEET
        # =================================

        self.setStyleSheet("""
            QMainWindow {
                background-color: #F5F3FA;
            }

            QWidget {
                font-family: 'Segoe UI';
                color: #45445A;
                font-size: 13px;
            }

            QWidget#card {
                background-color: #FFFFFF;
                border: 1px solid #EAE6F2;
                border-radius: 16px;
            }

            QLabel#title {
                font-size: 27px;
                font-weight: 700;
                color: #655783;
            }

            QLabel#subtitle {
                font-size: 13px;
                color: #9692A8;
            }

            QLabel#sectionTitle {
                font-size: 15px;
                font-weight: 600;
                color: #655783;
            }

            QLabel#status {
                color: #789B86;
                font-size: 12px;
            }

            QLabel#device {
                background-color: #EDF5F0;
                color: #638B73;
                padding: 7px 12px;
                border-radius: 10px;
                font-size: 11px;
                font-weight: 600;
            }

            QTextEdit {
                background-color: #FBFAFD;
                border: 1px solid #EAE6F2;
                border-radius: 11px;
                padding: 13px;
                color: #555268;
                selection-background-color: #DCD2EE;
                font-size: 13px;
            }

            QTextEdit:focus {
                border: 1px solid #B9A9D8;
            }

            QComboBox {
                background-color: #F5F2FA;
                border: 1px solid #E6DFF1;
                border-radius: 9px;
                padding: 9px 14px;
                min-width: 120px;
                color: #655783;
            }

            QComboBox::drop-down {
                border: none;
                width: 25px;
            }

            QComboBox QAbstractItemView {
                background-color: white;
                selection-background-color: #EDE7F6;
                color: #655783;
                border: 1px solid #E6DFF1;
            }

            QPushButton {
                border: none;
                border-radius: 10px;
                padding: 11px 18px;
                font-weight: 600;
            }

            QPushButton#uploadButton {
                background-color: #EDE7F6;
                color: #78649F;
            }

            QPushButton#uploadButton:hover {
                background-color: #E2D9F0;
            }

            QPushButton#clearButton {
                background-color: #F5F2F7;
                color: #89839A;
            }

            QPushButton#clearButton:hover {
                background-color: #ECE7F1;
            }

            QPushButton#generateButton {
                background-color: #9B88BE;
                color: white;
                padding: 14px;
                font-size: 14px;
            }

            QPushButton#generateButton:hover {
                background-color: #8975AE;
            }

            QPushButton#generateButton:disabled {
                background-color: #C9C1D7;
                color: #F7F5FA;
            }

            QPushButton#exportButton {
                background-color: #E5F0E9;
                color: #638B73;
            }

            QPushButton#exportButton:hover {
                background-color: #D7E9DE;
            }

            QPushButton:disabled {
                color: #AAA5B4;
            }

            QTabWidget::pane {
                border: 1px solid #EAE6F2;
                border-radius: 10px;
                background-color: white;
                top: -1px;
            }

            QTabBar::tab {
                background-color: #F5F3FA;
                color: #8B859B;
                padding: 11px 20px;
                margin-right: 5px;
                border-top-left-radius: 9px;
                border-top-right-radius: 9px;
            }

            QTabBar::tab:selected {
                background-color: #EDE7F6;
                color: #756394;
                font-weight: 600;
            }

            QProgressBar {
                background-color: #EEEAF4;
                border: none;
                border-radius: 5px;
                height: 8px;
                text-align: center;
            }

            QProgressBar::chunk {
                background-color: #A895C8;
                border-radius: 5px;
            }
        """)

        # =================================
        # HEADER
        # =================================

        header = QHBoxLayout()

        header_left = QVBoxLayout()

        title = QLabel("AI Document Summarizer")
        title.setObjectName("title")

        subtitle = QLabel(
            "Turn lengthy documents into clear, meaningful insights."
        )
        subtitle.setObjectName("subtitle")

        header_left.addWidget(title)
        header_left.addWidget(subtitle)

        header.addLayout(header_left)
        header.addStretch()

        self.device_label = QLabel(
            f"●  AI ENGINE: {DEVICE.upper()}"
        )
        self.device_label.setObjectName("device")

        header.addWidget(self.device_label)

        main_layout.addLayout(header)

        # =================================
        # INPUT CARD
        # =================================

        input_card = QWidget()
        input_card.setObjectName("card")

        input_layout = QVBoxLayout(input_card)
        input_layout.setContentsMargins(20, 18, 20, 20)
        input_layout.setSpacing(12)

        input_header = QHBoxLayout()

        input_label = QLabel("Your Document")
        input_label.setObjectName("sectionTitle")

        input_hint = QLabel(
            "Upload a PDF or paste your text below"
        )
        input_hint.setObjectName("subtitle")

        input_header.addWidget(input_label)
        input_header.addStretch()
        input_header.addWidget(input_hint)

        input_layout.addLayout(input_header)

        self.input_text = QTextEdit()

        self.input_text.setPlaceholderText(
            "Start typing or paste your paragraph here...\n\n"
            "You can also upload a PDF document using the button below."
        )

        self.input_text.setMinimumHeight(180)

        input_layout.addWidget(self.input_text)

        # Input action buttons

        action_layout = QHBoxLayout()

        self.upload_button = QPushButton(
            "＋  Upload PDF"
        )
        self.upload_button.setObjectName("uploadButton")

        self.upload_button.clicked.connect(
            self.upload_pdf
        )

        self.clear_button = QPushButton(
            "Clear Text"
        )
        self.clear_button.setObjectName("clearButton")

        self.clear_button.clicked.connect(
            self.clear_input
        )

        action_layout.addWidget(self.upload_button)
        action_layout.addWidget(self.clear_button)
        action_layout.addStretch()

        input_layout.addLayout(action_layout)

        main_layout.addWidget(input_card)

        # =================================
        # SETTINGS CARD
        # =================================

        settings_card = QWidget()
        settings_card.setObjectName("card")

        settings_layout = QHBoxLayout(settings_card)

        settings_layout.setContentsMargins(
            20, 15, 20, 15
        )

        settings_label = QLabel(
            "Summary Preferences"
        )
        settings_label.setObjectName("sectionTitle")

        settings_layout.addWidget(settings_label)

        settings_layout.addStretch()

        length_label = QLabel("Summary Length:")

        self.length_combo = QComboBox()

        self.length_combo.addItems([
            "Short",
            "Medium",
            "Long"
        ])

        self.length_combo.setCurrentText("Medium")

        settings_layout.addWidget(length_label)
        settings_layout.addWidget(self.length_combo)

        main_layout.addWidget(settings_card)

        # =================================
        # GENERATE BUTTON
        # =================================

        self.generate_button = QPushButton(
            "✦   Generate Summary"
        )

        self.generate_button.setObjectName(
            "generateButton"
        )

        self.generate_button.clicked.connect(
            self.generate
        )

        main_layout.addWidget(self.generate_button)

        # =================================
        # PROGRESS AND STATUS
        # =================================

        self.progress = QProgressBar()

        self.progress.setRange(0, 0)
        self.progress.hide()

        main_layout.addWidget(self.progress)

        self.status_label = QLabel(
            "●  Ready to summarize your document"
        )

        self.status_label.setObjectName("status")

        main_layout.addWidget(self.status_label)

        # =================================
        # RESULTS CARD
        # =================================

        result_card = QWidget()
        result_card.setObjectName("card")

        result_layout = QVBoxLayout(result_card)

        result_layout.setContentsMargins(
            20, 18, 20, 20
        )

        result_layout.setSpacing(12)

        result_title = QLabel("Your Results")
        result_title.setObjectName("sectionTitle")

        result_layout.addWidget(result_title)

        self.tabs = QTabWidget()

        self.summary_output = QTextEdit()
        self.summary_output.setReadOnly(True)

        self.summary_output.setPlaceholderText(
            "Your AI-generated summary will appear here..."
        )

        self.points_output = QTextEdit()
        self.points_output.setReadOnly(True)

        self.points_output.setPlaceholderText(
            "Important key points will appear here..."
        )

        self.keywords_output = QTextEdit()
        self.keywords_output.setReadOnly(True)

        self.keywords_output.setPlaceholderText(
            "Extracted keywords will appear here..."
        )

        self.tabs.addTab(
            self.summary_output,
            "✦  Summary"
        )

        self.tabs.addTab(
            self.points_output,
            "☷  Key Points"
        )

        self.tabs.addTab(
            self.keywords_output,
            "◇  Keywords"
        )

        result_layout.addWidget(self.tabs)

        main_layout.addWidget(result_card, 1)

        # =================================
        # EXPORT SECTION
        # =================================

        footer = QHBoxLayout()

        footer_label = QLabel(
            "Your insights, ready to save."
        )
        footer_label.setObjectName("subtitle")

        footer.addWidget(footer_label)
        footer.addStretch()

        self.export_button = QPushButton(
            "↓  Export Summary to PDF"
        )

        self.export_button.setObjectName(
            "exportButton"
        )

        self.export_button.setEnabled(False)

        self.export_button.clicked.connect(
            self.export_summary
        )

        footer.addWidget(self.export_button)

        main_layout.addLayout(footer)


    def upload_pdf(self):

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select PDF Document",
            "",
            "PDF Files (*.pdf)"
        )

        if file_path:

            try:

                self.status_label.setText(
                    "Extracting PDF text..."
                )

                QApplication.processEvents()

                text = extract_pdf_text(file_path)

                if not text.strip():

                    QMessageBox.warning(
                        self,
                        "Empty PDF",
                        "No selectable text was found. "
                        "This PDF may require OCR."
                    )

                    self.status_label.setText("Ready.")
                    return

                self.input_text.setPlainText(text)

                self.status_label.setText(
                    f"PDF loaded: {os.path.basename(file_path)}"
                )

            except Exception as e:

                QMessageBox.critical(
                    self,
                    "PDF Error",
                    str(e)
                )

                self.status_label.setText("Ready.")

    # --------------------------------------
    # CLEAR INPUT
    # --------------------------------------

    def clear_input(self):

        self.input_text.clear()

        self.summary_output.clear()
        self.points_output.clear()
        self.keywords_output.clear()

        self.summary = ""
        self.key_points = []
        self.keywords = []

        self.export_button.setEnabled(False)

        self.status_label.setText("Ready.")

    # --------------------------------------
    # GENERATE SUMMARY
    # --------------------------------------

    def generate(self):

        text = self.input_text.toPlainText()

        if not text.strip():

            QMessageBox.warning(
                self,
                "Missing Input",
                "Please upload a PDF or enter some text."
            )

            return

        self.generate_button.setEnabled(False)
        self.upload_button.setEnabled(False)
        self.clear_button.setEnabled(False)
        self.export_button.setEnabled(False)

        self.progress.show()

        self.status_label.setText(
            "AI is processing your document..."
        )

        self.worker = SummaryWorker(
            text,
            self.length_combo.currentText()
        )

        self.worker.finished.connect(
            self.display_results
        )

        self.worker.error.connect(
            self.show_error
        )

        self.worker.finished.connect(
            self.worker.deleteLater
        )

        self.worker.error.connect(
            self.worker.deleteLater
        )

        self.worker.finished.connect(
            self.processing_complete
        )

        self.worker.error.connect(
            self.processing_complete
        )

        self.worker.start()

    # --------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------

    def display_results(
        self,
        summary,
        points,
        keywords
    ):

        self.summary = summary
        self.key_points = points
        self.keywords = keywords

        self.summary_output.setPlainText(
            summary
        )

        self.points_output.setPlainText(
            "\n".join(
                f"{i + 1}. {point}"
                for i, point in enumerate(points)
            )
        )

        self.keywords_output.setPlainText(
            ", ".join(keywords)
        )

        self.export_button.setEnabled(True)

        self.status_label.setText(
            "Summary generated successfully!"
        )

        self.tabs.setCurrentIndex(0)

    # --------------------------------------
    # HANDLE ERRORS
    # --------------------------------------

    def show_error(self, message):

        QMessageBox.critical(
            self,
            "Processing Error",
            message
        )

        self.status_label.setText(
            "An error occurred."
        )

    # --------------------------------------
    # FINISH PROCESSING
    # --------------------------------------

    def processing_complete(self, *args):

        self.progress.hide()

        self.generate_button.setEnabled(True)
        self.upload_button.setEnabled(True)
        self.clear_button.setEnabled(True)

        self.worker = None

    # --------------------------------------
    # EXPORT PDF
    # --------------------------------------

    def export_summary(self):

        if not self.summary:
            return

        export_to_pdf(
            self.summary,
            self.key_points,
            self.keywords
        )


# ==========================================
# RUN APPLICATION
# ==========================================

if __name__ == "__main__":

    app = QApplication(sys.argv)

    window = SummarizerApp()

    window.show()

    sys.exit(app.exec_())
