import os
import re
import logging
from tqdm import tqdm
import fitz  # PyMuPDF
import csv
from langdetect import detect, LangDetectException

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize counters for skipped paragraphs
skip_counters = {
    "table_of_contents": 0,
    "acronyms": 0,
    "non_latin": 0,
    "short_paragraph": 0,
    "too_many_numbers": 0,
    "all_caps": 0,
    "references": 0,
    "unwanted_language": 0  # New counter for unwanted languages
}

# Define unwanted languages
# Define unwanted languages
UNWANTED_LANGUAGES = {'id', 'fr', 'es', 'ar', 'th'}  # 'id' for Bahasa Indonesian, 'fr' for French, 'es' for Spanish, 'ar' for Arabic, 'th' for Thai


def clean_text(text: str) -> str:
    """Clean extracted text"""
    # Remove page numbers
    text = re.sub(r'\b\d+\b(?:\s*\|\s*\d+)*', '', text)
    # Remove figure and table references
    text = re.sub(r'\b(Figure|Table)\s*\d+', '', text)
    # Remove URLs
    text = re.sub(r'http[s]?://(?:[a-zA-Z0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', text)
    # Remove email addresses
    text = re.sub(r'[\w\.-]+@[\w\.-]+\.\w+', '', text)
    # Remove multiple spaces and newlines
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def is_valid_paragraph(text: str, min_words: int = 10) -> bool:
    """Check if text is a valid paragraph"""
    global skip_counters

    # Skip paragraphs that include a table of contents
    if '...................................................................................................................................................' in text:
        skip_counters["table_of_contents"] += 1
        return False

    # Skip paragraphs containing "ACRONYMS AND ABBREVIATIONS"
    if "ACRONYMS AND ABBREVIATIONS" in text:
        skip_counters["acronyms"] += 1
        return False

    # Skip paragraphs containing non-Latin characters (Thai or Arabic)
    if re.search(r'[\u0E00-\u0E7F\u0600-\u06FF]', text):  # Thai or Arabic characters
        skip_counters["non_latin"] += 1
        return False

    # Skip if the paragraph is too short
    if len(text.split()) < min_words:
        skip_counters["short_paragraph"] += 1
        return False

    # Skip if the paragraph contains too many numbers or special characters
    if len(re.findall(r'\d', text)) / len(text) > 0.2:
        skip_counters["too_many_numbers"] += 1
        return False

    # Skip all-caps paragraphs (likely headers)
    if text.isupper():
        skip_counters["all_caps"] += 1
        return False

    # Skip references (like "1.", "et al.", etc.)
    if re.match(r'^(\d+\.|[A-Z][a-z]+\s*,|et al\.)', text):
        skip_counters["references"] += 1
        return False

    # Skip paragraphs in unwanted languages
    try:
        language = detect(text)  # Detect the language of the paragraph
        if language in UNWANTED_LANGUAGES:
            skip_counters["unwanted_language"] += 1
            return False
    except LangDetectException:
        # If langdetect fails, assume it's not one of the unwanted languages
        pass

    return True


def extract_paragraphs(pdf_path):
    try:
        # Ensure pdf_path is a string
        pdf_path = str(pdf_path)

        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        paragraphs = []
        with fitz.open(pdf_path) as doc:
            for page_num in range(len(doc)):
                try:
                    page = doc[page_num]
                    text = page.get_textpage().extractText()
                    # Split text into paragraphs and filter empty ones
                    page_paragraphs = [p.strip() for p in text.split('\n\n')
                                       if p.strip() and is_valid_paragraph(p.strip())]
                    paragraphs.extend(page_paragraphs)
                except Exception as page_error:
                    logger.warning(f"Skipping page {page_num} in {pdf_path}: {str(page_error)}")
                    continue
        return paragraphs

    except fitz.FileDataError as e:
        logger.error(f"Corrupted PDF {pdf_path}: {str(e)}")
        return []
    except Exception as e:
        logger.error(f"Error processing {pdf_path}: {str(e)}")
        return []


def extract_year_from_filename(filename: str) -> str:
    """Extract year from filename pattern document_YYYY-MM-DD_XXX.pdf"""
    match = re.search(r'document_(\d{4})-', filename)
    return match.group(1) if match else ""


def process_pdfs(pdf_dir, output_csv):
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(pdf_dir) if f.endswith('.pdf')]

    # Create CSV with headers
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['filename', 'year', 'text'])

    # Process in batches
    batch_size = 50
    for i in range(0, len(pdf_files), batch_size):
        batch = pdf_files[i:i + batch_size]

        with tqdm(batch, desc=f"Batch {i // batch_size + 1}") as pbar:
            for pdf_file in pbar:
                pdf_path = os.path.join(pdf_dir, pdf_file)
                try:
                    year = extract_year_from_filename(pdf_file)
                    paragraphs = extract_paragraphs(pdf_path)
                    if paragraphs:
                        with open(output_csv, 'a', newline='', encoding='utf-8') as f:
                            writer = csv.writer(f)
                            for p in paragraphs:
                                writer.writerow([pdf_file, year, p])
                except Exception as e:
                    logging.error(f"Error processing {pdf_path}: {str(e)}")
                    continue


def main(pdf_dir: str, output_csv: str) -> None:
    """Main function"""
    logger.info("Starting PDF processing...")

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    if os.path.exists(output_csv):
        os.remove(output_csv)

    process_pdfs(pdf_dir, output_csv)

    # After processing, log the skipped counts
    logger.info("Processing completed!")
    logger.info(f"Skipped paragraphs due to the following reasons:")
    for reason, count in skip_counters.items():
        logger.info(f"{reason}: {count}")


if __name__ == "__main__":
    pdf_dir = '/Users/liampowell/PycharmProjects/BERTopic Pilot/pdfs'
    output_csv = '/Users/liampowell/PycharmProjects/BERTopic Pilot/processed_paragraphs_filtered.csv'
    main(pdf_dir, output_csv)