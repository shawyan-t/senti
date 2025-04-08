import PyPDF2
import io
import os
import tempfile
import trafilatura
import requests
from bs4 import BeautifulSoup

def process_text_input(text):
    """
    Process raw text input.
    
    Args:
        text (str): The raw text input
        
    Returns:
        str: The processed text
    """
    # Simple processing - remove extra whitespace and normalize
    processed_text = " ".join(text.split())
    return processed_text

def extract_text_from_pdf(pdf_path):
    """
    Extract text from a PDF file.
    
    Args:
        pdf_path (str): Path to the PDF file
        
    Returns:
        str: Extracted text from the PDF
    """
    try:
        text = ""
        with open(pdf_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num in range(len(pdf_reader.pages)):
                page = pdf_reader.pages[page_num]
                text += page.extract_text() + "\n"
        
        # Clean up the extracted text
        text = " ".join(text.split())
        return text
    except Exception as e:
        print(f"Error extracting text from PDF: {e}")
        return None

def extract_text_from_html(url):
    """
    Extract text content from a website URL.
    
    Args:
        url (str): The URL of the website
        
    Returns:
        str: Extracted text content
    """
    try:
        # Use trafilatura for better text extraction
        downloaded = trafilatura.fetch_url(url)
        extracted_text = trafilatura.extract(downloaded)
        
        if extracted_text:
            return extracted_text
        
        # Fallback to BeautifulSoup if trafilatura doesn't extract text
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.extract()
        
        # Get text
        text = soup.get_text()
        
        # Break into lines and remove leading and trailing space on each
        lines = (line.strip() for line in text.splitlines())
        
        # Break multi-headlines into a line each
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        
        # Drop blank lines
        text = '\n'.join(chunk for chunk in chunks if chunk)
        
        return text
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return None
