"""
Module for processing various text inputs (PDF, CSV, HTML, etc.)
"""
import PyPDF2
import io
import re
import os
import tempfile
import trafilatura
import requests
import pandas as pd
from .openai_client import determine_input_type

def process_text_input(text):
    """
    Process raw text input.
    
    Args:
        text (str): The raw text input
        
    Returns:
        str: The processed text
    """
    # Basic cleaning: remove excessive whitespace
    processed_text = re.sub(r'\s+', ' ', text).strip()
    
    return processed_text

def detect_file_type(file_content):
    """
    Detect the type of file from its content.
    
    Args:
        file_content (bytes): Binary content of the file
        
    Returns:
        str: Detected file type ('csv', 'json', 'text', etc.)
    """
    # Look for CSV characteristics
    content_sample = file_content[:1024].decode('utf-8', errors='ignore')
    
    if content_sample.count(',') > 5 and '\n' in content_sample:
        lines = content_sample.split('\n')
        if len(lines) > 1 and lines[0].count(',') == lines[1].count(','):
            return 'csv'
    
    # Look for JSON characteristics
    if content_sample.strip().startswith('{') and '}' in content_sample:
        return 'json'
    
    # Look for XML characteristics
    if content_sample.strip().startswith('<') and '>' in content_sample:
        return 'xml'
    
    # Default to text
    return 'text'

def process_csv_data(csv_content):
    """
    Process CSV data into a readable text format.
    
    Args:
        csv_content (str): CSV content as string
        
    Returns:
        str: Processed text describing the CSV data
    """
    try:
        # Parse CSV with pandas
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Get basic information about the data
        num_rows, num_cols = df.shape
        columns = df.columns.tolist()
        
        # Generate summary statistics for numerical columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        stats = df[numeric_cols].describe().to_string() if len(numeric_cols) > 0 else ""
        
        # Generate a sample of the data (first 5 rows)
        sample = df.head(5).to_string()
        
        # Combine into a readable format
        text_output = f"""
        CSV DATA SUMMARY:
        
        Number of rows: {num_rows}
        Number of columns: {num_cols}
        Columns: {', '.join(columns)}
        
        Sample data (first 5 rows):
        {sample}
        
        Summary statistics for numerical columns:
        {stats}
        """
        
        return text_output
    
    except Exception as e:
        return f"Error processing CSV data: {str(e)}\n\nRaw content:\n{csv_content[:1000]}..."

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
            reader = PyPDF2.PdfReader(file)
            num_pages = len(reader.pages)
            
            for i in range(num_pages):
                page = reader.pages[i]
                page_text = page.extract_text()
                if page_text:
                    text += f"\n--- Page {i+1} ---\n{page_text}"
        
        return text
    
    except Exception as e:
        return f"Error extracting text from PDF: {str(e)}"

def extract_text_from_html(url):
    """
    Extract text content from a website URL.
    Uses trafilatura for better content extraction.
    
    Args:
        url (str): The URL of the website
        
    Returns:
        str: Extracted text content
    """
    try:
        # First, try with trafilatura for better content extraction
        downloaded = trafilatura.fetch_url(url)
        
        if downloaded:
            text = trafilatura.extract(downloaded)
            
            if text and len(text) > 200:  # If we got meaningful content
                return text
        
        # Fallback to basic request if trafilatura fails
        response = requests.get(url, headers={
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        if response.status_code == 200:
            # Try trafilatura one more time with response content
            extracted_text = trafilatura.extract(response.text)
            
            if extracted_text and len(extracted_text) > 200:
                return extracted_text
            
            # Last resort: return raw HTML (the LLM can still extract some meaning)
            return f"Raw HTML content from {url}:\n\n{response.text[:10000]}..."
        
        return f"Failed to fetch content from {url}. Status code: {response.status_code}"
    
    except Exception as e:
        return f"Error extracting content from URL: {str(e)}"

def extract_csv_from_file(file_path):
    """
    Extract and format data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        str: Formatted text representation of the CSV data
    """
    try:
        # Read the CSV file
        df = pd.read_csv(file_path)
        
        # Format the data as a readable string
        return process_csv_data(df.to_csv(index=False))
    
    except Exception as e:
        return f"Error extracting data from CSV file: {str(e)}"