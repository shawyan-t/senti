import PyPDF2
import io
import os
import tempfile
import trafilatura
import requests
import csv
import pandas as pd
import re
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

def detect_file_type(file_content):
    """
    Detect the type of file from its content.
    
    Args:
        file_content (bytes): Binary content of the file
        
    Returns:
        str: Detected file type ('csv', 'json', 'text', etc.)
    """
    # Check if it's CSV
    try:
        # Try to decode as UTF-8 and check for comma-separated values
        text_content = file_content.decode('utf-8')
        if ',' in text_content and '\n' in text_content:
            # Check if it has a consistent number of commas per line
            lines = text_content.split('\n')[:5]  # Check first 5 lines
            comma_counts = [line.count(',') for line in lines if line.strip()]
            if len(set(comma_counts)) == 1:  # All lines have same number of commas
                return 'csv'
    except:
        pass
    
    # Check if it's JSON
    try:
        import json
        json.loads(file_content.decode('utf-8'))
        return 'json'
    except:
        pass
    
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
        # Parse CSV into a DataFrame
        df = pd.read_csv(io.StringIO(csv_content))
        
        # Basic information about the dataset
        num_rows, num_cols = df.shape
        column_names = list(df.columns)
        
        # Create a summary of the data
        summary = f"CSV data with {num_rows} rows and {num_cols} columns.\n\n"
        summary += f"Column names: {', '.join(column_names)}\n\n"
        
        # Sample data (first 5 rows)
        summary += "Sample data (first 5 rows):\n"
        summary += df.head(5).to_string()
        summary += "\n\n"
        
        # Basic statistics for numerical columns
        summary += "Numerical column statistics:\n"
        for col in df.select_dtypes(include=['int64', 'float64']).columns:
            summary += f"{col}: min={df[col].min()}, max={df[col].max()}, mean={df[col].mean():.2f}, median={df[col].median()}\n"
        
        return summary
    except Exception as e:
        print(f"Error processing CSV data: {e}")
        return "Error processing CSV data."

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
        # Check if it's actually a URL or just text
        if not url.startswith(('http://', 'https://')):
            return process_text_input(url)
            
        # Use trafilatura for better text extraction
        downloaded = trafilatura.fetch_url(url)
        extracted_text = trafilatura.extract(downloaded)
        
        if extracted_text:
            # Add source information
            return f"Source URL: {url}\n\n{extracted_text}"
        
        # Fallback to BeautifulSoup if trafilatura doesn't extract text
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Try to get page title
        title = soup.title.string if soup.title else "Untitled Page"
        
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
        
        return f"Source URL: {url}\nTitle: {title}\n\n{text}"
    except Exception as e:
        print(f"Error extracting text from URL: {e}")
        return f"Failed to extract content from URL: {url}. Error: {str(e)}"

def extract_csv_from_file(file_path):
    """
    Extract and format data from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file
        
    Returns:
        str: Formatted text representation of the CSV data
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            csv_content = file.read()
        return process_csv_data(csv_content)
    except Exception as e:
        print(f"Error processing CSV file: {e}")
        return None
