import json
import os
import uuid
from datetime import datetime, date
import pandas as pd

# Directory for storing analysis results
DATA_DIR = "data"

# Ensure data directory exists
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

def save_analysis(source, text, summary, sentiment, metadata, timestamp, source_type):
    """
    Save the analysis results to a JSON file.
    
    Args:
        source (str): The source of the text (URL, file name, etc.)
        text (str): The processed text
        summary (str): The summary of the text
        sentiment (dict): The sentiment analysis results
        metadata (dict): The extracted metadata
        timestamp (datetime): When the analysis was performed
        source_type (str): Type of source (pdf, url, direct_text)
        
    Returns:
        str: The ID of the saved analysis
    """
    analysis_id = str(uuid.uuid4())
    
    analysis_data = {
        "id": analysis_id,
        "source": source,
        "source_type": source_type,
        "text": text,
        "summary": summary,
        "sentiment": sentiment,
        "metadata": metadata,
        "timestamp": timestamp.isoformat()
    }
    
    file_path = os.path.join(DATA_DIR, f"{analysis_id}.json")
    
    with open(file_path, 'w') as f:
        json.dump(analysis_data, f, indent=2)
    
    return analysis_id

def load_analysis(analysis_id):
    """
    Load an analysis by ID.
    
    Args:
        analysis_id (str): The ID of the analysis to load
        
    Returns:
        dict: The analysis data or None if not found
    """
    file_path = os.path.join(DATA_DIR, f"{analysis_id}.json")
    
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    
    return None

def load_all_analyses():
    """
    Load all analyses from the data directory.
    
    Returns:
        dict: A dictionary of all analyses, keyed by ID
    """
    analyses = {}
    
    if not os.path.exists(DATA_DIR):
        return analyses
    
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            file_path = os.path.join(DATA_DIR, filename)
            
            with open(file_path, 'r') as f:
                analysis = json.load(f)
                analyses[analysis['id']] = analysis
    
    return analyses

def get_filtered_data(analyses, start_date, end_date, topics=None, regions=None, commodities=None, sentiments=None):
    """
    Filter the analyses based on various criteria.
    
    Args:
        analyses (dict): Dictionary of all analyses
        start_date (date): Start date for filtering
        end_date (date): End date for filtering
        topics (list): List of topics to include
        regions (list): List of regions to include
        commodities (list): List of commodities to include
        sentiments (list): List of sentiments to include
        
    Returns:
        dict: Filtered analyses
    """
    if not analyses:
        return {}
    
    # Convert date objects to strings for comparison
    start_date_str = start_date.isoformat() if isinstance(start_date, date) else start_date
    end_date_str = end_date.isoformat() if isinstance(end_date, date) else end_date
    
    # Helper function to check if metadata contains any of the filter items
    def contains_any(items, filter_list):
        if not filter_list:
            return True
        return any(item in filter_list for item in items)
    
    filtered = {}
    
    for analysis_id, analysis in analyses.items():
        # Extract timestamp date part for comparison
        timestamp = analysis.get('timestamp', '')
        if timestamp:
            timestamp_date = timestamp.split('T')[0]
        else:
            continue
        
        # Check date range
        if timestamp_date < start_date_str or timestamp_date > end_date_str:
            continue
        
        # Check metadata filters
        metadata = analysis.get('metadata', {})
        
        # Check topics
        if topics and not contains_any(metadata.get('topics', []), topics):
            continue
        
        # Check regions
        if regions and not contains_any(metadata.get('regions', []), regions):
            continue
        
        # Check commodities
        if commodities and not contains_any(metadata.get('commodities', []), commodities):
            continue
        
        # Check sentiment
        sentiment = analysis.get('sentiment', {}).get('sentiment', '')
        if sentiments and sentiment not in sentiments:
            continue
        
        # If we got here, the analysis passed all filters
        filtered[analysis_id] = analysis
    
    return filtered
