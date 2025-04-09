"""
Module for handling data storage and retrieval of analysis results.
"""
import os
import json
import uuid
from datetime import datetime

# Define the data directory
DATA_DIR = "data/analyses"

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
    # Create a unique ID for this analysis
    analysis_id = str(uuid.uuid4())
    
    # Create the data structure
    analysis_data = {
        "id": analysis_id,
        "source": source,
        "source_type": source_type,
        "text": text[:1000] + "..." if len(text) > 1000 else text,  # Store a truncated version
        "summary": summary,
        "sentiment": sentiment,
        "metadata": metadata,
        "timestamp": timestamp.isoformat()
    }
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # Save to a JSON file
    file_path = os.path.join(DATA_DIR, f"{analysis_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
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
    
    if not os.path.exists(file_path):
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading analysis {analysis_id}: {e}")
        return None

def load_all_analyses():
    """
    Load all analyses from the data directory.
    
    Returns:
        dict: A dictionary of all analyses, keyed by ID
    """
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    analyses = {}
    
    # List all JSON files in the directory
    for filename in os.listdir(DATA_DIR):
        if filename.endswith('.json'):
            analysis_id = filename.replace('.json', '')
            analysis_data = load_analysis(analysis_id)
            if analysis_data:
                analyses[analysis_id] = analysis_data
    
    return analyses

def get_filtered_data(analyses, start_date=None, end_date=None, topics=None, regions=None, commodities=None, sentiments=None):
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
    # Convert to list for easier filtering
    analysis_list = list(analyses.values())
    filtered_list = []
    
    for analysis in analysis_list:
        # Convert string timestamp to datetime
        if isinstance(analysis.get('timestamp'), str):
            try:
                analysis_date = datetime.fromisoformat(analysis.get('timestamp'))
            except:
                # Skip if we can't parse the date
                continue
        else:
            analysis_date = analysis.get('timestamp')
        
        # Filter by date range
        if start_date and analysis_date.date() < start_date:
            continue
        if end_date and analysis_date.date() > end_date:
            continue
        
        # Filter by topics
        if topics and not contains_any(analysis.get('metadata', {}).get('topics', []), topics):
            continue
        
        # Filter by regions
        if regions and not contains_any(analysis.get('metadata', {}).get('regions', []), regions):
            continue
        
        # Filter by commodities
        if commodities and not contains_any(analysis.get('metadata', {}).get('commodities', []), commodities):
            continue
        
        # Filter by sentiment
        if sentiments and analysis.get('sentiment', {}).get('sentiment') not in sentiments:
            continue
        
        # If we get here, the analysis passes all filters
        filtered_list.append(analysis)
    
    # Convert back to dictionary
    filtered_dict = {analysis['id']: analysis for analysis in filtered_list}
    return filtered_dict

def contains_any(items, filter_list):
    """
    Check if any items from filter_list are in items.
    
    Args:
        items (list): List of items to check
        filter_list (list): List of items to check for
        
    Returns:
        bool: True if any items from filter_list are in items
    """
    if not filter_list:
        return True
    
    if not items:
        return False
    
    # Convert both to lowercase for case-insensitive comparison
    items_lower = [str(item).lower() for item in items]
    filter_lower = [str(f).lower() for f in filter_list]
    
    # Check if any of the filter items are in the items list
    return any(f in items_lower for f in filter_lower)