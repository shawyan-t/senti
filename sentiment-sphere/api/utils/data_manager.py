"""
Module for handling data storage and retrieval of analysis results.
"""
import os
import json
import uuid
from datetime import datetime

# Get the correct path to the newsentimizerfrontend directory
FRONTEND_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
print(f"Frontend root: {FRONTEND_ROOT}")

# Define the paths to the data directories
DATA_DIR = os.path.join(FRONTEND_ROOT, "data")
ANALYSES_DIR = os.path.join(DATA_DIR, "analyses")

# Create the directories if they don't exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(ANALYSES_DIR, exist_ok=True)

# Log the paths for debugging
print(f"Data directory: {DATA_DIR}")
print(f"Analyses directory: {ANALYSES_DIR}")
print(f"Data directory exists: {os.path.exists(DATA_DIR)}")
print(f"Analyses directory exists: {os.path.exists(ANALYSES_DIR)}")

def save_analysis(analysis_data):
    """
    Save analysis data to a JSON file
    
    Parameters:
    -----------
    analysis_data : dict
        The analysis data to save
    
    Returns:
    --------
    str
        The ID of the saved analysis
    """
    # Generate a unique ID for the analysis
    analysis_id = str(uuid.uuid4())
    
    # Add timestamp if not present
    if 'timestamp' not in analysis_data:
        analysis_data['timestamp'] = datetime.now().isoformat()
    
    # Add ID to the data
    analysis_data['id'] = analysis_id
    
    # Save to JSON file
    file_path = os.path.join(ANALYSES_DIR, f"{analysis_id}.json")
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(analysis_data, f, ensure_ascii=False, indent=2)
    
    print(f"Analysis saved to {file_path}")
    return analysis_id

def load_analysis(analysis_id):
    """
    Load a specific analysis from a JSON file
    
    Parameters:
    -----------
    analysis_id : str
        The ID of the analysis to load
    
    Returns:
    --------
    dict or None
        The analysis data if found, None otherwise
    """
    # First try in the analyses directory
    file_path = os.path.join(ANALYSES_DIR, f"{analysis_id}.json")
    
    # If not found there, try in the data directory root
    if not os.path.exists(file_path):
        file_path = os.path.join(DATA_DIR, f"{analysis_id}.json")
    
    if not os.path.exists(file_path):
        print(f"Analysis file not found: {file_path}")
        return None
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            print(f"Loaded analysis from {file_path}")
            return data
    except Exception as e:
        print(f"Error loading analysis {analysis_id}: {e}")
        return None

def load_all_analyses():
    """
    Load all analyses from the analyses directory and data directory
    
    Returns:
    --------
    dict
        A dictionary of all analyses, keyed by ID
    """
    analyses = {}
    
    # Function to process files in a directory
    def process_directory(directory, file_pattern="*.json"):
        if not os.path.exists(directory):
            print(f"Directory does not exist: {directory}")
            return
        
        print(f"Scanning directory for analyses: {directory}")
        
        # List all JSON files in the directory
        for filename in os.listdir(directory):
            if filename.endswith('.json'):
                analysis_id = filename.replace('.json', '')
                file_path = os.path.join(directory, filename)
                
                if os.path.isfile(file_path):
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            analysis_data = json.load(f)
                            
                            # Ensure the analysis has an ID
                            if 'id' not in analysis_data:
                                analysis_data['id'] = analysis_id
                            
                            analyses[analysis_id] = analysis_data
                            print(f"Loaded analysis {analysis_id} from {file_path}")
                    except Exception as e:
                        print(f"Error loading analysis from {file_path}: {e}")
    
    # Process files in the analyses directory
    process_directory(ANALYSES_DIR)
    
    # Process files in the data directory (excluding analysis subdirectory files)
    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        if os.path.isfile(file_path) and filename.endswith('.json'):
            analysis_id = filename.replace('.json', '')
            
            # Skip if already loaded from analyses directory
            if analysis_id in analyses:
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    analysis_data = json.load(f)
                    
                    # Ensure the analysis has an ID
                    if 'id' not in analysis_data:
                        analysis_data['id'] = analysis_id
                    
                    analyses[analysis_id] = analysis_data
                    print(f"Loaded analysis {analysis_id} from {file_path}")
            except Exception as e:
                print(f"Error loading analysis from {file_path}: {e}")
    
    print(f"Total analyses loaded: {len(analyses)}")
    return analyses

def delete_analysis(analysis_id):
    """
    Delete a specific analysis
    
    Parameters:
    -----------
    analysis_id : str
        The ID of the analysis to delete
    
    Returns:
    --------
    bool
        True if deletion was successful, False otherwise
    """
    # Try both possible locations
    file_paths = [
        os.path.join(ANALYSES_DIR, f"{analysis_id}.json"),
        os.path.join(DATA_DIR, f"{analysis_id}.json")
    ]
    
    for file_path in file_paths:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
                print(f"Deleted analysis file: {file_path}")
                return True
            except Exception as e:
                print(f"Error deleting analysis {analysis_id}: {e}")
    
    return False

# Add backward compatibility for old data format
def save_analysis_legacy(source, content, detailed_analysis, sentiment_result, metadata_result, timestamp, source_type):
    """Legacy function for backward compatibility"""
    analysis_data = {
        "source": source,
        "content": content,
        "summary": detailed_analysis,
        "sentiment": sentiment_result,
        "metadata": metadata_result,
        "timestamp": timestamp.isoformat(),
        "source_type": source_type
    }
    return save_analysis(analysis_data)

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