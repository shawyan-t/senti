"""
Configuration module for managing API keys and environment variables.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

def load_config():
    """
    Load configuration from .env file and environment variables.
    Prioritizes environment variables over .env file values.
    """
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    
    # Load .env file if it exists
    if env_path.exists():
        load_dotenv(env_path)
    
    # Configuration dictionary
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'twitter_api_key': os.getenv('TWITTER_API_KEY'),
        'twitter_api_secret': os.getenv('TWITTER_API_SECRET'),
        'news_api_key': os.getenv('NEWSAPI_KEY'),
        'google_search_api_key': os.getenv('GOOGLE_SEARCH_API_KEY'),
        'google_search_cx': os.getenv('GOOGLE_SEARCH_CX')
    }
    
    # Check for required keys
    if not config['openai_api_key']:
        raise ValueError(
            "OpenAI API key is required but not found. "
            "Please set it in your .env file or as an environment variable."
        )
    
    return config

# Load configuration on module import
config = load_config() 