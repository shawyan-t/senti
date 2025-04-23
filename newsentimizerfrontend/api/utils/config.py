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
    # Try loading from multiple possible locations
    env_loaded = False
    
    # Try the API directory first
    api_dir = Path(__file__).parent.parent
    api_env_path = api_dir / '.env'
    if api_env_path.exists():
        print(f"Loading .env from API directory: {api_env_path}")
        load_dotenv(api_env_path)
        env_loaded = True
    
    # Get the project root directory (2 levels up from this file)
    project_root = Path(__file__).parent.parent.parent
    env_path = project_root / '.env'
    
    # Load .env file if it exists
    if env_path.exists():
        print(f"Loading .env from project root: {env_path}")
        load_dotenv(env_path)
        env_loaded = True
    
    # Configuration dictionary
    config = {
        'openai_api_key': os.getenv('OPENAI_API_KEY'),
        'twitter_api_key': os.getenv('TWITTER_API_KEY'),
        'twitter_api_secret': os.getenv('TWITTER_API_SECRET'),
        'news_api_key': os.getenv('NEWSAPI_KEY'),
        'google_search_api_key': os.getenv('GOOGLE_SEARCH_API_KEY'),
        'google_search_cx': os.getenv('GOOGLE_SEARCH_CX')
    }
    
    # Double check for required keys
    if not config['openai_api_key']:
        raise ValueError(
            "OpenAI API key is required but not found. "
            "Please set it in your .env file or as an environment variable."
        )
    
    return config

# Load configuration on module import
config = load_config() 