import os
import json
from openai import OpenAI

# Get API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def summarize_text(text):
    """
    Summarize the input text using OpenAI's API.
    
    Args:
        text (str): The text to summarize
        
    Returns:
        str: A summary of the text
    """
    try:
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert text summarizer. Create a concise summary that captures the key points and overall message of the text."
                },
                {"role": "user", "content": text}
            ],
            max_tokens=500
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error in summarize_text: {e}")
        return "Failed to generate summary. Please try again."

def analyze_sentiment(text):
    """
    Analyze the sentiment of the input text using OpenAI's API.
    
    Args:
        text (str): The text to analyze
        
    Returns:
        dict: A dictionary containing sentiment analysis results
    """
    try:
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": "You are a sentiment analysis expert. Analyze the sentiment of the text and provide a sentiment classification as 'positive', 'neutral', or 'negative'. Also provide a numerical score from -1.0 (most negative) to 1.0 (most positive). Return the results in JSON format with 'sentiment' and 'score' fields."
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "sentiment": result.get("sentiment", "neutral"),
            "score": result.get("score", 0.0)
        }
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return {"sentiment": "neutral", "score": 0.0}

def extract_metadata(text):
    """
    Extract metadata from the input text using OpenAI's API.
    
    Args:
        text (str): The text to extract metadata from
        
    Returns:
        dict: A dictionary containing metadata such as topics, regions, and commodities
    """
    try:
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a metadata extraction expert. Analyze the text and extract the following metadata:
                    1. Topics: Identify the main topics discussed (e.g., Economy, Politics, Technology, Environment, etc.). List 1-3 main topics.
                    2. Regions: Identify any countries, regions, or geographic areas mentioned. Use standard country/region names.
                    3. Commodities: Identify any commodities or resources mentioned (e.g., Oil, Gas, Gold, Wheat, etc.).
                    
                    Return the results in JSON format with 'topics', 'regions', and 'commodities' fields, each containing an array of strings.
                    """
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        return {
            "topics": result.get("topics", []),
            "regions": result.get("regions", []),
            "commodities": result.get("commodities", [])
        }
    except Exception as e:
        print(f"Error in extract_metadata: {e}")
        return {"topics": [], "regions": [], "commodities": []}
