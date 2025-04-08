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
        dict: A dictionary containing detailed sentiment analysis results
    """
    try:
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a sentiment analysis expert. Analyze the sentiment of the text deeply and provide:
                    
                    1. A sentiment classification as one of the following: 'positive', 'mostly_positive', 'neutral', 'mostly_negative', or 'negative'.
                    2. A numerical score from -1.0 (most negative) to 1.0 (most positive).
                    3. A brief rationale (2-3 sentences max) explaining the sentiment classification.
                    
                    Return the results in JSON format with 'sentiment', 'score', and 'rationale' fields.
                    """
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        # Ensure sentiment is one of the expected values, defaulting to neutral if not
        sentiment = result.get("sentiment", "neutral")
        valid_sentiments = ["positive", "mostly_positive", "neutral", "mostly_negative", "negative"]
        if sentiment not in valid_sentiments:
            # Map to standard sentiment if needed
            score = result.get("score", 0.0)
            if score >= 0.7:
                sentiment = "positive"
            elif score >= 0.3:
                sentiment = "mostly_positive"
            elif score <= -0.7:
                sentiment = "negative"
            elif score <= -0.3:
                sentiment = "mostly_negative"
            else:
                sentiment = "neutral"
        
        return {
            "sentiment": sentiment,
            "score": result.get("score", 0.0),
            "rationale": result.get("rationale", "")
        }
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return {"sentiment": "neutral", "score": 0.0, "rationale": "Error during sentiment analysis."}

def extract_metadata(text):
    """
    Extract detailed metadata from the input text using OpenAI's API.
    
    Args:
        text (str): The text to extract metadata from
        
    Returns:
        dict: A dictionary containing detailed metadata
    """
    try:
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = openai.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "system",
                    "content": """You are a metadata extraction expert. Thoroughly analyze the text and extract the following detailed metadata:
                    
                    1. Topics: Identify all significant topics discussed, categorized as:
                       - main_topics: 1-3 primary themes of the content (e.g., Economy, Politics, Technology)
                       - subtopics: More specific topics within the main themes (e.g., "Inflation" under Economy)
                    
                    2. Regions: 
                       - countries: Specific countries mentioned (use standard country names)
                       - regions: Broader geographical regions (e.g., "Middle East", "Southeast Asia")
                       - cities: Any significant cities mentioned
                    
                    3. Commodities: 
                       - resources: Raw materials or natural resources (e.g., Oil, Gas, Gold)
                       - products: Manufactured goods or specific products 
                       - financial_instruments: Any financial assets, securities or instruments mentioned
                    
                    4. Temporal:
                       - time_period: The time period discussed (e.g., "Current", "Historical", "Future projection")
                       - key_dates: Any specific dates or time periods mentioned that are significant
                    
                    Return the results in a detailed JSON format with all these fields, each containing an array of strings.
                    If no relevant information is found for a particular field, return an empty array.
                    """
                },
                {"role": "user", "content": text}
            ],
            response_format={"type": "json_object"}
        )
        
        result = json.loads(response.choices[0].message.content)
        
        # Extract topics hierarchy
        topics = result.get("main_topics", [])
        subtopics = result.get("subtopics", [])
        
        # Extract geographical information
        countries = result.get("countries", [])
        regions = result.get("regions", [])
        cities = result.get("cities", [])
        
        # Combine all geographical references for backward compatibility
        all_regions = list(set(countries + regions + cities))
        
        # Extract commodity information
        resources = result.get("resources", [])
        products = result.get("products", [])
        financial_instruments = result.get("financial_instruments", [])
        
        # Combine all commodity references for backward compatibility
        all_commodities = list(set(resources + products + financial_instruments))
        
        # Extract temporal information
        time_period = result.get("time_period", [])
        key_dates = result.get("key_dates", [])
        
        return {
            # Core metadata for backward compatibility
            "topics": topics,
            "regions": all_regions,
            "commodities": all_commodities,
            
            # Detailed metadata
            "topic_details": {
                "main_topics": topics,
                "subtopics": subtopics
            },
            "geographical_details": {
                "countries": countries,
                "regions": regions,
                "cities": cities
            },
            "commodity_details": {
                "resources": resources,
                "products": products,
                "financial_instruments": financial_instruments
            },
            "temporal_details": {
                "time_period": time_period,
                "key_dates": key_dates
            }
        }
    except Exception as e:
        print(f"Error in extract_metadata: {e}")
        return {
            "topics": [], 
            "regions": [], 
            "commodities": [],
            "topic_details": {"main_topics": [], "subtopics": []},
            "geographical_details": {"countries": [], "regions": [], "cities": []},
            "commodity_details": {"resources": [], "products": [], "financial_instruments": []},
            "temporal_details": {"time_period": [], "key_dates": []}
        }
