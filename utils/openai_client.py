import os
import json
import csv
import base64
import io
import re
from openai import OpenAI

# Get API key from environment variable
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=OPENAI_API_KEY)

def determine_input_type(content):
    """
    Use GPT-o1 to determine the type of input provided.
    
    Args:
        content (str): The content to analyze
        
    Returns:
        dict: Information about the input type and confidence
    """
    try:
        # First 2000 characters should be enough to determine the type
        sample = content[:2000] if len(content) > 2000 else content
        
        prompt = f"""Analyze the following content and determine what type of input it is.
        
{sample}

Categorize this as one of: "article", "financial_report", "csv_data", "social_media", "news", "academic_paper", 
"product_review", "code", "legal_document", "general_text".

Also determine the general subject area (finance, politics, technology, etc.).

Format your response as JSON with these keys:
1. input_type: string (the category)
2. subject: string
3. confidence: number (0-1)
4. explanation: string (brief explanation for your classification)
"""
        
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert at identifying types of content."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=300
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        print(f"Error in determine_input_type: {e}")
        return {
            "input_type": "general_text",
            "subject": "unknown",
            "confidence": 0.5,
            "explanation": "Could not determine input type due to an error."
        }


def perform_detailed_analysis(content, input_type_info):
    """
    Use GPT-o1 (most powerful) to perform a detailed analysis of the content.
    
    Args:
        content (str): The content to analyze
        input_type_info (dict): Information about the input type
        
    Returns:
        str: Detailed analysis of the content
    """
    try:
        # Limit content length for API call
        max_length = 24000
        truncated_content = content[:max_length] if len(content) > max_length else content
        
        # Custom instructions based on input type
        type_specific_instructions = ""
        
        if input_type_info['input_type'] == "financial_report":
            type_specific_instructions = """
            - Identify key financial metrics and their trends
            - Analyze quarterly or annual performance
            - Note any significant changes in revenue, profit, or expenses
            - Identify company outlook and guidance
            - Extract any mentions of market conditions affecting the business
            """
        elif input_type_info['input_type'] == "news" or input_type_info['input_type'] == "article":
            type_specific_instructions = """
            - Identify the main events or developments reported
            - Note perspectives or biases in the reporting
            - Extract key persons, organizations, or entities involved
            - Identify geographical focus of the content
            - Note timing of events (current, historical, future predictions)
            """
        elif input_type_info['input_type'] == "csv_data":
            type_specific_instructions = """
            - Identify the structure and columns in the data
            - Analyze trends, patterns, or anomalies in the numerical data
            - Identify relationships between different variables
            - Provide statistical observations about the data
            - Note any time-based patterns if dates are present
            """
        elif input_type_info['input_type'] == "product_review":
            type_specific_instructions = """
            - Identify the product(s) being reviewed
            - Extract key positive and negative points
            - Note comparative statements with other products
            - Identify the reviewer's overall sentiment and recommendation
            - Extract any specific features or aspects mentioned
            """
        else:
            type_specific_instructions = """
            - Identify the main topics and themes
            - Note any arguments, evidence, or claims presented
            - Extract key entities, people, or organizations mentioned
            - Identify any conclusions or recommendations made
            - Note the overall purpose and intended audience
            """
        
        prompt = f"""Perform a comprehensive detailed analysis of the following content, 
which has been identified as: {input_type_info['input_type']} about {input_type_info['subject']}.

{truncated_content}

{type_specific_instructions}

Your analysis should be thorough and insightful, providing a detailed understanding of the content.
Focus on extracting meaningful insights rather than merely summarizing.
"""
        
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert analyst specializing in " + input_type_info['subject'] + ". Provide deep, insightful analysis."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=2000
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        print(f"Error in perform_detailed_analysis: {e}")
        return "Failed to generate detailed analysis. Please try again."


def analyze_sentiment(detailed_analysis):
    """
    Use GPT-4o to analyze sentiment from the detailed analysis.
    
    Args:
        detailed_analysis (str): The detailed analysis to evaluate
        
    Returns:
        dict: A dictionary containing sentiment analysis results
    """
    try:
        prompt = f"""Based on the following detailed analysis, provide a comprehensive sentiment assessment:

{detailed_analysis}

Classify the overall sentiment as ONLY one of these options: "positive", "negative", or "neutral".
Be definitive in your classification.

Provide your response in JSON format with these keys:
1. sentiment: string (positive, negative, or neutral)
2. score: number (-1 to 1, where -1 is very negative, 0 is neutral, 1 is very positive)
3. confidence: number (0-1)
4. rationale: string (brief explanation for your classification, 2-3 sentences max)
"""
        
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a sentiment analysis expert who provides clear, definitive sentiment classifications."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=500
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        return result
        
    except Exception as e:
        print(f"Error in analyze_sentiment: {e}")
        return {
            "sentiment": "neutral",
            "score": 0,
            "confidence": 0,
            "rationale": "Error in sentiment analysis"
        }


def extract_metadata(detailed_analysis):
    """
    Use GPT-o3-mini for efficient metadata extraction.
    
    Args:
        detailed_analysis (str): The detailed analysis to extract metadata from
        
    Returns:
        dict: A dictionary containing detailed metadata
    """
    try:
        prompt = f"""Extract structured metadata from this analysis:

{detailed_analysis}

Provide only the following categories:
1. topics: List of specific topics discussed (5 max)
2. regions: List of geographical regions mentioned (countries, cities, areas)
3. commodities: List of commodities or products mentioned (if applicable)
4. time_periods: List of time periods discussed (e.g., "Q4 2024", "1990s")
5. entities: List of key organizations or notable people mentioned

Provide ONLY these categories in JSON format. Each category should be a list of strings.
If no items for a category, use an empty array.
"""
        
        # The newest OpenAI model is "gpt-4o" which was released May 13, 2024.
        # Do not change this unless explicitly requested by the user
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a metadata extraction specialist focused on precision and conciseness."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            max_tokens=800
        )
        
        result = json.loads(response.choices[0].message.content.strip())
        
        # Process the result to ensure compatibility with existing code
        topics = result.get("topics", [])
        regions = result.get("regions", [])
        commodities = result.get("commodities", [])
        time_periods = result.get("time_periods", [])
        entities = result.get("entities", [])
        
        return {
            # Core metadata for backward compatibility
            "topics": topics,
            "regions": regions,
            "commodities": commodities,
            
            # Detailed metadata
            "topic_details": {
                "main_topics": topics[:3],  # First 3 are main topics
                "subtopics": topics[3:] if len(topics) > 3 else []
            },
            "geographical_details": {
                "countries": [r for r in regions if not any(term in r.lower() for term in ["region", "continent", "asia", "europe", "america"])],
                "regions": [r for r in regions if any(term in r.lower() for term in ["region", "continent", "asia", "europe", "america"])],
                "cities": []
            },
            "commodity_details": {
                "resources": commodities,
                "products": [],
                "financial_instruments": []
            },
            "temporal_details": {
                "time_period": time_periods,
                "key_dates": []
            },
            "entities": entities
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
            "temporal_details": {"time_period": [], "key_dates": []},
            "entities": []
        }
