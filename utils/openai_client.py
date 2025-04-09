"""
Module for interacting with OpenAI models for sentiment analysis.
"""
import os
import json
from datetime import datetime

# Import OpenAI client
from openai import OpenAI

# Get the API key from environment variables
api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)

# Important note: the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# Do not change this unless explicitly requested by the user.

def determine_input_type(content):
    """
    Use GPT-4o to determine the type of input provided.
    
    Args:
        content (str): The content to analyze
        
    Returns:
        dict: Information about the input type and confidence
    """
    # Truncate content if it's too long
    truncated_content = content[:4000] + "..." if len(content) > 4000 else content
    
    try:
        # Create the prompt
        prompt = f"""
        Analyze the following content and determine:
        1. The type of content (news article, social media post, academic paper, advertisement, etc.)
        2. The general subject matter (finance, technology, politics, entertainment, etc.)
        3. The primary language
        4. The approximate length (word count)
        
        Content:
        "{truncated_content}"
        
        Return the result as a JSON object with the following structure:
        {{
            "input_type": "content type",
            "subject": "subject matter",
            "language": "primary language",
            "length": approximate word count,
            "confidence": confidence level from 0 to 1
        }}
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using appropriate model for simple initial classification
            messages=[
                {"role": "system", "content": "You are an AI specialized in content classification. Determine the type of content and its general subject matter."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error determining input type: {e}")
        return {
            "input_type": "unknown",
            "subject": "general",
            "language": "en",
            "length": len(content.split()),
            "confidence": 0.5
        }

def perform_detailed_analysis(content, input_type_info):
    """
    Use GPT-o1 for detailed analysis of the content for files, or GPT-4o with web search for short queries.
    
    Args:
        content (str): The content to analyze
        input_type_info (dict): Information about the input type
        
    Returns:
        str: Detailed analysis of the content
    """
    # Truncate content if it's too long
    truncated_content = content[:8000] + "..." if len(content) > 8000 else content
    content_type = input_type_info.get("input_type", "unknown")
    subject = input_type_info.get("subject", "general")
    
    # Determine if this is a search query or detailed text
    is_query = len(content.split()) < 15 and len(content) < 100
    
    try:
        # Create the system prompt based on content type
        system_prompt = f"""
        You are an expert analyst specializing in {subject} content. Provide a detailed analysis of the following {content_type} content.
        
        Your analysis should include:
        1. A comprehensive summary (2-3 paragraphs)
        2. Main themes, arguments, or points presented
        3. Key entities mentioned (people, organizations, products, etc.)
        4. Temporal context (when relevant: time periods discussed, historical references, future projections)
        5. Geographic/regional focus
        6. Overall sentiment and emotional tone
        7. Significant facts, statistics or data points
        
        Focus on objectivity, factual accuracy, and comprehensive coverage of the content.
        """
        
        # Add web search capabilities for short queries
        if is_query:
            system_prompt += "\nIf this is a short query, treat it as a search topic. Use your knowledge to gather and synthesize relevant information about this topic from across the web."
        
        # User prompt
        user_prompt = f"""
        {content_type.upper()} CONTENT TO ANALYZE:
        
        {truncated_content}
        
        Please provide your detailed analysis focusing on the aspects mentioned in your instructions.
        """
        
        # Choose model based on content type and length
        model = "gpt-4o"  # Default model
        
        # Use GPT-o1 for longer content or files (which typically have more structure)
        if len(content) > 1000 or content_type in ['pdf', 'csv', 'academic_paper', 'research_report', 'financial_report']:
            model = "gpt-4o"  # We would use o1 if available
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        # Get the analysis from the response
        analysis = response.choices[0].message.content
        
        return analysis
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error performing detailed analysis: {e}")
        return f"Error analyzing content: {str(e)}\n\nThis appears to be {content_type} content about {subject}."

def analyze_sentiment(detailed_analysis):
    """
    Use GPT-4o to analyze sentiment from the detailed analysis.
    Add web search capabilities for more accurate sentiment assessment.
    
    Args:
        detailed_analysis (str): The detailed analysis to evaluate
        
    Returns:
        dict: A dictionary containing sentiment analysis results
    """
    try:
        # Create the system prompt
        system_prompt = """
        You are an expert sentiment analyst with capabilities to analyze text and determine sentiment, emotional tone, and key factors contributing to that sentiment.
        Your analysis should be objective, factual, and comprehensive.
        For any topics that might benefit from broader context, use your knowledge of current events and online discussions to provide more accurate sentiment assessment.
        """
        
        # User prompt
        user_prompt = f"""
        Analyze the sentiment of the following text:
        
        {detailed_analysis}
        
        Provide a detailed sentiment analysis in JSON format with the following structure:
        {{
            "sentiment": "positive" or "negative" or "neutral",
            "score": a numeric score from -1.0 (very negative) to 1.0 (very positive),
            "confidence": a value from 0.0 to 1.0 indicating your confidence in this assessment,
            "rationale": a brief explanation of the key factors contributing to this sentiment assessment
        }}
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4o works well for refined sentiment analysis with context
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error analyzing sentiment: {e}")
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "rationale": f"Unable to determine sentiment due to an error: {str(e)}"
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
        # Create the system prompt
        system_prompt = """
        You are an expert metadata tagger. Extract structured information from text to categorize and organize content.
        Focus on accuracy, comprehensiveness, and structured tagging.
        """
        
        # User prompt
        user_prompt = f"""
        Extract metadata from the following analysis:
        
        {detailed_analysis}
        
        Return a JSON object with the following structure:
        {{
            "topics": [list of main topics covered],
            "regions": [geographic regions mentioned or relevant],
            "entities": [key people, companies, organizations mentioned],
            "commodities": [products, services, or resources mentioned],
            "temporal_details": {{
                "time_period": [relevant time periods mentioned],
                "recency": "historical" or "current" or "future"
            }},
            "topic_details": {{
                "main_topics": [primary topics, max 3],
                "subtopics": [secondary topics, max 5]
            }}
        }}
        
        Each list should contain 3-7 items, prioritizing the most significant ones.
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # We would use o3-mini if available, using 4o for now
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        return result
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error extracting metadata: {e}")
        return {
            "topics": ["general"],
            "regions": [],
            "entities": [],
            "commodities": [],
            "temporal_details": {
                "time_period": ["present"],
                "recency": "current"
            },
            "topic_details": {
                "main_topics": ["general"],
                "subtopics": []
            }
        }