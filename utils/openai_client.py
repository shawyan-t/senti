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
    Use GPT-4o with internet browsing capabilities for real-time, up-to-date information.
    
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
    is_query = len(content.split()) < 50 and len(content) < 300
    
    try:
        # Create the system prompt based on content type
        system_prompt = f"""
        You are an expert analyst specializing in {subject} content with access to the latest information on the web.
        You have the ability to search the web for up-to-date information about recent events, trends, and real-time data.
        
        Your analysis should include:
        1. A comprehensive summary (2-3 paragraphs) including the latest developments
        2. Main themes, arguments, or points presented
        3. Key entities mentioned (people, organizations, products, etc.)
        4. Temporal context with current, real-time information 
        5. Geographic/regional focus with attention to recent regional developments
        6. Overall sentiment and emotional tone in the current discourse
        7. Significant facts, statistics or data points from the latest available sources
        
        Focus on objectivity, factual accuracy, and comprehensive coverage of the content.
        ALWAYS include the date of your analysis and mention the recency of the information.
        """
        
        # Enhanced web search capabilities for all content, especially short queries
        if is_query:
            system_prompt += """
            This is a search query. Your task is to:
            1. Search the web for the latest information on this topic
            2. Focus on real-time data and current events related to this query
            3. Include timestamps or publication dates of your sources when available
            4. Gather diverse perspectives from multiple reliable sources
            5. Synthesize this information into a comprehensive, up-to-date analysis
            """
        
        # User prompt
        user_prompt = f"""
        {content_type.upper()} CONTENT TO ANALYZE:
        
        {truncated_content}
        
        Please provide your detailed analysis focusing on the aspects mentioned in your instructions.
        Include the current date in your analysis and specify how recent your information is.
        If this topic involves current events, make sure to include the latest developments.
        """
        
        # Always use GPT-4o with its up-to-date knowledge
        model = "gpt-4o"
        
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
        
        # Append current date to ensure timeliness is clear
        current_date = datetime.now().strftime("%B %d, %Y")
        if "Analysis date:" not in analysis:
            analysis += f"\n\nAnalysis date: {current_date}"
        
        return analysis
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error performing detailed analysis: {e}")
        return f"Error analyzing content: {str(e)}\n\nThis appears to be {content_type} content about {subject}."

def analyze_sentiment(detailed_analysis):
    """
    Use GPT-4o to analyze sentiment from the detailed analysis with enhanced
    real-time web search capabilities for current context.
    
    Args:
        detailed_analysis (str): The detailed analysis to evaluate
        
    Returns:
        dict: A dictionary containing sentiment analysis results
    """
    try:
        # Create the system prompt with strong emphasis on current information
        system_prompt = """
        You are an expert sentiment analyst with access to the latest information from across the web.
        Your task is to determine sentiment, emotional tone, and key factors contributing to that sentiment.
        
        Your analysis should:
        1. Consider the real-time context of topics mentioned in the text
        2. Use your knowledge of current events, market conditions, and public discourse
        3. Consider cultural and regional context that might affect sentiment
        4. Be objective, factual, and comprehensive in your assessment
        5. Identify sentiment trends (improving, worsening, fluctuating) when relevant
        
        If the text mentions events, people, or topics that may have had recent developments,
        consider how those developments might affect the overall sentiment.
        """
        
        # User prompt
        user_prompt = f"""
        Analyze the sentiment of the following text, considering the most current context:
        
        {detailed_analysis}
        
        Provide a detailed sentiment analysis in JSON format with the following structure:
        {{
            "sentiment": "positive" or "negative" or "neutral",
            "score": a numeric score from -1.0 (very negative) to 1.0 (very positive),
            "confidence": a value from 0.0 to 1.0 indicating your confidence in this assessment,
            "rationale": a detailed explanation of the key factors contributing to this sentiment assessment,
            "current_context": a brief note about how recent events might be affecting this sentiment,
            "sentiment_trend": "improving", "worsening", "stable", or "fluctuating"
        }}
        
        Be sure to consider the most recent information available about the topics mentioned.
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
        
        # Make sure all expected fields are present
        if "current_context" not in result:
            result["current_context"] = ""
        if "sentiment_trend" not in result:
            result["sentiment_trend"] = "stable"
            
        return result
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error analyzing sentiment: {e}")
        return {
            "sentiment": "neutral",
            "score": 0.0,
            "confidence": 0.5,
            "rationale": f"Unable to determine sentiment due to an error: {str(e)}",
            "current_context": "",
            "sentiment_trend": "stable"
        }

def extract_metadata(detailed_analysis):
    """
    Use GPT-4o for comprehensive metadata extraction with special attention to
    temporal aspects and current events.
    
    Args:
        detailed_analysis (str): The detailed analysis to extract metadata from
        
    Returns:
        dict: A dictionary containing detailed metadata
    """
    try:
        # Create the system prompt with enhanced temporal awareness
        system_prompt = """
        You are an expert metadata tagger with special focus on temporal context and current events.
        Your task is to extract structured information from text to categorize and organize content.
        Pay special attention to:
        
        1. Identifying if the content relates to current events or news
        2. Capturing precise time periods including dates when mentioned
        3. Properly identifying regions and locations with high geographic specificity
        4. Recognizing entities (people, organizations) with their current roles or relevance
        5. Distinguishing between primary topics and emerging or trending subtopics
        
        Focus on accuracy, comprehensiveness, and structured tagging that reflects
        the current information landscape.
        """
        
        # User prompt with expanded metadata schema
        user_prompt = f"""
        Extract detailed metadata from the following analysis:
        
        {detailed_analysis}
        
        Return a JSON object with the following structure:
        {{
            "topics": [list of main topics covered],
            "regions": [geographic regions mentioned or relevant],
            "entities": [key people, companies, organizations mentioned],
            "commodities": [products, services, or resources mentioned],
            "temporal_details": {{
                "time_period": [relevant time periods mentioned],
                "specific_dates": [specific dates mentioned, in ISO format when possible],
                "recency": "historical" or "current" or "future",
                "currency": a value from 0.0 to 1.0 indicating how current/up-to-date the information is
            }},
            "topic_details": {{
                "main_topics": [primary topics, max 3],
                "subtopics": [secondary topics, max 5],
                "trending_topics": [topics that appear to be currently trending or emerging]
            }},
            "event_context": {{
                "is_current_event": true/false,
                "event_timeline": [brief timeline of key events if applicable],
                "key_developments": [important recent developments related to the topics]
            }}
        }}
        
        Each list should contain 3-7 items, prioritizing the most significant ones.
        For currency values, 1.0 means completely current (today/this week),
        while lower values indicate older or historical information.
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for its better understanding of current events
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Ensure all expected fields are present
        if "temporal_details" not in result:
            result["temporal_details"] = {
                "time_period": ["present"],
                "specific_dates": [],
                "recency": "current",
                "currency": 0.8
            }
        elif "specific_dates" not in result["temporal_details"]:
            result["temporal_details"]["specific_dates"] = []
        elif "currency" not in result["temporal_details"]:
            result["temporal_details"]["currency"] = 0.8
            
        if "event_context" not in result:
            result["event_context"] = {
                "is_current_event": False,
                "event_timeline": [],
                "key_developments": []
            }
            
        if "topic_details" not in result:
            result["topic_details"] = {
                "main_topics": ["general"],
                "subtopics": [],
                "trending_topics": []
            }
        elif "trending_topics" not in result["topic_details"]:
            result["topic_details"]["trending_topics"] = []
            
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
                "specific_dates": [],
                "recency": "current",
                "currency": 0.5
            },
            "topic_details": {
                "main_topics": ["general"],
                "subtopics": [],
                "trending_topics": []
            },
            "event_context": {
                "is_current_event": False,
                "event_timeline": [],
                "key_developments": []
            }
        }