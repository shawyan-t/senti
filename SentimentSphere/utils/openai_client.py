"""
Module for interacting with OpenAI models for sentiment analysis.
"""
import json
from datetime import datetime
from openai import OpenAI
from .config import config

# Initialize OpenAI client with API key from config
client = OpenAI(api_key=config['openai_api_key'])

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
    Use GPT-4o with enhanced prompting for real-time, up-to-date information,
    particularly for pop culture, current events, and trending topics.
    
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
        You are an expert analyst specializing in {subject} content with comprehensive, up-to-date knowledge.
        You excel at analyzing content related to pop culture, music, entertainment, sports, politics, 
        technology, and current events. 
        
        Your strengths include:
        1. Recognizing recent celebrities, artists, athletes, and public figures
        2. Understanding slang, internet culture, and trending topics
        3. Being aware of recent album releases, movies, sports events, and viral content
        4. Keeping track of global news events and developments

        Your analysis should include:
        1. A comprehensive summary (2-3 paragraphs) including latest developments
        2. Proper identification of people, places, and entities (especially newer/trending ones)
        3. Cultural context, especially for pop culture, music, sports, and entertainment
        4. Temporal context with up-to-date information
        5. Geographic/regional relevance
        6. Overall sentiment and emotional tone

        IMPORTANT: If the topic appears to be about a person, brand, event, or cultural reference:
        - Explicitly identify if it's related to music, sports, entertainment, politics, etc.
        - Provide context about who they are or what it refers to
        - Mention recency (e.g., "recently released album", "viral trend from May 2024", etc.)
        """
        
        # Enhanced capabilities for search queries
        if is_query:
            system_prompt += """
            This appears to be a search query or short topic reference. Your priorities are:
            1. Correctly identify what/who this query refers to (especially if it's a person, event, or cultural reference)
            2. Provide context about its significance and category (music, sports, politics, entertainment, etc.)
            3. Determine if this is a trending or recent topic and provide temporal context
            4. If this could refer to multiple things, acknowledge the ambiguity and address the most likely/recent reference first
            5. For music artists, athletes, celebrities, mention their most recent notable works or achievements
            
            Example analysis for "Swamp Izzo":
            "Swamp Izzo is an Atlanta-based music producer and DJ who gained significant attention in 2024 for his work with Drake on the album 'For All The Dogs'. He's known for his trap production style and has worked with other notable hip-hop artists including Future and 21 Savage. The query likely refers to his recent production credits or his growing prominence in the hip-hop industry."
            """
        
        # User prompt
        user_prompt = f"""
        {content_type.upper()} CONTENT TO ANALYZE:
        
        {truncated_content}
        
        Please provide your detailed analysis with special attention to correctly identifying any people, 
        brands, events, or cultural references. If this is related to pop culture, music, sports, or 
        entertainment, be sure to provide specific context about who/what it refers to and its relevance.
        
        Current date: {datetime.now().strftime("%B %d, %Y")}
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
    understanding of context, nuance, and cultural/temporal factors.
    
    Args:
        detailed_analysis (str): The detailed analysis to evaluate
        
    Returns:
        dict: A dictionary containing sentiment analysis results
    """
    try:
        # Check if there's real-time search data included
        has_realtime_data = "REAL-TIME SEARCH DATA" in detailed_analysis
        
        # Create the system prompt with strong emphasis on contextual understanding
        system_prompt = """
        You are an expert sentiment analyst with exceptional understanding of context and nuance.
        Your goal is to accurately determine sentiment while considering cultural, temporal, and
        subject-specific factors that might influence how sentiment should be interpreted.
        
        Your analysis must:
        1. Consider cultural context - what might be positive in one culture could be negative in another
        2. Consider temporal context - current events, recent developments that influence sentiment
        3. Consider domain-specific context - different expectations in finance vs entertainment vs politics
        4. Identify sentiment trends with concrete justification (improving, worsening, fluctuating)
        5. Recognize ambiguity and mixed sentiment - not everything is simply positive or negative
        6. Provide a confidence score that honestly reflects uncertainty when present
        
        For trending topics, celebrities, or current events:
        - Recognize that sentiment around public figures can be complex and multifaceted
        - Consider both public perception and factual achievements/controversies
        - Distinguish between artistic/professional evaluation and personal controversies
        - Provide nuanced rationale that acknowledges complexity
        
        IMPORTANT: When the text includes REAL-TIME SEARCH DATA:
        - Prioritize this current information over general knowledge
        - Base your sentiment analysis heavily on this real-time data
        - Explicitly reference the source information in your rationale
        - Be specific about how recent developments affect the sentiment
        - Acknowledge if real-time data presents a different sentiment than what might be expected
        
        Your sentiment assessment should be highly accurate, culturally informed, and contextually appropriate,
        with strong emphasis on real-time information when available.
        """
        
        # User prompt - adjust based on whether real-time data is present
        if has_realtime_data:
            user_prompt = f"""
            Analyze the sentiment of the following text that includes REAL-TIME SEARCH DATA.
            Pay special attention to the real-time search information and how it impacts the current sentiment.
            
            {detailed_analysis}
            
            IMPORTANT: Base your analysis primarily on the REAL-TIME SEARCH DATA section, as this contains 
            the most current information. The rest of the text provides additional context.
            
            Provide your sentiment analysis in JSON format with the following structure:
            {{
                "sentiment": "positive" or "negative" or "neutral" or "mixed",
                "score": a numeric score from -1.0 (very negative) to 1.0 (very positive),
                "confidence": a value from 0.0 to 1.0 indicating your confidence in this assessment,
                "rationale": a detailed explanation of the key factors contributing to this sentiment assessment,
                "current_context": a specific note about what the real-time data shows about current sentiment,
                "sentiment_trend": "improving", "worsening", "stable", or "fluctuating",
                "real_time_sources": what sources provided the most relevant real-time information,
                "sentiment_factors": [
                    {{
                        "factor": "name of sentiment factor", 
                        "impact": "positive" or "negative" or "neutral",
                        "weight": a value from 0.0 to 1.0 indicating the relative importance
                    }}
                ]
            }}
            
            Be sure to provide a nuanced assessment that acknowledges complexity when present,
            and explicitly reference the real-time data in your analysis.
            """
        else:
            # Standard prompt without real-time data emphasis
            user_prompt = f"""
            Analyze the sentiment of the following text with careful attention to cultural, temporal and domain-specific context:
            
            {detailed_analysis}
            
            Provide your sentiment analysis in JSON format with the following structure:
            {{
                "sentiment": "positive" or "negative" or "neutral" or "mixed",
                "score": a numeric score from -1.0 (very negative) to 1.0 (very positive),
                "confidence": a value from 0.0 to 1.0 indicating your confidence in this assessment,
                "rationale": a detailed explanation of the key factors contributing to this sentiment assessment,
                "current_context": a specific note about how recent events or cultural context affects this sentiment,
                "sentiment_trend": "improving", "worsening", "stable", or "fluctuating",
                "sentiment_factors": [
                    {{
                        "factor": "name of sentiment factor", 
                        "impact": "positive" or "negative" or "neutral",
                        "weight": a value from 0.0 to 1.0 indicating the relative importance
                    }}
                ]
            }}
            
            Be sure to provide a nuanced assessment that acknowledges complexity when present.
            For topics related to people, events, or cultural references, consider both general perception
            and specific recent developments that might influence sentiment.
            """
        
        # Call the OpenAI API with very high temperature for more nuanced analysis
        response = client.chat.completions.create(
            model="gpt-4o",  # GPT-4o has the most current knowledge
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.3  # Slightly increased temperature for more nuanced analysis
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Make sure all expected fields are present
        if "current_context" not in result:
            result["current_context"] = ""
        if "sentiment_trend" not in result:
            result["sentiment_trend"] = "stable"
        if "sentiment_factors" not in result:
            result["sentiment_factors"] = []
        if has_realtime_data and "real_time_sources" not in result:
            result["real_time_sources"] = "Real-time search data"
            
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
            "sentiment_trend": "stable",
            "sentiment_factors": [],
            "real_time_sources": "Real-time search data" if has_realtime_data else ""
        }

def extract_metadata(detailed_analysis):
    """
    Use GPT-4o for comprehensive metadata extraction with enhanced attention to
    current events, pop culture, trending topics, and temporal relevance.
    
    Args:
        detailed_analysis (str): The detailed analysis to evaluate
        
    Returns:
        dict: A dictionary containing extracted metadata
    """
    try:
        # Create the system prompt
        system_prompt = """
        You are an expert metadata extractor with deep knowledge of current events, trending topics,
        pop culture, entertainment, sports, politics, finance and technology.
        
        Your task is to extract comprehensive, accurate metadata from text with special attention to:
        1. Correctly identifying people, especially current celebrities, artists, athletes, and public figures
        2. Identifying trending topics and cultural phenomena
        3. Recognizing references to recent events, releases, or publications
        4. Accurately categorizing subject matter
        5. Extracting geographic and temporal information
        
        When extracting metadata about trending topics, celebrities, entertainment, or current events:
        - Ensure you have the correct category (music, sports, film, politics, etc.)
        - Provide specific details rather than general categories
        - For people, include their profession/role and recent notable works/events
        - For events, include temporal context (when it occurred/is occurring)
        - For creative works, include creators and release information
        
        Maintain high accuracy and specificity in your metadata extraction.
        """
        
        # User prompt
        user_prompt = f"""
        Extract comprehensive metadata from the following text:
        
        {detailed_analysis}
        
        Return the metadata in JSON format with the following structure:
        {{
            "topics": list of main topics discussed,
            "topic_details": {{
                "main_topics": list of primary topics,
                "subtopics": list of related subtopics,
                "trending_topics": list of topics currently trending (if any),
                "category": primary category of the content (e.g., "politics", "entertainment", "sports", "technology", "finance", "pop culture")
            }},
            "entities": list of key entities (people, organizations, brands, products),
            "entity_details": {{
                "people": [
                    {{
                        "name": "Person's name",
                        "role": "Their profession/role",
                        "relevance": "Why they're mentioned",
                        "recent_works": "Any recent notable works/events" (if applicable)
                    }}
                ],
                "organizations": list of organizations with details,
                "brands": list of brands with details,
                "products": list of products with details
            }},
            "regions": list of geographic regions mentioned,
            "region_details": {{
                "countries": list of countries,
                "cities": list of cities,
                "regions": list of regions
            }},
            "temporal_details": {{
                "time_period": general time frame referenced,
                "specific_dates": list of specific dates mentioned,
                "recency": "historical", "recent", "current", or "future"
            }},
            "event_context": {{
                "is_current_event": true/false,
                "event_type": type of event (if applicable),
                "key_developments": list of key developments,
                "event_timeline": timeline of events (if applicable)
            }},
            "commodities": list of commodities or products mentioned
        }}
        
        Be as comprehensive and accurate as possible, but do not invent details that aren't clearly indicated in the text.
        For pop culture, entertainment, and trending topics, ensure you provide specific details rather than general categories.
        """
        
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",  # Using GPT-4o for most up-to-date context awareness
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse the response
        result = json.loads(response.choices[0].message.content)
        
        # Make sure all expected top-level keys are present
        expected_keys = ["topics", "topic_details", "entities", "entity_details", "regions", 
                         "region_details", "temporal_details", "event_context", "commodities"]
        
        for key in expected_keys:
            if key not in result:
                if key in ["topics", "entities", "regions", "commodities"]:
                    result[key] = []
                else:
                    result[key] = {}
        
        return result
    
    except Exception as e:
        # Return a fallback if there's an error
        print(f"Error extracting metadata: {e}")
        return {
            "topics": [],
            "topic_details": {
                "main_topics": [],
                "subtopics": [],
                "trending_topics": [],
                "category": "general"
            },
            "entities": [],
            "entity_details": {
                "people": [],
                "organizations": [],
                "brands": [],
                "products": []
            },
            "regions": [],
            "region_details": {
                "countries": [],
                "cities": [],
                "regions": []
            },
            "temporal_details": {
                "time_period": ["present"],
                "specific_dates": [],
                "recency": "current"
            },
            "event_context": {
                "is_current_event": False,
                "event_type": "",
                "key_developments": [],
                "event_timeline": []
            },
            "commodities": []
        }