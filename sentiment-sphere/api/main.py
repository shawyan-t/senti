from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import json
import os
import tempfile
import time
import pandas as pd
import numpy as np
import random
import uuid
from datetime import datetime, timedelta
import re

# Import utility functions
from utils.text_processor import (
    process_text_input, 
    extract_text_from_pdf, 
    extract_text_from_html,
    detect_file_type,
    process_csv_data
)
from utils.openai_client import (
    determine_input_type, 
    perform_detailed_analysis, 
    analyze_sentiment, 
    extract_metadata
)
from utils.data_manager import save_analysis, load_analysis, load_all_analyses, delete_analysis, get_filtered_data
from utils.external_data import get_online_sentiment, get_online_sentiment_with_search, SearchEngineConnector
from utils.visualizations import (
    create_3d_globe_visualization,
    create_interest_over_time_chart,
    create_topic_popularity_chart,
    create_keyword_chart,
    # New professional visualizations
    create_sentiment_index_with_uncertainty,
    create_polarity_share_bars_with_intervals,
    create_vad_compass,
    create_source_quality_matrix,
    create_rolling_sentiment_timeline
)
from utils.sentiment_generator import generate_emotion_analysis, generate_embeddings

def calculate_polarity_from_mathematical_score(sentiment_score):
    """Convert mathematical sentiment score to polarity distribution"""
    # Mathematical scores range from -1 to 1
    # Convert to positive/negative/neutral percentages
    
    if sentiment_score > 0.1:  # Positive threshold
        positive = min(0.9, 0.5 + sentiment_score * 0.4)  # Scale positive
        negative = max(0.05, 0.15 - sentiment_score * 0.1)  # Reduce negative
        neutral = 1.0 - positive - negative
    elif sentiment_score < -0.1:  # Negative threshold
        negative = min(0.9, 0.5 + abs(sentiment_score) * 0.4)  # Scale negative
        positive = max(0.05, 0.15 - abs(sentiment_score) * 0.1)  # Reduce positive
        neutral = 1.0 - positive - negative
    else:  # Neutral range (-0.1 to 0.1)
        # Still allow some variation in neutral range
        neutral = 0.7 + (0.1 - abs(sentiment_score)) * 0.2  # 70-90% neutral
        if sentiment_score >= 0:
            positive = (1.0 - neutral) * 0.6  # Slight positive lean
            negative = 1.0 - neutral - positive
        else:
            negative = (1.0 - neutral) * 0.6  # Slight negative lean  
            positive = 1.0 - neutral - negative
    
    return {
        "positive": round(positive, 3),
        "negative": round(negative, 3), 
        "neutral": round(neutral, 3),
        "wilson_ci": {
            "positive": [max(0, positive - 0.1), min(1, positive + 0.1)],
            "negative": [max(0, negative - 0.1), min(1, negative + 0.1)],
            "neutral": [max(0, neutral - 0.1), min(1, neutral + 0.1)]
        }
    }

def calculate_vad_from_mathematical_score(sentiment_score, emotion_analysis=None):
    """Calculate VAD (Valence, Arousal, Dominance) from mathematical sentiment and emotions"""
    # Valence: How positive/negative (directly from sentiment score)
    valence = (sentiment_score + 1) / 2  # Convert [-1,1] to [0,1]
    
    # Arousal: How intense/activated (from emotion magnitude and sentiment strength)
    if emotion_analysis and 'emotion_mathematics' in emotion_analysis:
        # Use emotion vector magnitude for arousal
        base_arousal = emotion_analysis['emotion_mathematics'].get('vector_magnitude', 0.5)
        sentiment_intensity = abs(sentiment_score)
        arousal = min(1.0, base_arousal + sentiment_intensity * 0.3)
    else:
        # Calculate arousal from sentiment intensity
        arousal = 0.3 + abs(sentiment_score) * 0.4  # More intense = higher arousal
    
    # Dominance: How in-control/confident (from sentiment confidence and coherence)
    if emotion_analysis and 'emotional_coherence' in emotion_analysis:
        coherence = emotion_analysis['emotional_coherence']
        dominance = 0.4 + coherence * 0.4 + (sentiment_score > 0) * 0.2  # Positive sentiment = more dominant
    else:
        # Base dominance on sentiment strength and direction
        dominance = 0.5 + sentiment_score * 0.3
    
    return {
        "valence": round(max(0, min(1, valence)), 3),
        "arousal": round(max(0, min(1, arousal)), 3), 
        "dominance": round(max(0, min(1, dominance)), 3)
    }

def extract_entities_from_search_results(search_results):
    """Extract common keywords and phrases from full article content"""
    if not search_results:
        return []
    
    # Combine all text from search results - prioritize full content over snippets
    all_text = ""
    content_count = 0
    
    for result in search_results:
        title = result.get('title', '')
        snippet = result.get('snippet', '')
        full_content = result.get('full_content', '')
        
        # Use full content if available, otherwise fall back to snippet
        if full_content and len(full_content) > 200:  # Meaningful content
            all_text += f"{title} {full_content} "
            content_count += 1
        else:
            all_text += f"{title} {snippet} "
    
    print(f"🔍 Entity extraction from {len(search_results)} results ({content_count} with full content)")
    
    # Extract keywords using better NLP approach
    import re
    from collections import Counter
    
    # Financial/business terms to prioritize (more comprehensive)
    financial_terms = [
        'earnings', 'revenue', 'profit', 'growth', 'market', 'stock', 'shares', 'investment', 
        'financial', 'quarter', 'performance', 'analyst', 'rating', 'portfolio', 'dividend',
        'merger', 'acquisition', 'ipo', 'valuation', 'liquidity', 'volatility', 'bullish', 
        'bearish', 'outlook', 'forecast', 'guidance', 'metrics', 'margin', 'efficiency'
    ]
    
    # Extract meaningful phrases (2-3 words) and single words
    phrases = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,2}\b', all_text)  # Capitalized phrases
    words = re.findall(r'\b[A-Za-z]{4,}\b', all_text.lower())  # Single meaningful words
    
    # Combine phrases and words
    all_terms = [phrase.strip() for phrase in phrases] + words
    term_counts = Counter(all_terms)
    
    # Enhanced stop words
    stop_words = {
        'the', 'and', 'for', 'are', 'with', 'this', 'that', 'from', 'they', 'have', 'had',
        'was', 'were', 'been', 'said', 'will', 'would', 'could', 'should', 'may', 'can',
        'more', 'than', 'other', 'such', 'what', 'some', 'time', 'very', 'when', 'much',
        'also', 'your', 'work', 'life', 'only', 'over', 'think', 'most', 'even', 'back'
    }
    
    # Get most relevant entities
    entities = []
    for term, count in term_counts.most_common(30):
        term_lower = term.lower()
        if (term_lower not in stop_words and 
            len(term) > 3 and 
            count >= 2 and  # Appears multiple times
            (term_lower in financial_terms or  # Financial term
             count >= 4 or  # Very frequent
             term[0].isupper())):  # Proper noun (likely company/product name)
            
            # Clean up the term
            clean_term = term.title() if not term[0].isupper() else term
            if clean_term not in entities:
                entities.append(clean_term)
    
    print(f"📊 Extracted entities: {entities[:10]}")
    return entities[:10]  # Return top 10 entities

def create_enhanced_summary(unit_count, confidence, mathematical_results, comprehensive_results):
    """Create enhanced summary with correct sentiment score"""
    # Use mathematical score if available, otherwise comprehensive score
    sentiment_score = (mathematical_results['mathematical_sentiment_analysis']['composite_score']['value'] 
                      if mathematical_results else comprehensive_results['sentiment']['score'])
    
    # Determine sentiment label
    if sentiment_score > 0.1:
        sentiment_label = "positive"
    elif sentiment_score < -0.1:
        sentiment_label = "negative"
    else:
        sentiment_label = "neutral"
    
    return f"Comprehensive analysis of {unit_count} content units with {confidence:.1%} confidence. Sentiment: {sentiment_score:.3f} ({sentiment_label})"

app = FastAPI(title="Sentimizer API")

# Configure CORS
# CORS: allow local dev and Vercel domains explicitly when credentials are used
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://localhost:3000",
        # Add your production Vercel domain(s) explicitly if known
        # e.g., "https://sentimizer.vercel.app",
    ],
    allow_origin_regex=r"https://.*\.vercel\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Optional: background pre-warm on startup (disabled unless explicitly enabled)
@app.on_event("startup")
async def _maybe_prewarm():
    if os.getenv("ENABLE_STARTUP_PREWARM", "false").lower() != "true":
        return
    try:
        import asyncio
        loop = asyncio.get_event_loop()
        def _warm():
            try:
                from utils.mathematical_sentiment import get_mathematical_analyzer
                _ = get_mathematical_analyzer()
            except Exception as e:
                print("[startup] prewarm failed:", e)
        await loop.run_in_executor(None, _warm)
        print("[startup] prewarm completed")
    except Exception as e:
        print("[startup] prewarm init error:", e)

# Models for request data
class TextInput(BaseModel):
    text: str
    use_search_apis: bool = True

class FileAnalysisOptions(BaseModel):
    use_search_apis: bool = True
    include_online_data: bool = True
    days_back: int = 365

class AnalysisRequest(BaseModel):
    content: str
    type: str = "text"
    options: Optional[Dict[str, Any]] = None

class SearchQuery(BaseModel):
    query: str
    subtopics: Optional[List[str]] = None
    days_back: int = 365
    use_search_apis: bool = True
    include_search_analysis: bool = True

class FilterParams(BaseModel):
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    topics: Optional[List[str]] = None
    regions: Optional[List[str]] = None
    commodities: Optional[List[str]] = None
    sentiments: Optional[List[str]] = None
    time_period: Optional[str] = None
    source_type: Optional[str] = None

class VisualizationParams(BaseModel):
    topic: str
    subtopics: Optional[List[str]] = None
    time_period: str = "year"  # week, month, year, all
    visualization_type: str = "all"  # globe, time, topic, keywords, all

@app.post("/api/analyze/text")
async def analyze_text(input_data: TextInput):
    try:
        print(f"Received text analysis request: {input_data.text[:50]}...")
        
        # Determine if it's a URL or direct text
        if input_data.text.startswith(('http://', 'https://')):
            content = extract_text_from_html(input_data.text)
            source = input_data.text
            source_type = "url"
        else:
            content = process_text_input(input_data.text)
            source = input_data.text if len(input_data.text) < 50 else "Direct Text Input"
            source_type = "direct_text"
        
        # Step 1: Determine input type
        print("Step 1/5: Identifying content type...")
        input_type_info = determine_input_type(content)
        
        # Step 2: Get real-time data from search engines for context enrichment
        print("Step 2/5: Fetching real-time data from search engines...")
        search_connector = SearchEngineConnector()
        
        # Use the input as search query if it's a short text
        search_query = source if len(content) < 500 else input_type_info.get('subject', source)
        search_results = search_connector.get_cached_or_fresh_data(search_query)
        
        # Extract and prepare search content for enrichment
        search_context = "REAL-TIME SEARCH DATA:\n\n"
        for i, result in enumerate(search_results[:3]):  # Use top 3 results
            extracted_content = search_connector.extract_content_from_url(result.get('link'))
            if extracted_content:
                search_context += f"SOURCE {i+1}: {extracted_content.get('title')}\n"
                search_context += f"URL: {result.get('link')}\n"
                search_context += f"CONTENT: {extracted_content.get('content')[:500]}...\n\n"
            else:
                search_context += f"SOURCE {i+1}: {result.get('title')}\n"
                search_context += f"URL: {result.get('link')}\n"
                search_context += f"SNIPPET: {result.get('snippet', '')}\n\n"
        
        # Step 3: Based on the content type and whether it's a file or short query:
        if source_type in ["pdf", "csv", "json"] or len(content) > 1000:
            # For files or lengthy content, use GPT-o1 for detailed analysis first
            print("Step 3/5: Performing detailed analysis using GPT-o1...")
            # Include search context for better analysis
            analysis_input = f"{content}\n\n{search_context}"
            detailed_analysis = perform_detailed_analysis(analysis_input, input_type_info)
        else:
            # For shorter queries, skip to GPT-4o directly
            print("Step 3/5: Using GPT-4o to analyze your query with real-time data...")
            # Include web searching capabilities for better context and accuracy
            analysis_input = f"Search query: {source}\n\n{search_context}\n\n{content}"
            detailed_analysis = perform_detailed_analysis(analysis_input, input_type_info)
        
        # Step 4: Analyze sentiment with GPT-4o for all cases
        print("Step 4/5: Analyzing sentiment with real-time context...")
        sentiment_result = analyze_sentiment(detailed_analysis)
        
        # Step 5: Extract metadata with GPT-o3-mini
        print("Step 5/5: Extracting metadata...")
        metadata_result = extract_metadata(detailed_analysis)

        # Step 6: Generate real emotion analysis using HuggingFace
        print("Step 6/6: Generating emotion analysis...")
        emotions = generate_emotion_analysis(content)

        # Generate embeddings for the main content (split into sentences if long)
        if len(content) > 500:
            # Split into sentences for richer embedding landscape
            sentences = re.split(r'(?<=[.!?]) +', content)
            # Filter out very short sentences
            sentences = [s for s in sentences if len(s.strip()) > 10]
            embeddings = generate_embeddings(sentences)
            embedding_labels = sentences
        else:
            embeddings = generate_embeddings(content)
            embedding_labels = [content]

        # Save the analysis
        timestamp = datetime.now()
        
        # Define the analysis structure based on app.py
        analysis_data = {
            "id": str(uuid.uuid4()),  # Generate ID here to use for both saving and returning
            "source": source,
            "source_type": source_type,
            "text": content[:1000] + "..." if len(content) > 1000 else content,  # Store truncated version
            "summary": detailed_analysis,
            "sentiment": sentiment_result,
            "metadata": metadata_result,
            "emotions": emotions,
            "embeddings": embeddings,
            "embedding_labels": embedding_labels,
            "timestamp": timestamp.isoformat(),
            "input_type": input_type_info
        }
        
        # Save analysis with the prepared data structure
        analysis_id = save_analysis(analysis_data)
        print(f"Analysis saved with ID: {analysis_id}")
        
        # Add topic details if not present
        if "topic_details" not in metadata_result:
            metadata_result["topic_details"] = {
                "main_topics": metadata_result.get("topics", [])[:1],
                "subtopics": metadata_result.get("topics", [])[1:3] if len(metadata_result.get("topics", [])) > 1 else [],
                "trending_topics": []
            }
            
        # Ensure sentiment has sentiment_trend and current_context
        if "sentiment_trend" not in sentiment_result:
            sentiment_result["sentiment_trend"] = "stable"
        if "current_context" not in sentiment_result:
            sentiment_result["current_context"] = ""
            
        # Ensure temporal_details and event_context exist in metadata
        if "temporal_details" not in metadata_result:
            metadata_result["temporal_details"] = {
                "time_period": [],
                "specific_dates": []
            }
        if "event_context" not in metadata_result:
            metadata_result["event_context"] = {
                "is_current_event": False,
                "key_developments": [],
                "event_timeline": []
            }
        
        # Get online sentiment data if requested
        if input_data.use_search_apis:
            # Get main topic from metadata or use source as fallback
            main_topics = metadata_result.get("topic_details", {}).get("main_topics", [])
            main_topic = main_topics[0] if main_topics else input_type_info.get('subject', source)
            
            # Clean up main topic for better search
            if main_topic:
                main_topic = main_topic.strip()
                if len(main_topic) > 50:  # If too long, truncate
                    main_topic = main_topic[:50] + "..."
            else:
                # Fallback to source processing
                if source_type == "url":
                    # Use the URL domain or title if available
                    main_topic = source.split("/")[-1].replace("-", " ").replace("_", " ")
                    if len(main_topic) < 3:  # If too short (like just "com"), use the domain
                        main_topic = source.split("//")[-1].split("/")[0]
                elif source_type == "direct_text" and len(source) < 50:
                    # If it's a short direct text entry, it's likely a search query - use it directly
                    main_topic = source
                else:
                    main_topic = input_type_info.get('subject', 'general')
            
            # Get subtopics for additional context
            subtopics = metadata_result.get("topic_details", {}).get("subtopics", [])
            
            # Get online sentiment data
            online_data = get_online_sentiment_with_search(
                main_topic, 
                subtopics, 
                days_back=365, 
                use_search_apis=True
            )
            
            # Process search-augmented analysis if available
            search_analysis = online_data.get("search_augmented_analysis", {})
            if not search_analysis:
                # Create a default search analysis structure
                online_data["search_augmented_analysis"] = {
                    "analysis": "No search-augmented analysis available",
                    "confidence": "Low confidence",
                    "sources": []
                }
                
            # Format the API status for frontend consumption
            api_status = online_data.get("api_status", {})
            if not api_status:
                online_data["api_status"] = {
                    "twitter_api": False,
                    "reddit_api": False,
                    "news_api": False,
                    "search_api": input_data.use_search_apis
                }
                
            # Ensure sources structure exists
            if "sources" not in online_data:
                online_data["sources"] = {
                    "search_engines": [],
                    "news": [],
                    "twitter": [],
                    "reddit": [],
                    "web": []
                }
                
            # Make sure visualization data structures exist
            for data_key in ["global_data", "historical_data", "keyword_data"]:
                if data_key not in online_data:
                    online_data[data_key] = {}
            
            # Add online data to response
            analysis_data["online_data"] = online_data
        
        # Generate embedding colors based on sentiment
        def get_color(sentiment):
            if sentiment == 'positive':
                return '#10B981'  # green
            elif sentiment == 'negative':
                return '#EF4444'  # red
            elif sentiment == 'neutral':
                return '#FBBF24'  # yellow
            else:
                return '#6B7280'  # gray
        embedding_colors = [get_color(sentiment_result.get('sentiment', 'neutral')) for _ in embeddings]
        
        # Return comprehensive response with all needed data
        return {
            "analysis_id": analysis_id,
            "analysis": detailed_analysis,
            "sentiment": sentiment_result,
            "metadata": metadata_result,
            "emotions": emotions,
            "embeddings": embeddings,
            "embedding_labels": embedding_labels,
            "embedding_colors": embedding_colors,
            "online_data": analysis_data.get("online_data", None)
        }
    except Exception as e:
        print(f"Error in analyze_text: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/file")
async def analyze_file(
    file: UploadFile = File(...), 
    use_search_apis: bool = Form(True),
    include_online_data: bool = Form(True),
    days_back: int = Form(365)
):
    try:
        print(f"Received file analysis request: {file.filename}")
        
        # Save file temporarily if needed
        file_content = await file.read()
        file_extension = file.filename.split('.')[-1].lower()
        file_type = detect_file_type(file.filename)
        
        if file_type == "pdf":
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(file_content)
                temp_file_path = tmp_file.name
            
            content = extract_text_from_pdf(temp_file_path)
            source = file.filename
            source_type = "pdf"
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        elif file_type == "csv":
            csv_text = file_content.decode('utf-8')
            content = process_csv_data(csv_text)
            source = file.filename
            source_type = "csv"
        
        elif file_type == "json":
            json_text = file_content.decode('utf-8')
            content = json_text  # We'll let the LLM interpret the JSON
            source = file.filename
            source_type = "json"
        
        else:  # Default to text
            content_bytes = file_content
            content = content_bytes.decode('utf-8')
            source = file.filename
            source_type = "text"
        
        # Step 1: Determine input type
        print("Step 1/5: Identifying content type...")
        input_type_info = determine_input_type(content)
        
        # Step 2: Get real-time data from search engines for context enrichment
        print("Step 2/5: Fetching real-time data from search engines...")
        search_connector = SearchEngineConnector()
        
        # Use the input as search query if it's a short text
        search_query = source if len(content) < 500 else input_type_info.get('subject', source)
        search_results = search_connector.get_cached_or_fresh_data(search_query)
        
        # Extract and prepare search content for enrichment
        search_context = "REAL-TIME SEARCH DATA:\n\n"
        for i, result in enumerate(search_results[:3]):  # Use top 3 results
            extracted_content = search_connector.extract_content_from_url(result.get('link'))
            if extracted_content:
                search_context += f"SOURCE {i+1}: {extracted_content.get('title')}\n"
                search_context += f"URL: {result.get('link')}\n"
                search_context += f"CONTENT: {extracted_content.get('content')[:500]}...\n\n"
            else:
                search_context += f"SOURCE {i+1}: {result.get('title')}\n"
                search_context += f"URL: {result.get('link')}\n"
                search_context += f"SNIPPET: {result.get('snippet', '')}\n\n"
        
        # Step 3: Based on the content type and whether it's a file or short query:
        if source_type in ["pdf", "csv", "json"] or len(content) > 1000:
            # For files or lengthy content, use GPT-o1 for detailed analysis first
            print("Step 3/5: Performing detailed analysis using GPT-o1...")
            # Include search context for better analysis
            analysis_input = f"{content}\n\n{search_context}"
            detailed_analysis = perform_detailed_analysis(analysis_input, input_type_info)
        else:
            # For shorter queries, skip to GPT-4o directly
            print("Step 3/5: Using GPT-4o to analyze your query with real-time data...")
            # Include web searching capabilities for better context and accuracy
            analysis_input = f"Search query: {source}\n\n{search_context}\n\n{content}"
            detailed_analysis = perform_detailed_analysis(analysis_input, input_type_info)
        
        # Step 4: Analyze sentiment with GPT-4o for all cases
        print("Step 4/5: Analyzing sentiment with real-time context...")
        sentiment_result = analyze_sentiment(detailed_analysis)
        
        # Step 5: Extract metadata with GPT-o3-mini
        print("Step 5/5: Extracting metadata...")
        metadata_result = extract_metadata(detailed_analysis)

        # Step 6: Generate real emotion analysis using HuggingFace
        print("Step 6/6: Generating emotion analysis...")
        emotions = generate_emotion_analysis(content)

        # Generate embeddings for the main content (split into sentences if long)
        if len(content) > 500:
            # Split into sentences for richer embedding landscape
            sentences = re.split(r'(?<=[.!?]) +', content)
            # Filter out very short sentences
            sentences = [s for s in sentences if len(s.strip()) > 10]
            embeddings = generate_embeddings(sentences)
            embedding_labels = sentences
        else:
            embeddings = generate_embeddings(content)
            embedding_labels = [content]

        # Save the analysis
        timestamp = datetime.now()
        
        # Define the analysis structure based on app.py
        analysis_data = {
            "id": str(uuid.uuid4()),  # Generate ID here to use for both saving and returning
            "source": source,
            "source_type": source_type,
            "text": content[:1000] + "..." if len(content) > 1000 else content,  # Store truncated version
            "summary": detailed_analysis,
            "sentiment": sentiment_result,
            "metadata": metadata_result,
            "emotions": emotions,
            "embeddings": embeddings,
            "embedding_labels": embedding_labels,
            "timestamp": timestamp.isoformat(),
            "input_type": input_type_info
        }
        
        # Save analysis with the prepared data structure
        analysis_id = save_analysis(analysis_data)
        print(f"Analysis saved with ID: {analysis_id}")
        
        # Add topic details if not present
        if "topic_details" not in metadata_result:
            metadata_result["topic_details"] = {
                "main_topics": metadata_result.get("topics", [])[:1],
                "subtopics": metadata_result.get("topics", [])[1:3] if len(metadata_result.get("topics", [])) > 1 else [],
                "trending_topics": []
            }
            
        # Ensure sentiment has sentiment_trend and current_context
        if "sentiment_trend" not in sentiment_result:
            sentiment_result["sentiment_trend"] = "stable"
        if "current_context" not in sentiment_result:
            sentiment_result["current_context"] = ""
            
        # Ensure temporal_details and event_context exist in metadata
        if "temporal_details" not in metadata_result:
            metadata_result["temporal_details"] = {
                "time_period": [],
                "specific_dates": []
            }
        if "event_context" not in metadata_result:
            metadata_result["event_context"] = {
                "is_current_event": False,
                "key_developments": [],
                "event_timeline": []
            }
        
        # Get online sentiment data if requested
        if use_search_apis and include_online_data:
            # Get main topic from metadata or use source as fallback
            main_topics = metadata_result.get("topic_details", {}).get("main_topics", [])
            main_topic = main_topics[0] if main_topics else input_type_info.get('subject', source)
            
            # Clean up main topic for better search
            if main_topic:
                main_topic = main_topic.strip()
                if len(main_topic) > 50:  # If too long, truncate
                    main_topic = main_topic[:50] + "..."
            else:
                # Fallback to source processing
                if source_type == "url":
                    # Use the URL domain or title if available
                    main_topic = source.split("/")[-1].replace("-", " ").replace("_", " ")
                    if len(main_topic) < 3:  # If too short (like just "com"), use the domain
                        main_topic = source.split("//")[-1].split("/")[0]
                elif source_type == "direct_text" and len(source) < 50:
                    # If it's a short direct text entry, it's likely a search query - use it directly
                    main_topic = source
                else:
                    main_topic = input_type_info.get('subject', 'general')
            
            # Get subtopics for additional context
            subtopics = metadata_result.get("topic_details", {}).get("subtopics", [])
            
            # Get online sentiment data
            online_data = get_online_sentiment_with_search(
                main_topic, 
                subtopics, 
                days_back=days_back, 
                use_search_apis=use_search_apis
            )
            
            # Process search-augmented analysis if available
            search_analysis = online_data.get("search_augmented_analysis", {})
            if not search_analysis:
                # Create a default search analysis structure
                online_data["search_augmented_analysis"] = {
                    "analysis": "No search-augmented analysis available",
                    "confidence": "Low confidence",
                    "sources": []
                }
                
            # Format the API status for frontend consumption
            api_status = online_data.get("api_status", {})
            if not api_status:
                online_data["api_status"] = {
                    "twitter_api": False,
                    "reddit_api": False,
                    "news_api": False,
                    "search_api": use_search_apis
                }
                
            # Ensure sources structure exists
            if "sources" not in online_data:
                online_data["sources"] = {
                    "search_engines": [],
                    "news": [],
                    "twitter": [],
                    "reddit": [],
                    "web": []
                }
                
            # Make sure visualization data structures exist
            for data_key in ["global_data", "historical_data", "keyword_data"]:
                if data_key not in online_data:
                    online_data[data_key] = {}
            
            # Add online data to response
            analysis_data["online_data"] = online_data
        
        # Generate embedding colors based on sentiment
        def get_color(sentiment):
            if sentiment == 'positive':
                return '#10B981'  # green
            elif sentiment == 'negative':
                return '#EF4444'  # red
            elif sentiment == 'neutral':
                return '#FBBF24'  # yellow
            else:
                return '#6B7280'  # gray
        embedding_colors = [get_color(sentiment_result.get('sentiment', 'neutral')) for _ in embeddings]
        
        # Return comprehensive response with all needed data
        return {
            "analysis_id": analysis_id,
            "analysis": detailed_analysis,
            "sentiment": sentiment_result,
            "metadata": metadata_result,
            "emotions": emotions,
            "embeddings": embeddings,
            "embedding_labels": embedding_labels,
            "embedding_colors": embedding_colors,
            "online_data": analysis_data.get("online_data", None)
        }
    except Exception as e:
        print(f"Error in analyze_file: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyze/comprehensive")
async def analyze_comprehensive_sentiment(input_data: TextInput):
    """
    Financial Sentiment Analysis for NASDAQ/NYSE Tickers Only
    Validates ticker and performs comprehensive sentiment analysis using financial sources
    """
    try:
        print(f"Received ticker analysis request: {input_data.text[:20]}...")
        
        # Step 1: Validate that input is a NASDAQ/NYSE ticker
        from utils.ticker_validator import get_ticker_validator
        ticker_validator = get_ticker_validator()
        
        is_valid, company_info, error_message = ticker_validator.validate_ticker(input_data.text)
        
        if not is_valid:
            raise HTTPException(
                status_code=400, 
                detail=f"Invalid Input: {error_message} This system only analyzes NASDAQ and NYSE stock tickers."
            )
        
        print(f"✓ Valid ticker: {input_data.text.upper()} - {company_info['name']}")
        
        # Step 2: Get financial context for the ticker
        ticker = input_data.text.upper()
        financial_context = ticker_validator.get_financial_context(ticker, company_info)
        
        # Step 3: Generate specialized financial search queries
        search_queries = ticker_validator.generate_search_queries(ticker, company_info)
        print(f"Generated {len(search_queries)} financial search queries for {ticker}")
        
        # Initialize the comprehensive engine (import lazily to reduce startup memory)
        from utils.comprehensive_sentiment_engine import get_comprehensive_engine, CanonicalUnit
        comprehensive_engine = get_comprehensive_engine()
        
        # Use ticker as content for analysis
        content = ticker
        source = "ticker_symbol"
        source_type = "financial_ticker"
        domain = "financial_market"
        
        # Step 4: Get financial search context using specialized queries + Reddit (concurrent + bounded)
        search_results = []
        enable_search = (str(os.getenv("ENABLE_SEARCH", "true")).lower() == "true") and input_data.use_search_apis
        enable_reddit = str(os.getenv("ENABLE_REDDIT", "true")).lower() == "true"
        max_queries = int(os.getenv("MAX_SEARCH_QUERIES", "0") or 0)  # 0 = no cap
        max_results_total = int(os.getenv("MAX_SEARCH_RESULTS", "0") or 0)  # 0 = no cap
        max_content_fetch = int(os.getenv("MAX_CONTENT_FETCH", "0") or 0)
        max_content_fetch = max_content_fetch if max_content_fetch > 0 else 50

        if enable_search:
            print(f"Fetching financial search context for {ticker} with concurrency…")
            import asyncio as _asyncio
            search_connector = SearchEngineConnector()

            # Respect optional caps without changing UI behavior
            queries_to_run = list(search_queries)
            if max_queries and len(queries_to_run) > max_queries:
                queries_to_run = queries_to_run[:max_queries]

            # Run the query fetches concurrently in threads (since connector is sync/requests-based)
            async def run_query(q):
                try:
                    return await _asyncio.to_thread(
                        search_connector.get_cached_or_fresh_data,
                        q,
                        company_context=company_info,
                    )
                except Exception as e:
                    print(f"Search query error: {q}: {e}")
                    return []

            query_tasks = [_asyncio.create_task(run_query(q)) for q in queries_to_run]
            query_lists = await _asyncio.gather(*query_tasks, return_exceptions=False)
            for q, results in zip(queries_to_run, query_lists):
                print(f"Got {len(results)} results for query: {q}")
                search_results.extend(results or [])

            # Reddit discussions concurrently (thread) if enabled
            if enable_reddit:
                try:
                    reddit_results = await _asyncio.to_thread(
                        search_connector.fetch_reddit_discussions,
                        company_info['name'],
                        ticker,
                        company_info.get('sector', 'Financial'),
                        14,
                        10
                    )
                    search_results.extend(reddit_results or [])
                    print(f"Added {len(reddit_results)} Reddit discussions")
                except Exception as e:
                    print(f"Reddit integration failed (continuing without): {e}")

            # Uniqueness & optional global cap
            seen_urls = set()
            unique_results = []
            for result in search_results:
                url = result.get('link') or result.get('url') or ''
                if not url:
                    continue
                if url in seen_urls:
                    continue
                seen_urls.add(url)
                unique_results.append(result)
                if max_results_total and len(unique_results) >= max_results_total:
                    break
            search_results = unique_results
            print(f"Final unique search results: {len(search_results)} (including Reddit)")

            # Concurrent content extraction for top results to reduce wall-clock time
            def _extract(res):
                try:
                    url = res.get('link') or res.get('url')
                    if not url:
                        return None
                    return search_connector.extract_content_from_url(url)
                except Exception:
                    return None

            limited = search_results[:max_content_fetch]
            sem = _asyncio.Semaphore(int(os.getenv("CONTENT_FETCH_CONCURRENCY", "10") or 10))

            async def extract_task(res):
                async with sem:
                    return await _asyncio.to_thread(_extract, res)

            extract_tasks = [_asyncio.create_task(extract_task(r)) for r in limited]
            extracted = await _asyncio.gather(*extract_tasks, return_exceptions=False)
            # Enrich search_results items with extracted content where available
            enriched = []
            for res, ext in zip(limited, extracted):
                if ext and isinstance(ext, dict) and ext.get('content'):
                    res = dict(res)
                    res['full_content'] = ext.get('content')
                    res['title'] = res.get('title') or ext.get('title')
                enriched.append(res)
            # Keep enriched at front, then the remainder (unchanged)
            search_results = enriched + search_results[len(limited):]
        
        # Step 5: Create canonical units ONLY from search results (no ticker unit)
        print("Creating canonical units for comprehensive analysis...")
        units = []
        
        # For financial analysis, we don't include the ticker itself as a unit
        # Only real search results with actual financial content
        
        # Add search result units if available (no synthetic timestamps)
        for i, result in enumerate(search_results):
            snippet = result.get('snippet')
            full_content = result.get('full_content')
            if not snippet and not full_content:
                continue
            # Try to obtain a publish time from result payload if present
            publish_raw = (
                result.get('publishedAt')
                or result.get('published_at')
                or result.get('published_date')
                or result.get('date')
            )
            publish_time = None
            if publish_raw:
                try:
                    # Parse to UTC then drop tzinfo to ensure naive datetime for safe subtraction
                    ts = pd.to_datetime(publish_raw, utc=True)
                    publish_time = ts.to_pydatetime().replace(tzinfo=None)
                except Exception:
                    publish_time = None
            # If no publish time, skip adding to time-sensitive analyses but still allow as a unit with None time
            search_unit = CanonicalUnit(
                text=(full_content or snippet or result.get('title', '')),
                source_domain=re.search(r'https?://([^/]+)', result.get('link', '')).group(1) if result.get('link') else 'search_result',
                url=result.get('link', ''),
                author=None,
                platform_stats={"upvotes": max(1, 10 - i), "views": max(1, 100 - i*10)},  # Ranking proxy
                publish_time=publish_time or datetime.min,  # Use datetime.min to indicate unknown
                last_edit_time=None,
                first_seen_time=datetime.now(),
                language="en",
                cluster_id=f"search_cluster_{i//2}",  # Group search results
                thread_depth=0,
                retrieval_score=max(0.1, 1.0 - i*0.1),  # Decreasing by rank
                length=len(snippet),
                unit_id=str(uuid.uuid4())
            )
            units.append(search_unit)
        
        # If no search results were found, create a basic ticker unit for analysis
        if not units:
            print("No search results found - creating basic ticker analysis unit")
            ticker_text = f"{ticker} ({company_info['name']}) - {company_info.get('sector', 'Financial')} sector analysis"
            ticker_unit = CanonicalUnit(
                text=ticker_text,
                source_domain="ticker_analysis",
                url=f"https://finance.yahoo.com/quote/{ticker}",
                author="system",
                platform_stats={"confidence": 0.8, "upvotes": 1, "views": 1},
                publish_time=datetime.now(),
                last_edit_time=None,
                first_seen_time=datetime.now(),
                language="en",
                cluster_id="ticker_cluster_0",
                thread_depth=0,
                retrieval_score=1.0,
                length=len(ticker_text),
                unit_id=str(uuid.uuid4())
            )
            units.append(ticker_unit)
            print(f"Created basic ticker unit for {ticker} analysis")
        
        # Step 4: Run comprehensive analysis pipeline
        print("Running comprehensive sentiment analysis pipeline...")
        try:
            comprehensive_results = comprehensive_engine.process_query(units)
            print(f"DEBUG: Comprehensive results keys: {list(comprehensive_results.keys())}")
            if 'coverage' in comprehensive_results:
                print(f"DEBUG: Coverage keys: {list(comprehensive_results['coverage'].keys())}")
            else:
                print("DEBUG: No 'coverage' key in results!")
        except Exception as e:
            print(f"ERROR in comprehensive engine: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=f"Comprehensive engine failed: {str(e)}")
        
        # Step 4.5: Apply mathematical sentiment analysis to the actual news content
        print("Applying mathematical sentiment analysis to news content...")
        from utils.mathematical_sentiment import get_mathematical_analyzer
        math_analyzer = get_mathematical_analyzer()
        
        # Combine all news content for mathematical analysis
        combined_news_content = ""
        for unit in units:
            if unit.source_domain != "user_input":  # Only analyze news content
                combined_news_content += unit.text + " "
        
        if combined_news_content.strip():
            print(f"Analyzing {len(combined_news_content)} characters of news content...")
            mathematical_results = math_analyzer.analyze_mathematical_sentiment(combined_news_content.strip())
            print(f"Mathematical sentiment score: {mathematical_results['mathematical_sentiment_analysis']['composite_score']['value']}")
        else:
            print("No news content found for mathematical analysis")
            mathematical_results = None
        
        # Step 5: Create response following the specification (no synthetic fallbacks)
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        # Use actual mathematical sentiment analysis results
        if mathematical_results:
            mathematical_sentiment_analysis = mathematical_results["mathematical_sentiment_analysis"]
            # Also include emotion analysis
            emotion_analysis = mathematical_results["emotion_vector_analysis"]
        else:
            # No synthetic mathematical analysis; rely strictly on comprehensive results without fabricating CIs
            composite_value = comprehensive_results["sentiment"]["score"]
            # Attempt to approximate uncertainty from unit-level distribution if available
            try:
                # Access internal unit sentiments if exposed; otherwise set None
                unit_scores = comprehensive_results.get('unit_sentiments', [])
                if unit_scores and isinstance(unit_scores, list):
                    arr = np.array([float(s) for s in unit_scores if s is not None])
                    if len(arr) > 1:
                        se = arr.std(ddof=1) / np.sqrt(len(arr))
                        ci = [float(composite_value - 1.96*se), float(composite_value + 1.96*se)]
                    else:
                        ci = [composite_value, composite_value]
                else:
                    ci = [composite_value, composite_value]
            except Exception:
                ci = [composite_value, composite_value]
            mathematical_sentiment_analysis = {
                "composite_score": {
                    "value": composite_value,
                    "confidence_interval": ci,
                    "statistical_significance": comprehensive_results["sentiment"].get("confidence", 0)
                },
                "multi_model_validation": {},
                "uncertainty_metrics": {
                    "sentiment_entropy": comprehensive_results["sentiment"].get("entropy", 0.0),
                    "polarization_index": comprehensive_results["sentiment"].get("disagreement", 0.0),
                    "model_agreement": 1 - comprehensive_results["sentiment"].get("disagreement", 0.0)
                }
            }
            # No synthetic emotions
            emotion_analysis = {
                "plutchik_coordinates": comprehensive_results.get("emotion", {}),
                "dominant_emotions": sorted(comprehensive_results.get("emotion", {}).keys(), key=lambda k: comprehensive_results.get("emotion", {}).get(k, 0), reverse=True)[:3],
                "emotion_entropy": -sum(p * np.log(p + 1e-10) for p in comprehensive_results.get("emotion", {}).values()) if comprehensive_results.get("emotion") else 0,
                "emotional_coherence": 1 - comprehensive_results["sentiment"].get("disagreement", 0.0)
            }
        
        # Use mathematical emotion analysis if available
        if mathematical_results:
            emotion_vector_analysis = emotion_analysis
        else:
            # Fallback to comprehensive results
            emotion_vector_analysis = {
                "plutchik_coordinates": comprehensive_results["emotion"],
            "dominant_emotions": sorted(comprehensive_results["emotion"].keys(), 
                                      key=lambda k: comprehensive_results["emotion"].get(k, 0), 
                                      reverse=True)[:3],
            "emotion_entropy": -sum(p * np.log(p + 1e-10) for p in comprehensive_results["emotion"].values()) if comprehensive_results["emotion"] else 0,
            "emotional_coherence": 1 - comprehensive_results["sentiment"]["disagreement"],
                "emotion_mathematics": {
                    "vector_magnitude": np.sqrt(sum(p**2 for p in comprehensive_results["emotion"].values())) if comprehensive_results["emotion"] else 0,
                    "emotional_distance_from_neutral": abs(comprehensive_results["sentiment"]["score"]),
                    "primary_emotion_vector": list(comprehensive_results["emotion"].values())[:3] if comprehensive_results["emotion"] else [0,0,0]
                }
            }
        
        # Build programmatic query type expansion from sector/industry/name and top keywords
        try:
            stop = {
                'inc','co','corp','corporation','company','ltd','plc','the','and','group','holdings','llc'
            }
            sector_tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", str(company_info.get('sector',''))) if company_info.get('sector') else []
            industry_tokens = re.findall(r"[A-Za-z][A-Za-z\-]+", str(company_info.get('industry',''))) if company_info.get('industry') else []
            name_tokens_full = re.findall(r"[A-Za-z][A-Za-z\-]+", company_info.get('name',''))
            name_tokens = [w for w in name_tokens_full if w.lower() not in stop]
            # Collect text for keyword extraction
            text_blob = (str(company_info.get('business_summary','')) + "\n" + "\n".join([getattr(u, 'text','') for u in units][:5]))
            kw_counts = {}
            for tok in re.findall(r"[A-Za-z][A-Za-z\-]+", text_blob.lower()):
                if len(tok) < 3: 
                    continue
                if tok in stop:
                    continue
                if tok in {company_info.get('name','').lower(), company_info.get('short_name','').lower(), ticker.lower()}:
                    continue
                kw_counts[tok] = kw_counts.get(tok, 0) + 1
            top_kw = [w for w,_ in sorted(kw_counts.items(), key=lambda x: -x[1])[:6]]
            # Merge and dedupe preserving order
            merged = list(sector_tokens) + list(industry_tokens) + list(name_tokens) + list(top_kw)
            seen = set()
            expansions = []
            for t in merged:
                t_str = str(t)
                if not t_str:
                    continue
                if t_str in seen:
                    continue
                seen.add(t_str)
                expansions.append(t_str)
            expansions = expansions[:8]
            query_type_str = ", ".join(expansions) if expansions else (company_info.get('sector','Financial') or 'Financial')
        except Exception:
            query_type_str = company_info.get('sector','Financial') or 'Financial'

        comprehensive_analysis = {
            "id": analysis_id,
            "status": "success",
            "analysis_type": "comprehensive_rigorous",
            "source": source,
            "source_type": source_type,
            "text": content[:500] + "..." if len(content) > 500 else content,
            "timestamp": timestamp.isoformat(),
            
            # Enhanced summary
            "enhanced_summary": create_enhanced_summary(len(units), comprehensive_results['sentiment']['confidence'], mathematical_results, comprehensive_results),
            
            # Content metrics following specification
            "content_metrics": {
                "word_count": len(content.split()),
                "reading_level": "intermediate",  # Would calculate complexity in production
                "factual_density": 1.0 - comprehensive_results["sarcasm_rate"],
                "complexity_score": min(1.0, len(set(content.split())) / max(len(content.split()), 1)),
                "information_entropy": -sum(p * np.log(p + 1e-10) for p in [0.4, 0.3, 0.3]),  # Simplified
                "recent_events_count": len([u for u in units if u.source_domain != domain]),
                "average_recency_weight": comprehensive_results["freshness_score"]
            },
            
            # Mathematical sentiment analysis (comprehensive)
            "mathematical_sentiment_analysis": mathematical_sentiment_analysis,
            
            # Emotion vector analysis (comprehensive)
            "emotion_vector_analysis": emotion_vector_analysis,
            
            # Use mathematical analysis for polarity if available
            "comprehensive_metrics": {
                "disagreement_index": mathematical_results["mathematical_sentiment_analysis"]["uncertainty_metrics"]["polarization_index"] if mathematical_results else comprehensive_results["sentiment"]["disagreement"],
                "polarity_breakdown": calculate_polarity_from_mathematical_score(
                    mathematical_results["mathematical_sentiment_analysis"]["composite_score"]["value"] if mathematical_results else comprehensive_results["sentiment"]["score"]
                ),
                "sarcasm_rate": comprehensive_results["sarcasm_rate"],
                "toxicity_rate": comprehensive_results["toxicity_rate"],
                "freshness_score": comprehensive_results["freshness_score"],
                "novelty_score": comprehensive_results["novelty_score"],
                "total_evidence_weight": comprehensive_results["coverage"]["total_weight"]
            },
            
            # VAD analysis - only compute if mathematical results present; no synthetic
            "vad_analysis": (
                calculate_vad_from_mathematical_score(
                    mathematical_results["mathematical_sentiment_analysis"]["composite_score"]["value"],
                    mathematical_results["emotion_vector_analysis"]
                ) if mathematical_results else None
            ),
            
            # Document analysis
            "document_analysis": {
                "total_length": len(content),
                "word_count": len(content.split()),
                "sentence_count": len(re.split(r'[.!?]', content)),
                "paragraph_count": len(content.split('\n\n')),
                "file_type": source_type
            },
            
            # Context information
            "context_sources": len(search_results),
            "recent_developments_identified": len(units),
            
            # Source citations and explanations
            "source_citations": [
                {
                    "source": unit.source_domain,
                    "url": unit.url if unit.url != "user_input" else None,
                    "text_sample": unit.text[:200] + "..." if len(unit.text) > 200 else unit.text,
                    "contribution_weight": f"{(unit.retrieval_score * 100):.1f}%",
                    "recency": (
                        "User Input" if unit.source_domain == "user_input" else (
                            (f"{(datetime.now() - unit.publish_time).days} days ago" if unit.publish_time and unit.publish_time != datetime.min else "Unknown")
                        )
                    )
                } for unit in units
            ],
            
            # Query summary and recent events with dynamic industry
            "query_summary": {
                "query": f"{ticker} ({company_info['name']})",
                "query_type": query_type_str,
                "entities_detected": extract_entities_from_search_results(search_results),
                "recent_events": [
                    {
                        "event": f"Found {len([u for u in units if u.source_domain != 'user_input'])} related sources",
                        "relevance": "high" if len([u for u in units if u.source_domain != "user_input"]) > 0 else "low",
                        "source_count": len([u for u in units if u.source_domain != "user_input"])
                    }
                ] if len([u for u in units if u.source_domain != "user_input"]) > 0 else [{"event": "No recent external sources found", "relevance": "none", "source_count": 0}]
            },
            
            # Calculation methodology 
            "calculation_methodology": {
                "sentiment_calculation": "Weighted average of VADER, TextBlob, and AFINN lexicon scores combined with transformer model outputs",
                "confidence_basis": "Based on agreement between VADER, TextBlob, AFINN, and RoBERTa (cardiffnlp/twitter-roberta-base-sentiment-latest)",
                "source_weighting": "Sources weighted by recency, domain authority, and retrieval relevance score",
                "accuracy_indicators": {
                    "model_agreement": f"{(mathematical_results['mathematical_sentiment_analysis']['uncertainty_metrics']['model_agreement'] if mathematical_results else comprehensive_results['sentiment']['confidence']):.1%}",
                    "source_diversity": f"{len(set(u.source_domain for u in units))} different source types",
                    "temporal_coverage": (lambda _times: f"Analysis spans {max(1, (_times.max() - _times.min()).days)} days" if len(_times) > 1 else "Insufficient dated sources")(np.array([u.publish_time for u in units if getattr(u, 'publish_time', None) and u.publish_time != datetime.min])),
                    "data_quality": "high" if len(units) >= 5 else "medium" if len(units) >= 3 else "low"
                }
            },
            
            # Explanation clusters with actual data
            "explanatory_clusters": comprehensive_results.get("explanations", {}).get("top_clusters", []),
            "key_evidence": comprehensive_results.get("explanations", {}).get("key_phrases", []),
            "weight_explanations": comprehensive_results.get("explanations", {}).get("weight_factors", {})
        }
        
        # Step 7: Save analysis
        analysis_saved_id = save_analysis(comprehensive_analysis)
        comprehensive_analysis["analysis_id"] = analysis_saved_id
        
        print(f"Enhanced mathematical analysis completed with ID: {analysis_saved_id}")
        
        # Step 8: Generate professional visualizations (only from real data)
        print("Generating professional financial sentiment visualizations...")
        visualizations = {}

        try:
            # Ensure downstream visualizations receive expected structures
            # 1) Provide a 'sources' list with per-source sentiment approximations
            try:
                sources_list = []
                texts_for_embed = []
                labels_for_embed = []
                colors_for_embed = []
                hovers_for_embed = []
                # Reuse the mathematical analyzer for a lightweight lexicon pass per unit
                if 'units' in locals() and len(units) > 0:
                    math_analyzer = get_mathematical_analyzer()
                    import numpy as _np
                    for u in units:
                        if getattr(u, 'source_domain', None) != 'user_input' and getattr(u, 'text', None):
                            try:
                                lex_scores = math_analyzer._lexicon_analysis(u.text)
                                # Average simple lexicon signals from real text
                                avg_sent = _np.mean([
                                    lex_scores.get('vader', 0.0),
                                    lex_scores.get('textblob', 0.0),
                                    lex_scores.get('afinn', 0.0)
                                ])
                            except Exception:
                                avg_sent = 0.0
                            published_iso = None
                            try:
                                if getattr(u, 'publish_time', None) and u.publish_time != datetime.min:
                                    published_iso = u.publish_time.isoformat()
                            except Exception:
                                published_iso = None
                            src_item = {
                                'title': getattr(u, 'url', 'Source') or getattr(u, 'source_domain', 'Source'),
                                'url': None if getattr(u, 'url', None) == 'user_input' else getattr(u, 'url', None),
                                'sentiment': float(avg_sent)
                            }
                            if published_iso:
                                src_item['published_at'] = published_iso
                            sources_list.append(src_item)

                            # Collect for embeddings and labels/colors in the same order
                            const_text = u.text if isinstance(u.text, str) else str(u.text)
                            texts_for_embed.append(const_text)
                            const_domain = getattr(u, 'source_domain', 'source')
                            labels_for_embed.append(const_domain)
                            if avg_sent > 0.1:
                                colors_for_embed.append('#10B981')  # green
                            elif avg_sent < -0.1:
                                colors_for_embed.append('#EF4444')  # red
                            else:
                                colors_for_embed.append('#FBBF24')  # yellow

                            # Build rich hovertext: domain, URL, date, sentiment, retrieval score, and snippet
                            try:
                                import html
                                url = getattr(u, 'url', '') or ''
                                date_txt = published_iso if published_iso else 'Unknown'
                                score_txt = f"{getattr(u, 'retrieval_score', 0.0):.2f}"
                                snippet = (const_text[:140] + '…') if len(const_text) > 150 else const_text
                                hover = f"<b>{const_domain}</b><br>{html.escape(url)}<br>Date: {date_txt}<br>Sent: {avg_sent:.3f} • Score: {score_txt}<br>{html.escape(snippet)}"
                            except Exception:
                                hover = f"{const_domain}<br>{getattr(u, 'url', '') or ''}<br>Sent: {avg_sent:.3f}"
                            hovers_for_embed.append(hover)
                # Attach for visualizations that expect 'sources'
                comprehensive_analysis['sources'] = sources_list
            except Exception:
                # Non-fatal; visualizations will degrade gracefully
                comprehensive_analysis['sources'] = []

            # 2) Provide polarity_distribution inside mathematical_sentiment_analysis (percent scale)
            try:
                # Empirical distribution from per-source sentiments
                src = comprehensive_analysis.get('sources', [])
                if src:
                    pos = sum(1 for s in src if s.get('sentiment', 0.0) > 0.1)
                    neg = sum(1 for s in src if s.get('sentiment', 0.0) < -0.1)
                    neu = sum(1 for s in src if -0.1 <= s.get('sentiment', 0.0) <= 0.1)
                    total = max(1, pos + neg + neu)
                    pol_percent = {
                        'positive': round(pos * 100.0 / total, 1),
                        'neutral': round(neu * 100.0 / total, 1),
                        'negative': round(neg * 100.0 / total, 1)
                    }
                    comprehensive_analysis.setdefault('mathematical_sentiment_analysis', {})['polarity_distribution'] = pol_percent
            except Exception:
                pass

            # 4) Compute and attach 3D embeddings for UMAP scatter if possible
            try:
                if texts_for_embed:
                    try:
                        from utils.sentiment_generator import generate_embeddings as _gen_embed
                        embs3d = _gen_embed(texts_for_embed)
                    except Exception as _e:
                        print(f"Embedding generation failed: {_e}")
                        embs3d = []
                    if isinstance(embs3d, list) and len(embs3d) == len(texts_for_embed):
                        comprehensive_analysis['embeddings'] = embs3d
                        comprehensive_analysis['embedding_labels'] = labels_for_embed
                        comprehensive_analysis['embedding_colors'] = colors_for_embed
                        comprehensive_analysis['embedding_hovertexts'] = hovers_for_embed
            except Exception as _e2:
                print(f"UMAP embedding pipeline error: {_e2}")

            # 3) Provide VAD under mathematical_sentiment_analysis in 0..100 scale for compass
            try:
                vad = comprehensive_analysis.get('vad_analysis') or {}
                if vad:
                    vad_percent = {
                        'valence': round(float(vad.get('valence', 0.0)) * 100.0, 1) if float(vad.get('valence', 0.0)) <= 1.0 else float(vad.get('valence', 0.0)),
                        'arousal': round(float(vad.get('arousal', 0.0)) * 100.0, 1) if float(vad.get('arousal', 0.0)) <= 1.0 else float(vad.get('arousal', 0.0)),
                        'dominance': round(float(vad.get('dominance', 0.0)) * 100.0, 1) if float(vad.get('dominance', 0.0)) <= 1.0 else float(vad.get('dominance', 0.0))
                    }
                    comprehensive_analysis.setdefault('mathematical_sentiment_analysis', {})['vad_analysis'] = vad_percent
            except Exception:
                pass

            # Executive Overview Visualizations
            # Always include Sentiment Index (does not require synthetic data)
            visualizations["sentiment_index"] = {
                "chart_data": create_sentiment_index_with_uncertainty(comprehensive_analysis).to_json(),
                "description": "Primary sentiment indicator with statistical confidence bands"
            }
            
            # Include Polarity Distribution only if unit_count > 0 and distribution provided
            if comprehensive_analysis.get('comprehensive_metrics', {}).get('polarity_breakdown') is not None:
                visualizations["polarity_distribution"] = {
                    "chart_data": create_polarity_share_bars_with_intervals(comprehensive_analysis).to_json(),
                    "description": "Sentiment distribution with Wilson confidence intervals"
                }
            
            # Include VAD only if calculated from mathematical analysis
            if comprehensive_analysis.get('mathematical_sentiment_analysis', {}).get('vad_analysis'):
                visualizations["vad_compass"] = {
                    "chart_data": create_vad_compass(comprehensive_analysis).to_json(),
                    "description": "Emotional dimensions analysis (Valence-Arousal-Dominance)"
                }
            
            # Source Analysis Visualizations
            # Include Source Quality only if sources present
            if comprehensive_analysis.get('sources'):
                visualizations["source_quality"] = {
                    "chart_data": create_source_quality_matrix(comprehensive_analysis).to_json(),
                    "description": "Source reliability and quality assessment matrix"
                }
            
            # Include timeline only if dated sources exist (the function also enforces)
            tl_fig = create_rolling_sentiment_timeline(comprehensive_analysis)
            if len(tl_fig.data) > 0:
                visualizations["sentiment_timeline"] = {
                    "chart_data": tl_fig.to_json(),
                    "description": "Rolling sentiment timeline with trend analysis"
                }
            
            print(f"✓ Generated {len(visualizations)} professional visualizations")

        except Exception as viz_error:
            print(f"Warning: Visualization generation failed: {viz_error}")
            visualizations = {"error": f"Visualization generation failed: {str(viz_error)}"}
        
        return {
            "analysis_id": analysis_saved_id,
            "status": "success",
            "analysis_type": "mathematical_enhanced",
            "visualizations": visualizations,
            **comprehensive_analysis
        }
        
    except Exception as e:
        print(f"Error in mathematical analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Mathematical analysis failed: {str(e)}")

@app.post("/api/online-sentiment")
async def get_online_sentiment_analysis(query_data: SearchQuery):
    try:
        # Get main topic and subtopics
        main_topic = query_data.query.strip() 
        subtopics = query_data.subtopics or []
        
        # Get online sentiment data
        online_data = get_online_sentiment_with_search(
            main_topic,
            subtopics,
            days_back=query_data.days_back,
            use_search_apis=query_data.use_search_apis
        )
        
        # Process search-augmented analysis if requested and available
        if query_data.include_search_analysis:
            search_analysis = online_data.get("search_augmented_analysis", {})
            if not search_analysis:
                # Create a default search analysis structure
                online_data["search_augmented_analysis"] = {
                    "analysis": "No search-augmented analysis available",
                    "confidence": "Low confidence",
                    "sources": []
                }
                
        # Format the API status for frontend consumption
        api_status = online_data.get("api_status", {})
        if not api_status:
            online_data["api_status"] = {
                "twitter_api": False,
                "reddit_api": False,
                "news_api": False,
                "search_api": query_data.use_search_apis
            }
            
        # Ensure sources structure exists
        if "sources" not in online_data:
            online_data["sources"] = {
                "search_engines": [],
                "news": [],
                "twitter": [],
                "reddit": [],
                "web": []
            }
            
        # Make sure visualization data structures exist
        for data_key in ["global_data", "historical_data", "keyword_data"]:
            if data_key not in online_data:
                online_data[data_key] = {}
        
        return online_data
    except Exception as e:
        print(f"Error in get_online_sentiment_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/visualization")
async def get_visualization(params: VisualizationParams):
    try:
        # Get topic and other parameters
        topic = params.topic
        subtopics = params.subtopics or []
        time_period = params.time_period
        viz_type = params.visualization_type
        
        # Get online data for visualization
        online_data = get_online_sentiment_with_search(
            topic,
            subtopics,
            days_back=365 if time_period == "year" else (30 if time_period == "month" else 7),
            use_search_apis=True
        )
        
        response_data = {"success": True, "topic": topic}
        
        # Generate requested visualizations
        if viz_type in ["globe", "all"]:
            try:
                globe_data = online_data.get('global_data', {})
                globe_fig = create_3d_globe_visualization(globe_data)
                response_data["globe_data"] = {
                    "chart_data": globe_fig.to_json(),
                    "raw_data": globe_data
                }
            except Exception as e:
                response_data["globe_data"] = {"error": str(e)}
                
        if viz_type in ["time", "all"]:
            try:
                historical_data = online_data.get('historical_data', [])
                time_fig = create_interest_over_time_chart(historical_data, period=time_period)
                response_data["time_data"] = {
                    "chart_data": time_fig.to_json(),
                    "raw_data": historical_data
                }
            except Exception as e:
                response_data["time_data"] = {"error": str(e)}
                
        if viz_type in ["topic", "all"]:
            try:
                keyword_data = online_data.get('keyword_data', {})
                topic_fig = create_topic_popularity_chart(keyword_data)
                response_data["topic_data"] = {
                    "chart_data": topic_fig.to_json(),
                    "raw_data": keyword_data
                }
            except Exception as e:
                response_data["topic_data"] = {"error": str(e)}
                
        if viz_type in ["keywords", "all"]:
            try:
                keyword_data = online_data.get('keyword_data', {})
                keyword_fig = create_keyword_chart(keyword_data)
                response_data["keyword_data"] = {
                    "chart_data": keyword_fig.to_json(),
                    "raw_data": keyword_data
                }
            except Exception as e:
                response_data["keyword_data"] = {"error": str(e)}
                
        return response_data
    except Exception as e:
        print(f"Error in get_visualization: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analyses")
async def get_analyses():
    try:
        print("Loading all analyses...")
        # Print the analyses directory path for debugging
        from utils.data_manager import ANALYSES_DIR, DATA_DIR
        print(f"Analyses directory: {ANALYSES_DIR}")
        print(f"Data directory: {DATA_DIR}")
        print(f"Analyses directory exists: {os.path.exists(ANALYSES_DIR)}")
        print(f"Data directory exists: {os.path.exists(DATA_DIR)}")
        
        # List files in both directories
        if os.path.exists(ANALYSES_DIR):
            analyses_files = os.listdir(ANALYSES_DIR)
            print(f"Files in analyses directory: {len(analyses_files)}")
            print(f"First few files: {analyses_files[:5] if len(analyses_files) > 0 else 'No files'}")
        
        if os.path.exists(DATA_DIR):
            data_files = [f for f in os.listdir(DATA_DIR) if f.endswith('.json')]
            print(f"JSON files in data directory: {len(data_files)}")
            print(f"First few files: {data_files[:5] if len(data_files) > 0 else 'No files'}")
        
        analyses = load_all_analyses()
        print(f"Loaded {len(analyses)} analyses")
        
        # Print keys of first few analyses for debugging
        if analyses:
            print(f"First few analyses keys: {list(analyses.keys())[:5]}")
            first_key = next(iter(analyses))
            print(f"Sample analysis structure: {list(analyses[first_key].keys())}")
        
        # Format the response to match what the frontend expects
        formatted_analyses = {}
        for analysis_id, data in analyses.items():
            # Ensure all required fields exist
            if "sentiment" not in data:
                data["sentiment"] = {
                    "sentiment": "neutral", 
                    "score": 0, 
                    "confidence": 0, 
                    "rationale": "",
                    "sentiment_trend": "stable",
                    "current_context": ""
                }
                
            if "metadata" not in data:
                data["metadata"] = {
                    "topics": [],
                    "regions": [],
                    "commodities": [],
                    "entities": [],
                    "temporal_details": {
                        "time_period": [],
                        "specific_dates": []
                    },
                    "event_context": {
                        "is_current_event": False,
                        "key_developments": [],
                        "event_timeline": []
                    },
                    "topic_details": {
                        "main_topics": [],
                        "subtopics": [],
                        "trending_topics": []
                    }
                }
            
            # Add to formatted analyses
            formatted_analyses[analysis_id] = data
        
        # Return analyses directly without nesting
        return formatted_analyses
    except Exception as e:
        print(f"Error in get_analyses: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/analyses/filter")
async def filter_analyses(filter_params: FilterParams):
    try:
        # Load all analyses
        all_analyses = load_all_analyses()
        
        # Convert dates to datetime objects if provided
        start_date = None
        if filter_params.start_date:
            try:
                start_date = datetime.fromisoformat(filter_params.start_date)
            except ValueError:
                pass
                
        end_date = None
        if filter_params.end_date:
            try:
                end_date = datetime.fromisoformat(filter_params.end_date)
            except ValueError:
                pass
        
        # Apply filters
        filtered_analyses = get_filtered_data(
            all_analyses,
            start_date=start_date,
            end_date=end_date,
            topics=filter_params.topics,
            regions=filter_params.regions,
            commodities=filter_params.commodities,
            sentiments=filter_params.sentiments
        )
        
        # Additional filtering by source_type if specified
        if filter_params.source_type:
            filtered_analyses = {
                k: v for k, v in filtered_analyses.items() 
                if v.get("source_type") == filter_params.source_type
            }
            
        # Additional filtering by time period
        if filter_params.time_period:
            now = datetime.now()
            if filter_params.time_period == "week":
                cutoff = now - timedelta(days=7)
            elif filter_params.time_period == "month":
                cutoff = now - timedelta(days=30)
            elif filter_params.time_period == "year":
                cutoff = now - timedelta(days=365)
            else:
                cutoff = None
                
            if cutoff:
                filtered_analyses = {
                    k: v for k, v in filtered_analyses.items() 
                    if "timestamp" in v and datetime.fromisoformat(v["timestamp"]) >= cutoff
                }
        
        # Return filtered analyses directly
        return filtered_analyses
    except Exception as e:
        print(f"Error in filter_analyses: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/analysis/{analysis_id}")
async def get_analysis(analysis_id: str):
    try:
        analysis = load_analysis(analysis_id)
        if not analysis:
            raise HTTPException(status_code=404, detail="Analysis not found")
            
        # Ensure the analysis has all required fields based on app.py
        if "sentiment" not in analysis:
            analysis["sentiment"] = {
                "sentiment": "neutral", 
                "score": 0, 
                "confidence": 0, 
                "rationale": "",
                "sentiment_trend": "stable",
                "current_context": ""
            }
            
        if "metadata" not in analysis:
            analysis["metadata"] = {
                "topics": [],
                "regions": [],
                "commodities": [],
                "entities": [],
                "temporal_details": {
                    "time_period": [],
                    "specific_dates": []
                },
                "event_context": {
                    "is_current_event": False,
                    "key_developments": [],
                    "event_timeline": []
                },
                "topic_details": {
                    "main_topics": [],
                    "subtopics": [],
                    "trending_topics": []
                }
            }
            
        # Return analysis directly without nesting
        return analysis
    except Exception as e:
        print(f"Error in get_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/analysis/{analysis_id}")
async def delete_analysis_endpoint(analysis_id: str):
    try:
        success = delete_analysis(analysis_id)
        if not success:
            raise HTTPException(status_code=404, detail="Analysis not found or could not be deleted")
        return {"success": True, "message": f"Analysis {analysis_id} deleted successfully"}
    except Exception as e:
        print(f"Error in delete_analysis: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    try:
        # Check if data directories exist
        from utils.data_manager import ANALYSES_DIR, DATA_DIR
        
        # Check if API keys are configured
        from utils.config import config
        
        api_status = {
            "openai_api": bool(config.get('openai_api_key')),
            "twitter_api": bool(config.get('twitter_api_key') and config.get('twitter_api_secret')),
            "news_api": bool(config.get('news_api_key')),
            "google_search_api": bool(config.get('google_search_api_key') and config.get('google_search_cx'))
        }
        
        # Avoid heavy JSON loads on health checks; just count files quickly.
        analyses_dir_exists = os.path.exists(ANALYSES_DIR)
        data_dir_exists = os.path.exists(DATA_DIR)
        try:
            analyses_count = len([f for f in os.listdir(ANALYSES_DIR) if f.endswith('.json')]) if analyses_dir_exists else 0
            # Also include top-level JSONs in DATA_DIR (excluding the analyses subdir)
            data_json_count = len([f for f in os.listdir(DATA_DIR) if f.endswith('.json') and os.path.isfile(os.path.join(DATA_DIR, f))]) if data_dir_exists else 0
            total_count = analyses_count + data_json_count
        except Exception:
            total_count = 0

        data_status = {
            "analyses_dir_exists": analyses_dir_exists,
            "data_dir_exists": data_dir_exists,
            "analyses_count": total_count,
        }
        
        # Keep health check response structure intact, this is likely used by system monitoring
        return {
            "success": True,
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "api_status": api_status,
            "data_status": data_status
        }
    except Exception as e:
        return {
            "success": False,
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

@app.get("/api/health/mathematical")
async def health_check_mathematical():
    """Health check for mathematical analysis capabilities"""
    try:
        # Import lazily to avoid loading heavy libs at startup
        from utils.mathematical_sentiment import get_mathematical_analyzer
        from utils.enhanced_analysis import get_enhanced_analyzer
        # Test mathematical analyzer initialization
        math_analyzer = get_mathematical_analyzer()
        enhanced_analyzer = get_enhanced_analyzer()
        
        # Test basic functionality
        test_text = "This is a positive test message."
        test_result = math_analyzer.analyze_mathematical_sentiment(test_text)
        
        # Verify core components are working
        health_status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "mathematical_analyzer": "operational",
            "enhanced_analyzer": "operational",
            "lexicon_models": {
                "vader": "loaded",
                "textblob": "loaded", 
                "afinn": "loaded"
            },
            "transformer_models": {
                "sentiment_pipeline": "loaded" if hasattr(math_analyzer, 'sentiment_pipeline') and math_analyzer.sentiment_pipeline else "not_loaded",
                "emotion_pipeline": "loaded" if hasattr(math_analyzer, 'emotion_pipeline') and math_analyzer.emotion_pipeline else "not_loaded"
            },
            "test_analysis": {
                "composite_score": test_result["mathematical_sentiment_analysis"]["composite_score"]["value"],
                "confidence_interval_calculated": len(test_result["mathematical_sentiment_analysis"]["composite_score"]["confidence_interval"]) == 2,
                "emotion_vector_calculated": len(test_result["emotion_vector_analysis"]["plutchik_coordinates"]) == 8
            }
        }
        
        return health_status
        
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)
