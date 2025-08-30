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
    create_keyword_chart
)
from utils.sentiment_generator import generate_emotion_analysis, generate_embeddings

# New mathematical analysis imports
from utils.mathematical_sentiment import get_mathematical_analyzer
from utils.enhanced_analysis import get_enhanced_analyzer

app = FastAPI(title="Sentimizer API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000", "*"],  # Allow frontend origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

@app.post("/api/analyze/mathematical")
async def analyze_mathematical_sentiment(input_data: TextInput):
    """
    Enhanced mathematical sentiment analysis endpoint with statistical rigor
    """
    try:
        print(f"Received mathematical analysis request: {input_data.text[:50]}...")
        
        # Initialize analyzers
        math_analyzer = get_mathematical_analyzer()
        enhanced_analyzer = get_enhanced_analyzer()
        
        # Step 1: Process input content
        if input_data.text.startswith(('http://', 'https://')):
            content = extract_text_from_html(input_data.text)
            source = input_data.text
            source_type = "url"
        else:
            content = process_text_input(input_data.text)
            source = input_data.text if len(input_data.text) < 50 else "Direct Text Input"
            source_type = "direct_text"
        
        # Step 2: Get search context for enhanced analysis
        search_results = []
        if input_data.use_search_apis:
            print("Fetching search context for enhanced analysis...")
            search_connector = SearchEngineConnector()
            search_query = source if len(content) < 500 else content[:100]
            search_results = search_connector.get_cached_or_fresh_data(search_query)[:5]
        
        # Step 3: Perform mathematical sentiment analysis
        print("Performing mathematical sentiment analysis...")
        mathematical_results = math_analyzer.analyze_mathematical_sentiment(content)
        
        # Step 4: Generate enhanced summary with contextual metrics
        print("Generating enhanced summary...")
        enhanced_summary_results = enhanced_analyzer.generate_enhanced_summary(
            content, search_results
        )
        
        # Step 5: Analyze content structure
        structure_analysis = enhanced_analyzer.analyze_content_structure(content, source_type)
        
        # Step 6: Create comprehensive response
        analysis_id = str(uuid.uuid4())
        timestamp = datetime.now()
        
        comprehensive_analysis = {
            "id": analysis_id,
            "source": source,
            "source_type": source_type,
            "text": content[:500] + "..." if len(content) > 500 else content,
            "timestamp": timestamp.isoformat(),
            
            # Enhanced summary focusing on recent developments
            "enhanced_summary": enhanced_summary_results["enhanced_summary"],
            "content_metrics": {
                "word_count": enhanced_summary_results["word_count"],
                "reading_level": "graduate" if structure_analysis.get("complexity_score", 0) > 0.7 else "intermediate",
                **enhanced_summary_results["contextual_metrics"]
            },
            
            # Mathematical sentiment analysis (replaces arbitrary scores)
            **mathematical_results,
            
            # Content structure analysis
            "document_analysis": structure_analysis,
            
            # Search context
            "context_sources": len(search_results),
            "recent_developments_identified": enhanced_summary_results["recent_events_identified"]
        }
        
        # Step 7: Save analysis
        analysis_saved_id = save_analysis(comprehensive_analysis)
        comprehensive_analysis["analysis_id"] = analysis_saved_id
        
        print(f"Enhanced mathematical analysis completed with ID: {analysis_saved_id}")
        
        return {
            "analysis_id": analysis_saved_id,
            "status": "success",
            "analysis_type": "mathematical_enhanced",
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
        
        data_status = {
            "analyses_dir_exists": os.path.exists(ANALYSES_DIR),
            "data_dir_exists": os.path.exists(DATA_DIR),
            "analyses_count": len(load_all_analyses())
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
    uvicorn.run(app, host="0.0.0.0", port=8000)