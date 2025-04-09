import streamlit as st
import os
import tempfile
import time
import pandas as pd
import random
from datetime import datetime, timedelta
import json

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
from utils.data_manager import save_analysis, load_analysis, load_all_analyses
from utils.external_data import get_online_sentiment
from utils.visualizations import (
    create_3d_globe_visualization,
    create_interest_over_time_chart,
    create_topic_popularity_chart,
    create_keyword_chart
)

# Page configuration
st.set_page_config(
    page_title="Sentimizer | Advanced Sentiment Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    /* Typography */
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap');
    
    body {
        font-family: 'Roboto', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
    }
    
    /* Main title */
    .main-title {
        font-family: 'Roboto', sans-serif;
        font-weight: 700;
        font-size: 3rem;
        background: linear-gradient(45deg, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
        letter-spacing: -1px;
    }
    
    /* Subtitle */
    .subtitle {
        font-family: 'Roboto', sans-serif;
        font-weight: 300;
        font-size: 1.1rem;
        color: #6B7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Input container */
    .input-container {
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 1rem;
        background-color: #F9FAFB;
        margin-bottom: 1rem;
    }
    
    /* Analysis container */
    .analysis-container {
        border-radius: 8px;
        padding: 1.5rem;
        background-color: #F9FAFB;
        margin-bottom: 1.5rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    /* Sentiment styles */
    .sentiment-positive {
        color: #10B981;
        font-weight: 500;
    }
    
    .sentiment-neutral {
        color: #6B7280;
        font-weight: 500;
    }
    
    .sentiment-negative {
        color: #EF4444;
        font-weight: 500;
    }
    
    /* Badge styles */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.5rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    .badge-blue {
        background-color: #EBF5FF;
        color: #3B82F6;
    }
    
    .badge-green {
        background-color: #ECFDF5;
        color: #10B981;
    }
    
    .badge-yellow {
        background-color: #FFFBEB;
        color: #F59E0B;
    }
    
    .badge-red {
        background-color: #FEF2F2;
        color: #EF4444;
    }
    
    .badge-gray {
        background-color: #F3F4F6;
        color: #6B7280;
    }
    
    /* Processing indicator */
    .processing-indicator {
        text-align: center;
        padding: 2rem;
        border-radius: 8px;
        background-color: #F9FAFB;
        margin-bottom: 1.5rem;
    }
    
    /* Section titles */
    .section-title {
        font-size: 1.25rem;
        font-weight: 600;
        color: #1F2937;
        margin-bottom: 1rem;
        border-bottom: 1px solid #E5E7EB;
        padding-bottom: 0.5rem;
    }
    
    /* No results placeholder */
    .no-results {
        text-align: center;
        padding: 3rem 1rem;
        color: #6B7280;
        background-color: #F9FAFB;
        border-radius: 8px;
        border: 1px dashed #E5E7EB;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables
if 'history' not in st.session_state:
    st.session_state.history = []
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'current_analysis' not in st.session_state:
    st.session_state.current_analysis = None
if 'analyses' not in st.session_state:
    st.session_state.analyses = load_all_analyses()

# Title
st.markdown("<h1 class='main-title'>SENTIMIZER</h1>", unsafe_allow_html=True)
st.markdown("<p class='subtitle'>Advanced AI-Powered Sentiment Analysis Platform</p>", unsafe_allow_html=True)

# Main input
with st.container():
    st.markdown("<div class='input-container'>", unsafe_allow_html=True)
    
    # Main tabs for input types
    input_tabs = st.tabs(["Text/URL Input", "Upload File", "My Analyses"])
    
    with input_tabs[0]:
        text_input = st.text_area(
            "Enter text, URL, or paste content to analyze:",
            height=150,
            placeholder="Enter a URL, article, financial report, or any text you want to analyze..."
        )
        
        col1, col2 = st.columns([4, 1])
        with col1:
            st.caption("Type or paste text, enter a URL, or simply ask a question about a topic.")
        with col2:
            submit_text = st.button("Analyze", type="primary", use_container_width=True)
    
    with input_tabs[1]:
        uploaded_file = st.file_uploader(
            "Upload a file to analyze (PDF, CSV, JSON, or text file)",
            type=["pdf", "csv", "json", "txt"]
        )
        
        if uploaded_file:
            file_details = {
                "FileName": uploaded_file.name,
                "FileType": uploaded_file.type,
                "FileSize": f"{uploaded_file.size / 1024:.2f} KB"
            }
            st.write(f"File: **{file_details['FileName']}** ({file_details['FileSize']})")
        
        submit_file = st.button("Analyze File", type="primary", disabled=not uploaded_file)
    
    with input_tabs[2]:
        if st.session_state.analyses:
            analysis_options = {
                analysis_id: f"{data.get('source', 'Unknown')} ({data.get('timestamp', '').split('T')[0]})"
                for analysis_id, data in st.session_state.analyses.items()
            }
            
            selected_analysis = st.selectbox(
                "Select a previous analysis to view:",
                options=list(analysis_options.keys()),
                format_func=lambda x: analysis_options[x]
            )
            
            view_analysis = st.button("View Analysis", type="primary")
        else:
            st.info("No previous analyses found. Start by analyzing some content!")
            view_analysis = False
            selected_analysis = None
    
    st.markdown("</div>", unsafe_allow_html=True)

# Process input when submitted
if submit_text and text_input and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.spinner("Processing your input..."):
        # Determine if it's a URL or direct text
        if text_input.startswith(('http://', 'https://')):
            content = extract_text_from_html(text_input)
            source = text_input
            source_type = "url"
        else:
            content = process_text_input(text_input)
            source = text_input if len(text_input) < 50 else "Direct Text Input"
            source_type = "direct_text"
        
        # Step 1: Determine input type
        progress_text = st.empty()
        progress_text.markdown("**Step 1/4:** Identifying content type...")
        input_type_info = determine_input_type(content)
        
        # Step 2: Based on the content type and whether it's a file or short query:
        if source_type in ["pdf", "csv", "json"] or len(content) > 1000:
            # For files or lengthy content, use GPT-o1 for detailed analysis first
            progress_text.markdown("**Step 2/4:** Performing detailed analysis using GPT-o1...")
            detailed_analysis = perform_detailed_analysis(content, input_type_info)
        else:
            # For shorter queries, skip to GPT-4o directly
            progress_text.markdown("**Step 2/4:** Using GPT-4o to analyze your query...")
            # Include web searching capabilities for better context and accuracy
            detailed_analysis = f"Search query: {source}\n\n"
            detailed_analysis += perform_detailed_analysis(content, input_type_info)
        
        # Step 3: Analyze sentiment with GPT-4o for all cases
        progress_text.markdown("**Step 3/4:** Gathering online sentiment data...")
        sentiment_result = analyze_sentiment(detailed_analysis)
        
        # Step 4: Extract metadata with GPT-o3-mini
        progress_text.markdown("**Step 4/4:** Extracting metadata...")
        metadata_result = extract_metadata(detailed_analysis)
        
        # Save the analysis
        timestamp = datetime.now()
        analysis_id = save_analysis(
            source,
            content,
            detailed_analysis,
            sentiment_result,
            metadata_result,
            timestamp,
            source_type
        )
        
        # Update analyses in session state
        st.session_state.analyses = load_all_analyses()
        
        # Store the current analysis
        st.session_state.current_analysis = {
            "id": analysis_id,
            "source": source,
            "source_type": source_type,
            "detailed_analysis": detailed_analysis,
            "sentiment": sentiment_result,
            "metadata": metadata_result,
            "timestamp": timestamp.isoformat(),
            "input_type": input_type_info
        }
        
        # Add to history
        st.session_state.history.append(st.session_state.current_analysis)
    
    st.session_state.processing = False
    st.rerun()

# Process file when submitted
if submit_file and uploaded_file and not st.session_state.processing:
    st.session_state.processing = True
    
    with st.spinner("Processing your file..."):
        # Determine file type and extract content
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        if file_extension == 'pdf':
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                temp_file_path = tmp_file.name
            
            content = extract_text_from_pdf(temp_file_path)
            source = uploaded_file.name
            source_type = "pdf"
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
        
        elif file_extension == 'csv':
            content_bytes = uploaded_file.getvalue()
            csv_text = content_bytes.decode('utf-8')
            content = process_csv_data(csv_text)
            source = uploaded_file.name
            source_type = "csv"
        
        elif file_extension == 'json':
            content_bytes = uploaded_file.getvalue()
            json_text = content_bytes.decode('utf-8')
            content = json_text  # We'll let the LLM interpret the JSON
            source = uploaded_file.name
            source_type = "json"
        
        else:  # Default to text
            content_bytes = uploaded_file.getvalue()
            content = content_bytes.decode('utf-8')
            source = uploaded_file.name
            source_type = "text"
        
        # Step 1: Determine input type
        progress_text = st.empty()
        progress_text.markdown("**Step 1/4:** Identifying content type...")
        input_type_info = determine_input_type(content)
        
        # Step 2: Use GPT-o1 for file content (which is more suitable for structured data)
        progress_text.markdown("**Step 2/4:** Performing detailed analysis using GPT-o1...")
        detailed_analysis = perform_detailed_analysis(content, input_type_info)
        
        # Step 3: Analyze sentiment with GPT-4o and add online context
        progress_text.markdown("**Step 3/4:** Gathering online sentiment data...")
        sentiment_result = analyze_sentiment(detailed_analysis)
        
        # Step 4: Extract metadata with GPT-o3-mini
        progress_text.markdown("**Step 4/4:** Extracting metadata...")
        metadata_result = extract_metadata(detailed_analysis)
        
        # Save the analysis
        timestamp = datetime.now()
        analysis_id = save_analysis(
            source,
            content,
            detailed_analysis,
            sentiment_result,
            metadata_result,
            timestamp,
            source_type
        )
        
        # Update analyses in session state
        st.session_state.analyses = load_all_analyses()
        
        # Store the current analysis
        st.session_state.current_analysis = {
            "id": analysis_id,
            "source": source,
            "source_type": source_type,
            "detailed_analysis": detailed_analysis,
            "sentiment": sentiment_result,
            "metadata": metadata_result,
            "timestamp": timestamp.isoformat(),
            "input_type": input_type_info
        }
        
        # Add to history
        st.session_state.history.append(st.session_state.current_analysis)
    
    st.session_state.processing = False
    st.rerun()

# View selected analysis
if view_analysis and selected_analysis:
    analysis_data = st.session_state.analyses.get(selected_analysis)
    if analysis_data:
        # Convert to the same format as current_analysis
        st.session_state.current_analysis = {
            "id": selected_analysis,
            "source": analysis_data.get("source", "Unknown"),
            "source_type": analysis_data.get("source_type", "unknown"),
            "detailed_analysis": analysis_data.get("summary", "No summary available"),
            "sentiment": analysis_data.get("sentiment", {"sentiment": "neutral", "score": 0, "confidence": 0, "rationale": ""}),
            "metadata": analysis_data.get("metadata", {"topics": [], "regions": [], "commodities": []}),
            "timestamp": analysis_data.get("timestamp", datetime.now().isoformat()),
            "input_type": {"input_type": "unknown", "subject": "general"}
        }
        st.rerun()

# Show processing indicator
if st.session_state.processing:
    st.markdown("""
    <div class="processing-indicator">
        <p>Processing your content...</p>
        <div class="progress">
            <div class="progress-bar"></div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display current analysis
if st.session_state.current_analysis:
    analysis = st.session_state.current_analysis
    sentiment = analysis["sentiment"]
    metadata = analysis["metadata"]
    
    # Determine sentiment class for styling
    sentiment_class = f"sentiment-{sentiment.get('sentiment', 'neutral')}"
    
    # Format timestamp
    if isinstance(analysis["timestamp"], str):
        try:
            timestamp = datetime.fromisoformat(analysis["timestamp"])
            formatted_date = timestamp.strftime("%B %d, %Y at %I:%M %p")
        except:
            formatted_date = analysis["timestamp"]
    else:
        formatted_date = analysis["timestamp"].strftime("%B %d, %Y at %I:%M %p")
    
    # Use the actual search query or title as the main topic for more accurate representation
    if analysis["source_type"] == "url":
        # Use the URL domain or title if available
        main_topic = analysis["source"].split("/")[-1].replace("-", " ").replace("_", " ")
        if len(main_topic) < 3:  # If too short (like just "com"), use the domain
            main_topic = analysis["source"].split("//")[-1].split("/")[0]
    elif analysis["source_type"] == "direct_text" and len(analysis["source"]) < 50:
        # If it's a short direct text entry, it's likely a search query - use it directly
        main_topic = analysis["source"]
    else:
        # Otherwise use topics from metadata
        main_topics = metadata.get("topic_details", {}).get("main_topics", [])
        main_topic = main_topics[0] if main_topics else analysis.get('input_type', {}).get('subject', 'general')
    
    # Clean up main topic
    main_topic = main_topic.strip()
    if len(main_topic) > 50:  # If too long, truncate
        main_topic = main_topic[:50] + "..."
        
    # Get subtopics for additional context
    subtopics = metadata.get("topic_details", {}).get("subtopics", [])
    
    # Display analysis results
    st.markdown("<div class='analysis-container'>", unsafe_allow_html=True)
    
    # Header with source info
    st.markdown(f"""
    <h3>{analysis["source"]}</h3>
    <p style='color: #6B7280; font-size: 0.9rem;'>
        Analyzed on {formatted_date} • 
        Type: {analysis.get('input_type', {}).get('input_type', 'unknown').replace('_', ' ').title()} • 
        Subject: {analysis.get('input_type', {}).get('subject', 'general').title()}
    </p>
    """, unsafe_allow_html=True)
    
    # Sentiment overview
    st.markdown("<div class='section-title'>Sentiment Analysis</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 3])
    
    with col1:
        sentiment_emoji = "😀" if sentiment.get('sentiment') == "positive" else "😐" if sentiment.get('sentiment') == "neutral" else "😞"
        sentiment_text = sentiment.get('sentiment', 'neutral').capitalize()
        score = sentiment.get('score', 0)
        confidence = sentiment.get('confidence', 0)
        
        # Get the new sentiment trend if available
        sentiment_trend = sentiment.get('sentiment_trend', 'stable')
        trend_icon = "↗️" if sentiment_trend == "improving" else "↘️" if sentiment_trend == "worsening" else "↔️" if sentiment_trend == "stable" else "↕️"
        trend_text = sentiment_trend.capitalize()
        
        st.markdown(f"""
        <div style='text-align: center; padding: 1rem; background-color: white; border-radius: 8px; box-shadow: 0 1px 2px rgba(0, 0, 0, 0.05);'>
            <h1 style='font-size: 3rem; margin: 0;'>{sentiment_emoji}</h1>
            <h3 class='{sentiment_class}' style='margin: 0.5rem 0;'>{sentiment_text}</h3>
            <p style='margin: 0; font-size: 0.9rem;'>Score: {score:.2f} • Confidence: {confidence:.0%}</p>
            <p style='margin-top: 0.5rem; font-size: 0.9rem;'>Trend: {trend_icon} {trend_text}</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Key Sentiment Factors</h4>", unsafe_allow_html=True)
        
        # Display the rationale
        st.markdown(f"<p>{sentiment.get('rationale', 'No rationale provided.')}</p>", unsafe_allow_html=True)
        
        # Add the current context if available
        current_context = sentiment.get('current_context', '')
        if current_context:
            st.markdown("<h4>Current Context</h4>", unsafe_allow_html=True)
            st.markdown(f"<p><em>{current_context}</em></p>", unsafe_allow_html=True)
    
    # Content analysis
    st.markdown("<div class='section-title'>Detailed Analysis</div>", unsafe_allow_html=True)
    st.markdown(f"<p>{analysis['detailed_analysis']}</p>", unsafe_allow_html=True)
    
    # Metadata visualization
    st.markdown("<div class='section-title'>Content Classification</div>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("<h4>Topics</h4>", unsafe_allow_html=True)
        topics = metadata.get("topics", [])
        if topics:
            topic_html = "".join([f"<span class='badge badge-blue'>{topic}</span>" for topic in topics])
            st.markdown(topic_html, unsafe_allow_html=True)
        else:
            st.markdown("<p><em>No topics identified</em></p>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<h4>Regions</h4>", unsafe_allow_html=True)
        regions = metadata.get("regions", [])
        if regions:
            region_html = "".join([f"<span class='badge badge-green'>{region}</span>" for region in regions])
            st.markdown(region_html, unsafe_allow_html=True)
        else:
            st.markdown("<p><em>No regions identified</em></p>", unsafe_allow_html=True)
    
    with col3:
        st.markdown("<h4>Commodities/Products</h4>", unsafe_allow_html=True)
        commodities = metadata.get("commodities", [])
        if commodities:
            commodity_html = "".join([f"<span class='badge badge-yellow'>{commodity}</span>" for commodity in commodities])
            st.markdown(commodity_html, unsafe_allow_html=True)
        else:
            st.markdown("<p><em>No commodities identified</em></p>", unsafe_allow_html=True)
    
    # Entities and time periods
    if "entities" in metadata or "temporal_details" in metadata:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h4>Key Entities</h4>", unsafe_allow_html=True)
            entities = metadata.get("entities", [])
            if entities:
                entity_html = "".join([f"<span class='badge badge-gray'>{entity}</span>" for entity in entities])
                st.markdown(entity_html, unsafe_allow_html=True)
            else:
                st.markdown("<p><em>No entities identified</em></p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h4>Time Periods</h4>", unsafe_allow_html=True)
            time_periods = metadata.get("temporal_details", {}).get("time_period", [])
            if time_periods:
                time_html = "".join([f"<span class='badge badge-red'>{period}</span>" for period in time_periods])
                st.markdown(time_html, unsafe_allow_html=True)
                
                # Add specific dates if available
                specific_dates = metadata.get("temporal_details", {}).get("specific_dates", [])
                if specific_dates:
                    st.markdown("<h5>Key Dates</h5>", unsafe_allow_html=True)
                    dates_html = "".join([f"<span class='badge badge-blue'>{date}</span>" for date in specific_dates])
                    st.markdown(dates_html, unsafe_allow_html=True)
            else:
                st.markdown("<p><em>No time periods identified</em></p>", unsafe_allow_html=True)
    
    # Add trending topics and current events section
    event_context = metadata.get("event_context", {})
    topic_details = metadata.get("topic_details", {})
    
    if event_context.get("is_current_event") or topic_details.get("trending_topics"):
        st.markdown("<h4>Current Events & Trending Topics</h4>", unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("<h5>Trending Topics</h5>", unsafe_allow_html=True)
            trending_topics = topic_details.get("trending_topics", [])
            if trending_topics:
                trending_html = "".join([f"<span class='badge badge-yellow'>{topic}</span>" for topic in trending_topics])
                st.markdown(trending_html, unsafe_allow_html=True)
            else:
                st.markdown("<p><em>No trending topics identified</em></p>", unsafe_allow_html=True)
        
        with col2:
            st.markdown("<h5>Key Developments</h5>", unsafe_allow_html=True)
            key_developments = event_context.get("key_developments", [])
            if key_developments:
                for development in key_developments:
                    st.markdown(f"• {development}", unsafe_allow_html=True)
            else:
                st.markdown("<p><em>No key developments identified</em></p>", unsafe_allow_html=True)
        
        # If it's a current event with a timeline, show it
        if event_context.get("is_current_event") and event_context.get("event_timeline"):
            st.markdown("<h5>Event Timeline</h5>", unsafe_allow_html=True)
            timeline = event_context.get("event_timeline", [])
            for event in timeline:
                st.markdown(f"• {event}", unsafe_allow_html=True)
    
    # Online Data and Visualizations
    st.markdown("<div class='section-title'>Online Sentiment & Visualizations</div>", unsafe_allow_html=True)
    
    # Show a loading spinner while fetching online data
    with st.spinner(f"Fetching online data about {main_topic}..."):
        # Get online sentiment data
        online_data = get_online_sentiment(main_topic, subtopics, days_back=365)  # One year of data
        
        # Display API status
        api_status = online_data.get('api_status', {})
        
        st.markdown("<h4>Data Sources</h4>", unsafe_allow_html=True)
        
        # Show which APIs are available
        apis_col1, apis_col2, apis_col3, apis_col4 = st.columns(4)
        
        with apis_col1:
            twitter_status = "Connected" if api_status.get('twitter_api', False) else "Not Connected"
            twitter_color = "green" if api_status.get('twitter_api', False) else "gray"
            st.markdown(f"<p>Twitter/X API: <span style='color: {twitter_color};'>{twitter_status}</span></p>", unsafe_allow_html=True)
        
        with apis_col2:
            reddit_status = "Connected" if api_status.get('reddit_api', False) else "Not Connected"
            reddit_color = "green" if api_status.get('reddit_api', False) else "gray"
            st.markdown(f"<p>Reddit API: <span style='color: {reddit_color};'>{reddit_status}</span></p>", unsafe_allow_html=True)
        
        with apis_col3:
            news_status = "Connected" if api_status.get('news_api', False) else "Not Connected"
            news_color = "green" if api_status.get('news_api', False) else "gray"
            st.markdown(f"<p>News API: <span style='color: {news_color};'>{news_status}</span></p>", unsafe_allow_html=True)
        
        with apis_col4:
            web_status = "Active"
            web_color = "green"
            st.markdown(f"<p>Web Scraping: <span style='color: {web_color};'>{web_status}</span></p>", unsafe_allow_html=True)
        
        # Visualization tabs
        viz_tabs = st.tabs(["Regional Interest", "Temporal Trends", "Topic Popularity", "Keywords"])
        
        with viz_tabs[0]:  # 3D Globe
            st.subheader(f"Global Interest in {main_topic}")
            
            # Get globe data
            globe_data = online_data.get('global_data', {})
            
            # Create the globe visualization
            try:
                globe_fig = create_3d_globe_visualization(globe_data)
                st.plotly_chart(globe_fig, use_container_width=True)
                
                # Add a note about the data source
                st.caption("Data source: Global online interest analysis of major news and social media platforms.")
            except Exception as e:
                st.error(f"Error creating globe visualization: {e}")
        
        with viz_tabs[1]:  # Temporal Trends
            st.subheader(f"Interest Over Time: {main_topic}")
            
            # Time period selection
            time_period = st.radio(
                "Select time period:",
                ["Week", "Month", "Year", "All"],
                horizontal=True,
                index=2  # Default to Year
            )
            
            # Get historical data
            historical_data = online_data.get('historical_data', [])
            
            # Create the time series visualization
            try:
                time_fig = create_interest_over_time_chart(historical_data, period=time_period.lower())
                st.plotly_chart(time_fig, use_container_width=True)
                
                # Add a note about the data source
                st.caption("Data source: Temporal analysis of online mentions and engagement across multiple platforms.")
            except Exception as e:
                st.error(f"Error creating time series visualization: {e}")
        
        with viz_tabs[2]:  # Topic Popularity
            st.subheader("Topic and Subtopic Popularity")
            
            # Get keyword data
            keyword_data = online_data.get('keyword_data', {})
            
            # Create the topic popularity visualization
            try:
                topic_fig = create_topic_popularity_chart(keyword_data)
                st.plotly_chart(topic_fig, use_container_width=True)
                
                # Add a note about the data source
                st.caption("Data source: Analysis of topic mentions and engagement across news articles, social media, and online forums.")
            except Exception as e:
                st.error(f"Error creating topic popularity visualization: {e}")
        
        with viz_tabs[3]:  # Keywords
            st.subheader(f"Top Keywords for {main_topic}")
            
            # Get keyword data
            keyword_data = online_data.get('keyword_data', {})
            
            # Create the keyword visualization
            try:
                keyword_fig = create_keyword_chart(keyword_data)
                st.plotly_chart(keyword_fig, use_container_width=True)
                
                # Add a note about the data source
                st.caption("Data source: Frequency analysis of terms associated with the topic across various online platforms.")
            except Exception as e:
                st.error(f"Error creating keyword visualization: {e}")
        
        # Display a sample of sources
        st.markdown("<h4>Online Sources</h4>", unsafe_allow_html=True)
        
        sources_tabs = st.tabs(["News Articles", "Social Media", "Web Content"])
        
        with sources_tabs[0]:  # News
            news_data = online_data.get('sources', {}).get('news', [])
            if news_data:
                for i, article in enumerate(news_data[:3]):  # Show top 3
                    st.markdown(f"**{article.get('title', 'Untitled')}**")
                    st.markdown(f"Source: {article.get('source', 'Unknown')} • {article.get('published_at', '')}")
                    st.markdown(f"{article.get('content', '')[:200]}...")
                    if i < 2:
                        st.markdown("---")
            else:
                st.info("No news articles found.")
        
        with sources_tabs[1]:  # Social Media
            twitter_data = online_data.get('sources', {}).get('twitter', [])
            reddit_data = online_data.get('sources', {}).get('reddit', [])
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Twitter/X Posts**")
                if twitter_data:
                    for tweet in twitter_data[:3]:  # Show top 3
                        st.markdown(f"_{tweet.get('text', '')}_ ({tweet.get('retweets', 0)} retweets)")
                else:
                    st.info("No Twitter data available.")
            
            with col2:
                st.markdown("**Reddit Posts**")
                if reddit_data:
                    for post in reddit_data[:3]:  # Show top 3
                        st.markdown(f"**r/{post.get('subreddit', '')}**: {post.get('title', '')}")
                else:
                    st.info("No Reddit data available.")
        
        with sources_tabs[2]:  # Web
            web_data = online_data.get('sources', {}).get('web', [])
            if web_data:
                for content in web_data[:3]:  # Show top 3
                    st.markdown(f"**{content.get('title', 'Untitled')}**")
                    st.markdown(f"Source: [{content.get('url', '')}]({content.get('url', '')})")
                    st.markdown(f"{content.get('content', '')[:200]}...")
                    st.markdown("---")
            else:
                st.info("No web content found.")
                
        # Informational message about API status
        api_info_expander = st.expander("About API Data Sources")
        with api_info_expander:
            st.markdown("""
            ### API Integration Status
            
            This application can provide more accurate results when integrated with the following APIs:
            
            - **Twitter/X API**: Used to gather real-time social media sentiment
            - **NewsAPI**: Used to access current news articles
            - **Google Trends API**: Used to provide geographic interest data
            
            Contact the developer to integrate these APIs for comprehensive sentiment analysis.
            """)
    
    st.markdown("</div>", unsafe_allow_html=True)

# Display history if there's no current analysis
elif st.session_state.history and not st.session_state.processing:
    st.markdown("<h3>Previous Analyses</h3>", unsafe_allow_html=True)
    
    for idx, analysis in enumerate(reversed(st.session_state.history[-5:])):  # Show last 5 analyses
        sentiment = analysis["sentiment"]
        sentiment_class = f"sentiment-{sentiment.get('sentiment', 'neutral')}"
        
        st.markdown(f"""
        <div style='padding: 1rem; margin-bottom: 1rem; border: 1px solid #E5E7EB; border-radius: 8px; cursor: pointer;' 
             onclick="this.querySelector('button').click()">
            <h4>{analysis["source"]}</h4>
            <p><span class='{sentiment_class}'>{sentiment.get('sentiment', 'neutral').capitalize()}</span> • 
               Score: {sentiment.get('score', 0):.2f} • 
               {analysis.get('timestamp', '').split('T')[0]}</p>
            <button id='btn_{idx}' style='display:none'></button>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button(f"View Analysis", key=f"view_{idx}", use_container_width=True):
            st.session_state.current_analysis = analysis
            st.rerun()

# No content placeholder
if not st.session_state.current_analysis and not st.session_state.history and not st.session_state.processing:
    st.markdown("""
    <div class="no-results">
        <h3>Welcome to Sentimizer!</h3>
        <p>Enter text, a URL, or upload a file above to begin analyzing sentiment.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div style='text-align: center; margin-top: 3rem; padding-top: 1rem; border-top: 1px solid #E5E7EB;'>
    <p style='color: #6B7280; font-size: 0.8rem;'>
        Powered by OpenAI's GPT models • © 2025 Sentimizer
    </p>
</div>
""", unsafe_allow_html=True)