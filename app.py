import streamlit as st
import pandas as pd
import io
import tempfile
import base64
import os
import time
import random
from datetime import datetime, timedelta
import json
import plotly.express as px
import plotly.graph_objects as go
from streamlit_lottie import st_lottie
import requests

from utils.text_processor import process_text_input, extract_text_from_pdf, extract_text_from_html
from utils.openai_client import summarize_text, analyze_sentiment, extract_metadata
from utils.data_manager import save_analysis, load_all_analyses, get_filtered_data
from utils.visualizations import (
    create_global_sentiment_map,
    create_sentiment_time_chart,
    create_topic_distribution_chart,
    create_commodity_price_chart
)

# Page configuration with custom theme
st.set_page_config(
    page_title="Sentimizer | Advanced Sentiment Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for improved aesthetics
st.markdown("""
<style>
    .main {
        background-color: #f8f9fa;
    }
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    h1, h2, h3 {
        color: #1E3A8A;
    }
    .sentiment-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: white;
        transition: transform 0.3s ease;
    }
    .sentiment-card:hover {
        transform: translateY(-5px);
    }
    .positive-label {
        color: #10B981;
        font-weight: bold;
    }
    .neutral-label {
        color: #6B7280;
        font-weight: bold;
    }
    .negative-label {
        color: #EF4444;
        font-weight: bold;
    }
    .mostly-positive-label {
        color: #34D399;
        font-weight: bold;
    }
    .mostly-negative-label {
        color: #F87171;
        font-weight: bold;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        text-align: center;
    }
    .stButton>button {
        background-color: #1E3A8A;
        color: white;
        border-radius: 5px;
        padding: 0.5rem 1rem;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #2563EB;
        border-color: #2563EB;
    }
    .section-divider {
        margin-top: 2rem;
        margin-bottom: 2rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .sidebar .sidebar-content {
        background-color: #1E3A8A;
    }
    .loader {
        border: 8px solid #f3f3f3;
        border-top: 8px solid #3498db;
        border-radius: 50%;
        width: 50px;
        height: 50px;
        animation: spin 1s linear infinite;
        margin: 20px auto;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    .source-card {
        border-left: 4px solid #3B82F6;
        padding-left: 1rem;
        margin-bottom: 1rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 1rem;
    }
    .stTabs [data-baseweb="tab"] {
        height: 4rem;
        white-space: pre-wrap;
        background-color: white;
        border-radius: 5px 5px 0 0;
        padding: 1rem;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    }
    .stTabs [aria-selected="true"] {
        background-color: #EFF6FF;
        border-bottom: 2px solid #3B82F6;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analyses' not in st.session_state:
    st.session_state.analyses = load_all_analyses()
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'input_type' not in st.session_state:
    st.session_state.input_type = None
if 'input_content' not in st.session_state:
    st.session_state.input_content = None
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'progress' not in st.session_state:
    st.session_state.progress = 0
if 'show_animation' not in st.session_state:
    st.session_state.show_animation = True

# Functions for page navigation
def go_to_home():
    st.session_state.page = 'home'
    st.session_state.show_animation = True

def go_to_input():
    st.session_state.page = 'input'
    
def go_to_dashboard():
    st.session_state.page = 'dashboard'

def go_to_analysis():
    st.session_state.page = 'analysis'
    st.session_state.processing = False
    st.session_state.progress = 0

def load_lottie_url(url: str):
    """Load a Lottie animation from URL"""
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

# Get trending topics (simulated for demonstration)
def get_trending_topics():
    trending_topics = [
        {"topic": "Global Economy", "sentiment": "mostly_negative", "count": random.randint(150, 300)},
        {"topic": "Climate Change", "sentiment": "negative", "count": random.randint(150, 300)},
        {"topic": "Tech Innovation", "sentiment": "positive", "count": random.randint(150, 300)},
        {"topic": "Healthcare", "sentiment": "neutral", "count": random.randint(150, 300)},
        {"topic": "Oil Prices", "sentiment": "mostly_negative", "count": random.randint(150, 300)},
        {"topic": "Renewable Energy", "sentiment": "mostly_positive", "count": random.randint(150, 300)},
        {"topic": "Education", "sentiment": "neutral", "count": random.randint(150, 300)},
        {"topic": "Cryptocurrency", "sentiment": "mostly_positive", "count": random.randint(150, 300)}
    ]
    return trending_topics

def sentiment_to_emoji(sentiment):
    """Convert sentiment label to emoji"""
    sentiment_emojis = {
        "positive": "😀",
        "mostly_positive": "🙂",
        "neutral": "😐",
        "mostly_negative": "🙁",
        "negative": "😞"
    }
    return sentiment_emojis.get(sentiment, "😐")

def sentiment_to_color(sentiment):
    """Convert sentiment label to color"""
    sentiment_colors = {
        "positive": "#10B981",
        "mostly_positive": "#34D399",
        "neutral": "#6B7280",
        "mostly_negative": "#F87171",
        "negative": "#EF4444"
    }
    return sentiment_colors.get(sentiment, "#6B7280")

def sentiment_to_class(sentiment):
    """Convert sentiment label to CSS class"""
    sentiment_classes = {
        "positive": "positive-label",
        "mostly_positive": "mostly-positive-label",
        "neutral": "neutral-label",
        "mostly_negative": "mostly-negative-label",
        "negative": "negative-label"
    }
    return sentiment_classes.get(sentiment, "neutral-label")

def process_input():
    """Process the input content and perform sentiment analysis"""
    st.session_state.processing = True
    
    progress_text = "Processing your content. Please wait..."
    progress_bar = st.progress(0, text=progress_text)
    
    # Step 1: Extract text based on input type
    st.session_state.progress = 10
    progress_bar.progress(st.session_state.progress, text="Step 1/5: Extracting text...")
    time.sleep(0.5)
    
    try:
        if st.session_state.input_type == "text":
            processed_text = process_text_input(st.session_state.input_content)
            source = "Direct Text Input"
            source_type = "direct_text"
        elif st.session_state.input_type == "pdf":
            # Save PDF temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(st.session_state.input_content.getvalue())
                temp_file_path = tmp_file.name
            
            processed_text = extract_text_from_pdf(temp_file_path)
            source = st.session_state.input_content.name
            source_type = "pdf"
            
            # Clean up temp file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)
                
        elif st.session_state.input_type == "url":
            processed_text = extract_text_from_html(st.session_state.input_content)
            source = st.session_state.input_content
            source_type = "url"
            
        if not processed_text:
            st.error("Failed to extract text from the provided source.")
            st.session_state.processing = False
            return
            
        # Step 2: Generate summary
        st.session_state.progress = 30
        progress_bar.progress(st.session_state.progress, text="Step 2/5: Generating comprehensive summary...")
        time.sleep(0.5)
        
        summary = summarize_text(processed_text)
        
        # Step 3: Analyze sentiment
        st.session_state.progress = 50
        progress_bar.progress(st.session_state.progress, text="Step 3/5: Analyzing sentiment patterns...")
        time.sleep(0.5)
        
        sentiment = analyze_sentiment(summary)
        
        # Step 4: Extract metadata
        st.session_state.progress = 70
        progress_bar.progress(st.session_state.progress, text="Step 4/5: Extracting metadata and classifications...")
        time.sleep(0.5)
        
        metadata = extract_metadata(summary)
        
        # Step 5: Save analysis
        st.session_state.progress = 90
        progress_bar.progress(st.session_state.progress, text="Step 5/5: Saving analysis results...")
        time.sleep(0.5)
        
        # Save with timestamp
        timestamp = datetime.now()
        analysis_id = save_analysis(
            source,
            processed_text,
            summary,
            sentiment,
            metadata,
            timestamp,
            source_type
        )
        
        # Update analyses in session state
        st.session_state.analyses = load_all_analyses()
        
        # Store the current analysis result
        st.session_state.analysis_result = {
            "id": analysis_id,
            "source": source,
            "source_type": source_type,
            "summary": summary,
            "sentiment": sentiment,
            "metadata": metadata,
            "timestamp": timestamp.isoformat()
        }
        
        st.session_state.progress = 100
        progress_bar.progress(st.session_state.progress, text="Analysis complete!")
        time.sleep(1)
        
        # Navigate to analysis results
        go_to_analysis()
        
    except Exception as e:
        st.error(f"An error occurred during processing: {str(e)}")
        st.session_state.processing = False

# HOME PAGE
if st.session_state.page == 'home':
    # Header with animation
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("<h1 style='font-size:3.5rem; margin-top:4rem;'>Sentimizer</h1>", unsafe_allow_html=True)
        st.markdown("<p style='font-size:1.5rem; color:#4B5563;'>Unlock the emotional pulse of text through advanced AI sentiment analysis</p>", unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
        
        col_a, col_b, col_c = st.columns([1, 1, 2])
        with col_a:
            if st.button("Start Analyzing", key="start_btn", use_container_width=True):
                go_to_input()
        with col_b:
            if st.button("View Dashboard", key="dashboard_btn", use_container_width=True):
                go_to_dashboard()
    
    with col2:
        if st.session_state.show_animation:
            # Load sentiment analysis animation
            lottie_url = "https://assets6.lottiefiles.com/packages/lf20_mhdn5srb.json"
            lottie_json = load_lottie_url(lottie_url)
            if lottie_json:
                st_lottie(lottie_json, speed=1, height=300, key="sentiment_anim")
            else:
                st.image("https://via.placeholder.com/400x300?text=Sentiment+Analysis+Visualization", width=300)
    
    # Trending topics section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>Trending Topics</h2>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280;'>Explore the sentiment landscape of current popular topics</p>", unsafe_allow_html=True)
    
    # Generate trending topics in a grid
    trending_topics = get_trending_topics()
    
    # Display trending topics in 4 columns
    cols = st.columns(4)
    for i, topic in enumerate(trending_topics):
        with cols[i % 4]:
            sentiment_class = sentiment_to_class(topic["sentiment"])
            emoji = sentiment_to_emoji(topic["sentiment"])
            color = sentiment_to_color(topic["sentiment"])
            
            st.markdown(f"""
            <div class='sentiment-card' style='border-top: 3px solid {color};'>
                <h3>{topic["topic"]}</h3>
                <p>Sentiment: <span class='{sentiment_class}'>{topic["sentiment"].replace('_', ' ').title()} {emoji}</span></p>
                <p>Mentions: {topic["count"]}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Features section
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.markdown("<h2>Key Features</h2>", unsafe_allow_html=True)
    
    feat_col1, feat_col2, feat_col3 = st.columns(3)
    
    with feat_col1:
        st.markdown("""
        <div class='sentiment-card'>
            <h3>🌍 Global Sentiment Mapping</h3>
            <p>Visualize sentiment distribution across different geographical regions with interactive heatmaps and 3D globe visualizations.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col2:
        st.markdown("""
        <div class='sentiment-card'>
            <h3>📈 Temporal Analysis</h3>
            <p>Track sentiment changes over time with detailed trend analysis and forecasting capabilities.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with feat_col3:
        st.markdown("""
        <div class='sentiment-card'>
            <h3>🧠 Advanced AI Analysis</h3>
            <p>Leverage state-of-the-art language models to extract nuanced sentiment insights, topics, and metadata.</p>
        </div>
        """, unsafe_allow_html=True)

# INPUT PAGE
elif st.session_state.page == 'input':
    # Header
    st.markdown("<h1>What would you like to analyze?</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280; margin-bottom:2rem;'>Choose your input method to begin sentiment analysis</p>", unsafe_allow_html=True)
    
    # Input method selection with icons
    input_col1, input_col2, input_col3 = st.columns(3)
    
    with input_col1:
        st.markdown("""
        <div class='sentiment-card' style='text-align:center;'>
            <h3>📝 Text Input</h3>
            <p>Analyze sentiment from direct text entry</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Text Input", key="text_btn"):
            st.session_state.input_type = "text"
    
    with input_col2:
        st.markdown("""
        <div class='sentiment-card' style='text-align:center;'>
            <h3>📄 PDF Document</h3>
            <p>Extract and analyze text from PDF files</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select PDF Upload", key="pdf_btn"):
            st.session_state.input_type = "pdf"
    
    with input_col3:
        st.markdown("""
        <div class='sentiment-card' style='text-align:center;'>
            <h3>🌐 Website URL</h3>
            <p>Analyze content from any web page</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("Select Website URL", key="url_btn"):
            st.session_state.input_type = "url"
    
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    
    # Display the appropriate input form based on selection
    if st.session_state.input_type:
        if st.session_state.input_type == "text":
            st.subheader("Enter Text to Analyze")
            text_input = st.text_area(
                "Paste your text here:",
                height=250,
                placeholder="Enter the text you want to analyze for sentiment..."
            )
            
            if st.button("Analyze Text", key="analyze_text_btn", disabled=not text_input):
                st.session_state.input_content = text_input
                process_input()
        
        elif st.session_state.input_type == "pdf":
            st.subheader("Upload PDF Document")
            uploaded_file = st.file_uploader("Choose a PDF file:", type=["pdf"])
            
            if uploaded_file:
                st.success(f"File uploaded: {uploaded_file.name}")
                
                if st.button("Analyze PDF", key="analyze_pdf_btn"):
                    st.session_state.input_content = uploaded_file
                    process_input()
        
        elif st.session_state.input_type == "url":
            st.subheader("Enter Website URL")
            url_input = st.text_input(
                "Website URL:",
                placeholder="https://example.com/article"
            )
            
            if st.button("Analyze Website", key="analyze_url_btn", disabled=not url_input):
                st.session_state.input_content = url_input
                process_input()
    
    # Processing state
    if st.session_state.processing:
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Processing Your Content")
        
        # This will be handled by the process_input function
    
    # Navigation buttons
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    if st.button("← Back to Home", key="back_to_home_btn"):
        go_to_home()

# ANALYSIS RESULTS PAGE
elif st.session_state.page == 'analysis':
    if st.session_state.analysis_result:
        result = st.session_state.analysis_result
        
        # Header with source info
        st.markdown(f"<h1>Analysis Results</h1>", unsafe_allow_html=True)
        
        source_type_icon = {"pdf": "📄", "url": "🌐", "direct_text": "📝"}.get(result['source_type'], "📝")
        st.markdown(f"<p style='color:#6B7280; font-size:1.2rem;'>{source_type_icon} Source: {result['source']}</p>", unsafe_allow_html=True)
        st.markdown(f"<p style='color:#6B7280;'>Analyzed on: {result['timestamp']}</p>", unsafe_allow_html=True)
        
        # Sentiment result with visual indicator
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        
        sentiment_label = result['sentiment']['sentiment']
        sentiment_score = result['sentiment']['score']
        
        # Convert numerical score to more descriptive sentiment
        if sentiment_label == "positive" and sentiment_score < 0.7:
            display_sentiment = "mostly_positive"
        elif sentiment_label == "negative" and sentiment_score > -0.7:
            display_sentiment = "mostly_negative"
        else:
            display_sentiment = sentiment_label
            
        sentiment_emoji = sentiment_to_emoji(display_sentiment)
        sentiment_class = sentiment_to_class(display_sentiment)
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.markdown(f"""
            <div style='background-color:white; padding:20px; border-radius:10px; text-align:center; box-shadow:0 4px 6px rgba(0,0,0,0.1);'>
                <h1 style='font-size:5rem; margin:0;'>{sentiment_emoji}</h1>
                <h3 class='{sentiment_class}' style='margin:0; text-transform:capitalize;'>{display_sentiment.replace('_', ' ')}</h3>
                <p>Score: {sentiment_score:.2f}</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            # Summary of the analyzed content
            st.subheader("Summary")
            st.markdown(f"<div style='background-color:white; padding:20px; border-radius:10px; box-shadow:0 4px 6px rgba(0,0,0,0.1);'>{result['summary']}</div>", unsafe_allow_html=True)
        
        # Metadata section
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        st.subheader("Content Classification")
        
        meta_col1, meta_col2, meta_col3 = st.columns(3)
        
        with meta_col1:
            st.markdown("<h4>📊 Topics</h4>", unsafe_allow_html=True)
            topics = result['metadata']['topics']
            if topics:
                for topic in topics:
                    st.markdown(f"<div style='background-color:#EFF6FF; padding:8px 12px; border-radius:5px; margin-bottom:5px;'>{topic}</div>", unsafe_allow_html=True)
            else:
                st.info("No topics identified")
        
        with meta_col2:
            st.markdown("<h4>🌎 Regions</h4>", unsafe_allow_html=True)
            regions = result['metadata']['regions']
            if regions:
                for region in regions:
                    st.markdown(f"<div style='background-color:#ECFDF5; padding:8px 12px; border-radius:5px; margin-bottom:5px;'>{region}</div>", unsafe_allow_html=True)
            else:
                st.info("No regions identified")
        
        with meta_col3:
            st.markdown("<h4>💹 Commodities</h4>", unsafe_allow_html=True)
            commodities = result['metadata']['commodities']
            if commodities:
                for commodity in commodities:
                    st.markdown(f"<div style='background-color:#FEF3C7; padding:8px 12px; border-radius:5px; margin-bottom:5px;'>{commodity}</div>", unsafe_allow_html=True)
            else:
                st.info("No commodities identified")
        
        # Action buttons
        st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            if st.button("← New Analysis", key="new_analysis_btn"):
                go_to_input()
        
        with col2:
            if st.button("View Dashboard →", key="view_dashboard_btn"):
                go_to_dashboard()
    
    else:
        st.error("No analysis results found. Please submit content for analysis.")
        if st.button("Go to Input Page", key="no_result_input_btn"):
            go_to_input()

# DASHBOARD PAGE
elif st.session_state.page == 'dashboard':
    # Dashboard header
    st.markdown("<h1>Sentiment Analysis Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<p style='color:#6B7280;'>Explore sentiment trends across time, topics, and regions</p>", unsafe_allow_html=True)
    
    # Sidebar for filters
    with st.sidebar:
        st.sidebar.title("Filters")
        
        # Date range filter
        st.subheader("Date Range")
        today = datetime.now()
        start_date = st.date_input(
            "Start date",
            today - timedelta(days=30)
        )
        end_date = st.date_input(
            "End date",
            today
        )
        
        # Topic filter
        st.subheader("Topics")
        if st.session_state.analyses:
            all_topics = set()
            for analysis in st.session_state.analyses.values():
                if 'metadata' in analysis and 'topics' in analysis['metadata']:
                    all_topics.update(analysis['metadata']['topics'])
            
            selected_topics = st.multiselect(
                "Select topics:",
                options=sorted(list(all_topics)),
                default=[]
            )
        else:
            selected_topics = []
        
        # Region filter
        st.subheader("Regions")
        if st.session_state.analyses:
            all_regions = set()
            for analysis in st.session_state.analyses.values():
                if 'metadata' in analysis and 'regions' in analysis['metadata']:
                    all_regions.update(analysis['metadata']['regions'])
            
            selected_regions = st.multiselect(
                "Select regions:",
                options=sorted(list(all_regions)),
                default=[]
            )
        else:
            selected_regions = []
        
        # Commodity filter
        st.subheader("Commodities")
        if st.session_state.analyses:
            all_commodities = set()
            for analysis in st.session_state.analyses.values():
                if 'metadata' in analysis and 'commodities' in analysis['metadata']:
                    all_commodities.update(analysis['metadata']['commodities'])
            
            selected_commodities = st.multiselect(
                "Select commodities:",
                options=sorted(list(all_commodities)),
                default=[]
            )
        else:
            selected_commodities = []
        
        # Sentiment filter
        st.subheader("Sentiment")
        selected_sentiments = st.multiselect(
            "Select sentiments:",
            options=["positive", "neutral", "negative"],
            default=["positive", "neutral", "negative"]
        )
        
        # Apply filters button
        apply_filters = st.button("Apply Filters", key="apply_filters_btn")
    
    # Get filtered data based on selections
    filtered_data = get_filtered_data(
        st.session_state.analyses,
        start_date,
        end_date,
        selected_topics,
        selected_regions,
        selected_commodities,
        selected_sentiments
    )
    
    # Display summary metrics in animated cards
    st.subheader("Summary Metrics")
    
    metric_cols = st.columns(4)
    
    with metric_cols[0]:
        total_sources = len(filtered_data)
        st.markdown(f"""
        <div class='metric-card'>
            <h3 style='margin:0; color:#6B7280;'>Total Sources</h3>
            <p style='font-size:2rem; font-weight:bold; margin:5px 0;'>{total_sources}</p>
        </div>
        """, unsafe_allow_html=True)
    
    sentiment_counts = {
        "positive": 0,
        "neutral": 0,
        "negative": 0
    }
    
    if filtered_data:
        for item in filtered_data.values():
            sentiment = item['sentiment']['sentiment']
            sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
    
    # Display sentiment percentages
    for i, (sentiment, count) in enumerate(sentiment_counts.items(), start=1):
        percentage = round((count / total_sources) * 100) if total_sources else 0
        sentiment_class = sentiment_to_class(sentiment)
        
        with metric_cols[i]:
            st.markdown(f"""
            <div class='metric-card'>
                <h3 style='margin:0; color:#6B7280;'>{sentiment.capitalize()} Sentiment</h3>
                <p style='font-size:2rem; font-weight:bold; margin:5px 0;' class='{sentiment_class}'>{percentage}%</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Visualization tabs with enhanced styling
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Interactive Visualizations")
    
    tabs = st.tabs(["🌎 Global Map", "📈 Time Series", "🏷️ Topics", "💹 Commodities"])
    
    with tabs[0]:
        st.markdown("<h3>Global Sentiment Distribution</h3>", unsafe_allow_html=True)
        st.markdown("<p>Geographical distribution of sentiment across different regions</p>", unsafe_allow_html=True)
        
        if filtered_data:
            map_fig = create_global_sentiment_map(filtered_data)
            st.plotly_chart(map_fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No geographical data available. Add content with regional information for visualization.")
    
    with tabs[1]:
        st.markdown("<h3>Sentiment Trends Over Time</h3>", unsafe_allow_html=True)
        st.markdown("<p>How sentiment has changed over the selected time period</p>", unsafe_allow_html=True)
        
        if filtered_data:
            time_fig = create_sentiment_time_chart(filtered_data)
            st.plotly_chart(time_fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No time series data available. Add more content for temporal analysis.")
    
    with tabs[2]:
        st.markdown("<h3>Topic Distribution & Sentiment</h3>", unsafe_allow_html=True)
        st.markdown("<p>Most common topics and their associated sentiment scores</p>", unsafe_allow_html=True)
        
        if filtered_data:
            topic_fig = create_topic_distribution_chart(filtered_data)
            st.plotly_chart(topic_fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No topic data available. Add content with topic classifications for visualization.")
    
    with tabs[3]:
        st.markdown("<h3>Commodity Price & Sentiment Correlation</h3>", unsafe_allow_html=True)
        st.markdown("<p>Relationship between commodity prices and sentiment indicators</p>", unsafe_allow_html=True)
        
        if filtered_data:
            commodity_fig = create_commodity_price_chart(filtered_data)
            st.plotly_chart(commodity_fig, use_container_width=True, config={'displayModeBar': False})
        else:
            st.info("No commodity data available. Add content with commodity mentions for visualization.")
    
    # Source Explorer with enhanced styling
    st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
    st.subheader("Source Explorer")
    
    if filtered_data:
        # Create dataframe for source selection
        source_df = pd.DataFrame([
            {
                "ID": k,
                "Source": v.get('source', ''),
                "Type": v.get('source_type', ''),
                "Date": v.get('timestamp', '').split('T')[0] if 'timestamp' in v else '',
                "Sentiment": v.get('sentiment', {}).get('sentiment', ''),
                "Topics": ", ".join(v.get('metadata', {}).get('topics', [])),
                "Regions": ", ".join(v.get('metadata', {}).get('regions', [])),
            }
            for k, v in filtered_data.items()
        ])
        
        # Source selection
        selected_indices = st.multiselect(
            "Select sources to view details:",
            options=source_df['ID'].tolist(),
            default=[],
            format_func=lambda x: f"{filtered_data[x]['source']} ({filtered_data[x]['sentiment']['sentiment'].capitalize()}) - {filtered_data[x]['timestamp'].split('T')[0] if 'timestamp' in filtered_data[x] else ''}"
        )
        
        # Display source details
        if selected_indices:
            st.session_state.selected_sources = selected_indices
            
            for source_id in selected_indices:
                source = filtered_data[source_id]
                
                # Get sentiment class for styling
                sentiment_class = sentiment_to_class(source['sentiment']['sentiment'])
                sentiment_emoji = sentiment_to_emoji(source['sentiment']['sentiment'])
                
                # Source card with enhanced styling
                st.markdown(f"""
                <div style='background-color:white; padding:20px; border-radius:10px; margin-bottom:20px; box-shadow:0 4px 6px rgba(0,0,0,0.1);'>
                    <h3>{source['source']} <span class='{sentiment_class}'>{sentiment_emoji}</span></h3>
                    <p style='color:#6B7280;'>Date: {source['timestamp'].split('T')[0] if 'timestamp' in source else 'N/A'} | Type: {source['source_type']}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Details in columns
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    # Summary
                    st.markdown("<h4>Summary</h4>", unsafe_allow_html=True)
                    st.markdown(f"<div style='background-color:#F9FAFB; padding:15px; border-radius:5px;'>{source['summary']}</div>", unsafe_allow_html=True)
                
                with col2:
                    # Metadata and sentiment
                    st.markdown("<h4>Analysis Details</h4>", unsafe_allow_html=True)
                    
                    # Sentiment score with visual indicator
                    score = source['sentiment']['score']
                    sentiment = source['sentiment']['sentiment']
                    
                    # Create a simple gauge for sentiment score
                    gauge_value = (score + 1) / 2  # Normalize from -1,1 to 0,1
                    
                    st.markdown(f"""
                    <p>Sentiment: <span class='{sentiment_class}'>{sentiment.capitalize()}</span></p>
                    <p>Score: {score:.2f}</p>
                    """, unsafe_allow_html=True)
                    
                    # Create mini progress bar as sentiment gauge
                    st.progress(gauge_value)
                    
                    # Topics, regions and commodities
                    st.markdown("<h5>Topics</h5>", unsafe_allow_html=True)
                    if source['metadata']['topics']:
                        st.markdown(", ".join([f"<span style='background-color:#EFF6FF; padding:2px 6px; border-radius:3px; margin-right:5px;'>{topic}</span>" for topic in source['metadata']['topics']]), unsafe_allow_html=True)
                    else:
                        st.markdown("<em>None identified</em>", unsafe_allow_html=True)
                    
                    st.markdown("<h5>Regions</h5>", unsafe_allow_html=True)
                    if source['metadata']['regions']:
                        st.markdown(", ".join([f"<span style='background-color:#ECFDF5; padding:2px 6px; border-radius:3px; margin-right:5px;'>{region}</span>" for region in source['metadata']['regions']]), unsafe_allow_html=True)
                    else:
                        st.markdown("<em>None identified</em>", unsafe_allow_html=True)
                    
                    st.markdown("<h5>Commodities</h5>", unsafe_allow_html=True)
                    if source['metadata']['commodities']:
                        st.markdown(", ".join([f"<span style='background-color:#FEF3C7; padding:2px 6px; border-radius:3px; margin-right:5px;'>{commodity}</span>" for commodity in source['metadata']['commodities']]), unsafe_allow_html=True)
                    else:
                        st.markdown("<em>None identified</em>", unsafe_allow_html=True)
                
                # Full text expander
                with st.expander("View Full Text"):
                    st.text_area("", source['text'], height=200)
        else:
            st.info("Select sources from the dropdown to view detailed analysis.")
    else:
        st.info("No sources found with the current filter settings. Try adjusting your filters or add new content for analysis.")
    
    # Navigation button
    st.markdown("<div style='margin-top:2rem;'></div>", unsafe_allow_html=True)
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("← Back to Home", key="dash_back_btn"):
            go_to_home()
    with col2:
        if st.button("Add New Analysis", key="add_new_analysis_btn"):
            go_to_input()

# Footer
st.markdown("<div class='section-divider'></div>", unsafe_allow_html=True)
st.caption("Sentimizer | Advanced AI-Powered Sentiment Analysis Dashboard")
