import streamlit as st
import pandas as pd
import io
import tempfile
import base64
import os
from datetime import datetime, timedelta
import json

from utils.text_processor import process_text_input, extract_text_from_pdf, extract_text_from_html
from utils.openai_client import summarize_text, analyze_sentiment, extract_metadata
from utils.data_manager import save_analysis, load_all_analyses, get_filtered_data
from utils.visualizations import (
    create_global_sentiment_map,
    create_sentiment_time_chart,
    create_topic_distribution_chart,
    create_commodity_price_chart
)

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

# Initialize session state for storing analyses
if 'analyses' not in st.session_state:
    st.session_state.analyses = load_all_analyses()
    
if 'selected_sources' not in st.session_state:
    st.session_state.selected_sources = []

# Title and description
st.title("📊 Sentiment Analysis Dashboard")
st.markdown("""
This dashboard allows you to analyze sentiment across multiple text sources.
Upload documents or paste URLs/text to analyze sentiment trends across time and geography.
""")

# Sidebar for input and filters
with st.sidebar:
    st.header("Input Source")
    
    input_type = st.radio(
        "Select input type:",
        ["Text", "PDF Upload", "Website URL"]
    )
    
    # Input based on selected type
    if input_type == "Text":
        text_input = st.text_area("Enter text to analyze:", height=200)
        process_btn = st.button("Process Text")
        
        if process_btn and text_input:
            with st.spinner("Processing text..."):
                processed_text = process_text_input(text_input)
                summary = summarize_text(processed_text)
                sentiment = analyze_sentiment(summary)
                metadata = extract_metadata(summary)
                
                # Save the analysis with timestamp
                timestamp = datetime.now()
                analysis_id = save_analysis(
                    text_input, 
                    processed_text,
                    summary, 
                    sentiment, 
                    metadata, 
                    timestamp, 
                    "direct_text"
                )
                
                st.session_state.analyses = load_all_analyses()
                st.success(f"Text processed successfully! Sentiment: {sentiment['sentiment']}")
    
    elif input_type == "PDF Upload":
        uploaded_file = st.file_uploader("Upload PDF", type="pdf")
        
        if uploaded_file:
            with st.spinner("Processing PDF..."):
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                    tmp_file.write(uploaded_file.getvalue())
                    temp_file_path = tmp_file.name
                
                try:
                    extracted_text = extract_text_from_pdf(temp_file_path)
                    
                    if extracted_text:
                        st.success("PDF text extracted!")
                        
                        process_pdf_btn = st.button("Process PDF Content")
                        
                        if process_pdf_btn:
                            with st.spinner("Analyzing content..."):
                                summary = summarize_text(extracted_text)
                                sentiment = analyze_sentiment(summary)
                                metadata = extract_metadata(summary)
                                
                                # Save the analysis with timestamp
                                timestamp = datetime.now()
                                analysis_id = save_analysis(
                                    uploaded_file.name, 
                                    extracted_text,
                                    summary, 
                                    sentiment, 
                                    metadata, 
                                    timestamp, 
                                    "pdf"
                                )
                                
                                st.session_state.analyses = load_all_analyses()
                                st.success(f"PDF processed successfully! Sentiment: {sentiment['sentiment']}")
                    else:
                        st.error("Couldn't extract text from the PDF. Please try another file.")
                
                finally:
                    # Clean up the temp file
                    if os.path.exists(temp_file_path):
                        os.unlink(temp_file_path)
    
    elif input_type == "Website URL":
        url_input = st.text_input("Enter website URL:")
        process_url_btn = st.button("Process URL")
        
        if process_url_btn and url_input:
            with st.spinner("Fetching and processing website content..."):
                try:
                    extracted_text = extract_text_from_html(url_input)
                    
                    if extracted_text:
                        summary = summarize_text(extracted_text)
                        sentiment = analyze_sentiment(summary)
                        metadata = extract_metadata(summary)
                        
                        # Save the analysis with timestamp
                        timestamp = datetime.now()
                        analysis_id = save_analysis(
                            url_input, 
                            extracted_text,
                            summary, 
                            sentiment, 
                            metadata, 
                            timestamp, 
                            "url"
                        )
                        
                        st.session_state.analyses = load_all_analyses()
                        st.success(f"Website content processed successfully! Sentiment: {sentiment['sentiment']}")
                    else:
                        st.error("Couldn't extract text from the website. Please check the URL.")
                except Exception as e:
                    st.error(f"Error processing website: {str(e)}")
    
    # Filters section
    st.header("Filters")
    
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
            "Select topics to include:",
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
            "Select regions to include:",
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
            "Select commodities to include:",
            options=sorted(list(all_commodities)),
            default=[]
        )
    else:
        selected_commodities = []
    
    # Sentiment filter
    st.subheader("Sentiment")
    selected_sentiments = st.multiselect(
        "Select sentiments to include:",
        options=["positive", "neutral", "negative"],
        default=["positive", "neutral", "negative"]
    )

# Main dashboard area
filtered_data = get_filtered_data(
    st.session_state.analyses,
    start_date,
    end_date,
    selected_topics,
    selected_regions,
    selected_commodities,
    selected_sentiments
)

# Display summary metrics
st.header("Summary Metrics")
col1, col2, col3, col4 = st.columns(4)

with col1:
    total_sources = len(filtered_data)
    st.metric("Total Sources", total_sources)

with col2:
    if filtered_data:
        positive_count = sum(1 for item in filtered_data.values() if item['sentiment']['sentiment'] == 'positive')
        positive_percentage = round((positive_count / total_sources) * 100) if total_sources else 0
        st.metric("Positive Sentiment", f"{positive_percentage}%")
    else:
        st.metric("Positive Sentiment", "0%")

with col3:
    if filtered_data:
        neutral_count = sum(1 for item in filtered_data.values() if item['sentiment']['sentiment'] == 'neutral')
        neutral_percentage = round((neutral_count / total_sources) * 100) if total_sources else 0
        st.metric("Neutral Sentiment", f"{neutral_percentage}%")
    else:
        st.metric("Neutral Sentiment", "0%")

with col4:
    if filtered_data:
        negative_count = sum(1 for item in filtered_data.values() if item['sentiment']['sentiment'] == 'negative')
        negative_percentage = round((negative_count / total_sources) * 100) if total_sources else 0
        st.metric("Negative Sentiment", f"{negative_percentage}%")
    else:
        st.metric("Negative Sentiment", "0%")

# Visualization tabs
st.header("Visualizations")
tab1, tab2, tab3, tab4 = st.tabs(["Global Map", "Time Series", "Topics", "Commodity Prices"])

with tab1:
    st.subheader("Global Sentiment Distribution")
    if filtered_data:
        map_fig = create_global_sentiment_map(filtered_data)
        st.plotly_chart(map_fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters. Please add some sources or adjust filters.")

with tab2:
    st.subheader("Sentiment Over Time")
    if filtered_data:
        time_fig = create_sentiment_time_chart(filtered_data)
        st.plotly_chart(time_fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters. Please add some sources or adjust filters.")

with tab3:
    st.subheader("Topic Distribution")
    if filtered_data:
        topic_fig = create_topic_distribution_chart(filtered_data)
        st.plotly_chart(topic_fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters. Please add some sources or adjust filters.")

with tab4:
    st.subheader("Commodity Price Correlation")
    if filtered_data:
        commodity_fig = create_commodity_price_chart(filtered_data)
        st.plotly_chart(commodity_fig, use_container_width=True)
    else:
        st.info("No data available for the selected filters. Please add some sources or adjust filters.")

# Source Explorer
st.header("Source Explorer")
if filtered_data:
    source_df = pd.DataFrame([
        {
            "ID": k,
            "Source": v.get('source', ''),
            "Type": v.get('source_type', ''),
            "Date": v.get('timestamp', ''),
            "Sentiment": v.get('sentiment', {}).get('sentiment', ''),
            "Topics": ", ".join(v.get('metadata', {}).get('topics', [])),
            "Regions": ", ".join(v.get('metadata', {}).get('regions', [])),
        }
        for k, v in filtered_data.items()
    ])
    
    selected_indices = st.multiselect(
        "Select sources to view details:",
        options=source_df['ID'].tolist(),
        default=[],
        format_func=lambda x: f"{filtered_data[x]['source']} ({filtered_data[x]['sentiment']['sentiment']})"
    )
    
    # Source details
    if selected_indices:
        st.session_state.selected_sources = selected_indices
        for source_id in selected_indices:
            source = filtered_data[source_id]
            
            with st.expander(f"{source['source']} ({source['sentiment']['sentiment']})", expanded=True):
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.subheader("Source Information")
                    st.write(f"**Type:** {source['source_type']}")
                    st.write(f"**Date:** {source['timestamp']}")
                    st.write(f"**Sentiment:** {source['sentiment']['sentiment']} (Score: {source['sentiment']['score']})")
                
                with col2:
                    st.subheader("Metadata")
                    st.write(f"**Topics:** {', '.join(source['metadata']['topics'])}")
                    st.write(f"**Regions:** {', '.join(source['metadata']['regions'])}")
                    st.write(f"**Commodities:** {', '.join(source['metadata']['commodities'])}")
                
                st.subheader("Summary")
                st.write(source['summary'])
                
                with st.expander("View Full Text"):
                    st.text_area("", source['text'], height=200)
    else:
        st.info("Select sources from the dropdown to view details.")
else:
    st.info("No sources available with the current filter settings.")

# Footer
st.markdown("---")
st.caption("Sentiment Analysis Dashboard | Created with Streamlit")
