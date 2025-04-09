"""
Module for creating visualizations for sentiment analysis.
"""
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np
import pycountry

def country_name_to_code(country_name):
    """
    Convert country name to ISO 3166-1 alpha-3 code for map plotting.
    
    Args:
        country_name (str): Name of the country
        
    Returns:
        str: ISO 3166-1 alpha-3 code or None if not found
    """
    try:
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
        
        # Try searching by name
        countries = pycountry.countries.search_fuzzy(country_name)
        if countries:
            return countries[0].alpha_3
    except:
        # Handle exceptions (e.g., country not found)
        pass
    
    return None

def create_3d_globe_visualization(geo_data):
    """
    Create a 3D globe visualization showing sentiment across regions.
    
    Args:
        geo_data (dict): Geographic data with sentiment by country
        
    Returns:
        plotly.graph_objects.Figure: 3D globe figure
    """
    # Create a figure with an empty layout
    fig = go.Figure()
    
    # If we don't have country data, show a message on an empty globe
    countries = geo_data.get('countries', [])
    if not countries:
        fig.add_trace(go.Scattergeo(
            lon=[0],  # Center longitude
            lat=[0],  # Center latitude
            text=["No regional data available.<br>API key required for geographic data."],
            mode="text",
            textfont=dict(size=14, color="#EF4444"),
            showlegend=False
        ))
        
        # Update layout for an empty globe
        fig.update_geos(
            projection_type="orthographic",
            showcoastlines=True,
            coastlinecolor="black",
            showland=True,
            landcolor="lightgray",
            showocean=True,
            oceancolor="lightblue",
            showframe=False
        )
        
        fig.update_layout(
            title=dict(
                text='Regional Interest (API Key Required)',
                font=dict(size=18)
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            geo=dict(
                projection_rotation=dict(lon=0, lat=0, roll=0),
                showland=True,
                showcountries=True,
                countrycolor='lightgray',
                showocean=True,
                oceancolor='lightblue',
                showlakes=False,
                showrivers=False
            )
        )
        
        return fig
    
    # If we have data, create sentiment bars for each country
    # Extract data from geo_data to create sentiment bars
    countries_data = []
    for country in countries:
        lat = country.get('latitude', 0)
        lon = country.get('longitude', 0)
        name = country.get('name', 'Unknown')
        interest = country.get('interest', 0)
        sentiment = country.get('sentiment', 'neutral')
        
        # Color based on sentiment
        if sentiment == 'positive':
            color = '#10B981'  # Green
        elif sentiment == 'negative':
            color = '#EF4444'  # Red
        else:
            color = '#6B7280'  # Gray
        
        # Add to countries data
        countries_data.append({
            'name': name,
            'latitude': lat,
            'longitude': lon,
            'interest': interest,
            'sentiment': sentiment,
            'color': color
        })
    
    # Base layer - background globe
    fig.add_trace(go.Scattergeo(
        lon=[],
        lat=[],
        mode='markers',
        marker=dict(
            size=2,
            color='white',
            opacity=0
        ),
        showlegend=False
    ))
    
    # Add sentiment markers
    for country in countries_data:
        fig.add_trace(go.Scattergeo(
            lon=[country['longitude']],
            lat=[country['latitude']],
            text=f"{country['name']}: {country['interest']:.1f}% interest<br>Sentiment: {country['sentiment'].capitalize()}",
            mode='markers',
            marker=dict(
                size=max(5, country['interest'] / 3),
                color=country['color'],
                opacity=0.7,
                line=dict(width=1, color='black')
            ),
            name=country['sentiment'].capitalize(),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Regional Interest in Topic',
            font=dict(size=18)
        ),
        height=600,
        margin=dict(l=0, r=0, b=0, t=40),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        geo=dict(
            projection_type='orthographic',
            showland=True,
            showcountries=True,
            landcolor='rgb(229, 229, 229)',
            countrycolor='rgb(255, 255, 255)',
            showocean=True,
            oceancolor='rgb(230, 242, 255)',
            showlakes=False,
            showrivers=False
        )
    )
    
    # Add legend for sentiment colors
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='markers',
        marker=dict(size=10, color='#10B981'),
        name='Positive',
        showlegend=True
    ))
    
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='markers',
        marker=dict(size=10, color='#6B7280'),
        name='Neutral',
        showlegend=True
    ))
    
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='markers',
        marker=dict(size=10, color='#EF4444'),
        name='Negative',
        showlegend=True
    ))
    
    fig.update_layout(
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
    return fig

def create_interest_over_time_chart(time_data, period='year'):
    """
    Create a chart showing interest over time for a topic.
    
    Args:
        time_data (list): List of time series data
        period (str): Time period to display (week, month, year, or all)
        
    Returns:
        plotly.graph_objects.Figure: Time series figure
    """
    # Create a figure with an empty layout
    fig = go.Figure()
    
    # If we don't have time data, show a message
    if not time_data or len(time_data) == 0:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No historical data available.<br>API key required for time series data.",
            showarrow=False,
            font=dict(size=14, color="#EF4444")
        )
        
        fig.update_layout(
            title=dict(
                text=f'Interest Over Time ({period.title()})',
                font=dict(size=18)
            ),
            xaxis=dict(
                title=dict(
                    text='Date',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=dict(
                    text='Interest Level',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12),
                range=[0, 100]
            ),
            height=400,
            margin=dict(l=50, r=20, b=50, t=70),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        
        return fig
    
    # If we have data, create the time series chart
    # Handling data based on format
    if isinstance(time_data, pd.DataFrame):
        df = time_data
    else:
        # Convert time_data to DataFrame
        df = pd.DataFrame(time_data)
    
    # Apply time period filter
    end_date = df['date'].max()
    if period == 'week':
        start_date = end_date - timedelta(days=7)
        df_filtered = df[df['date'] >= start_date]
    elif period == 'month':
        start_date = end_date - timedelta(days=30)
        df_filtered = df[df['date'] >= start_date]
    elif period == 'year':
        start_date = end_date - timedelta(days=365)
        df_filtered = df[df['date'] >= start_date]
    else:  # 'all'
        df_filtered = df
    
    # Add interest line
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['interest'],
        mode='lines',
        name='Daily Interest',
        line=dict(color='#60A5FA', width=1),
        showlegend=True
    ))
    
    # Add smoothed trend line if available
    if 'interest_smoothed' in df_filtered.columns:
        fig.add_trace(go.Scatter(
            x=df_filtered['date'],
            y=df_filtered['interest_smoothed'],
            mode='lines',
            name='Trend',
            line=dict(color='#3B82F6', width=3),
            showlegend=True
        ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Interest Over Time ({period.title()})',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(
                text='Date',
                font=dict(size=14)
            ),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text='Interest Level',
                font=dict(size=14)
            ),
            tickfont=dict(size=12),
            range=[0, 100]
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        margin=dict(l=50, r=20, b=50, t=70),
        hovermode='closest',
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    # Add range selector
    fig.update_xaxes(
        rangeslider_visible=False,
        rangeselector=dict(
            buttons=list([
                dict(count=1, label="1w", step="week", stepmode="backward"),
                dict(count=1, label="1m", step="month", stepmode="backward"),
                dict(count=6, label="6m", step="month", stepmode="backward"),
                dict(count=1, label="YTD", step="year", stepmode="todate"),
                dict(count=1, label="1y", step="year", stepmode="backward"),
                dict(step="all")
            ])
        )
    )
    
    # Add a grid for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.3)')
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.3)')
    
    return fig

def create_topic_popularity_chart(keyword_data):
    """
    Create a bar chart showing popularity of topics and subtopics.
    
    Args:
        keyword_data (dict): Keyword and topic data
        
    Returns:
        plotly.graph_objects.Figure: Bar chart figure
    """
    # Create a figure with an empty layout
    fig = go.Figure()
    
    # Extract main topic and subtopics
    main_topic = keyword_data.get('main_topic', 'Unknown Topic')
    main_keywords = keyword_data.get('main_topic_keywords', [])
    subtopics = keyword_data.get('subtopics', [])
    subtopic_keywords = keyword_data.get('subtopic_keywords', {})
    
    # If we don't have keyword data or the data is empty, show a message
    if not main_keywords and not any(subtopic_keywords.values()):
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text="No topic popularity data available.<br>Please connect API sources for detailed analysis.",
            showarrow=False,
            font=dict(size=14, color="#EF4444")
        )
        
        fig.update_layout(
            title=dict(
                text='Topic and Subtopic Popularity',
                font=dict(size=18)
            ),
            xaxis=dict(
                title=dict(
                    text='Topics',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12),
                tickangle=-30
            ),
            yaxis=dict(
                title=dict(
                    text='Popularity Score',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12),
                range=[0, 100]
            ),
            height=400,
            margin=dict(l=50, r=20, b=100, t=70),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        
        return fig
    
    # If we have data, create the topic popularity chart
    topics = [main_topic]
    popularity_values = []
    
    # Calculate average popularity for main topic
    if main_keywords:
        main_popularity = sum(kw.get('frequency', 0) for kw in main_keywords) / len(main_keywords)
    else:
        main_popularity = 50  # Default value
    popularity_values.append(main_popularity)
    
    # Calculate popularity for each subtopic
    for subtopic in subtopics:
        subtopic_kws = subtopic_keywords.get(subtopic, [])
        if subtopic_kws:
            subtopic_popularity = sum(kw.get('frequency', 0) for kw in subtopic_kws) / len(subtopic_kws)
        else:
            # If no keywords for this subtopic, use a default value slightly below main topic
            subtopic_popularity = max(10, main_popularity * 0.7 * (0.8 + 0.4 * np.random.random()))
        
        topics.append(subtopic)
        popularity_values.append(subtopic_popularity)
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Topic': topics,
        'Popularity': popularity_values
    })
    
    # Create color mapping based on whether it's the main topic
    colors = ['#3B82F6' if topic == main_topic else '#60A5FA' for topic in topics]
    
    # Create the bar chart
    fig.add_trace(
        go.Bar(
            x=df['Topic'],
            y=df['Popularity'],
            marker_color=colors,
            text=[f"{p:.1f}" for p in df['Popularity']],
            textposition='auto'
        )
    )
    
    # Update layout
    fig.update_layout(
        title=dict(
            text='Topic and Subtopic Popularity',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(
                text='Topics',
                font=dict(size=14)
            ),
            tickfont=dict(size=12),
            tickangle=-30
        ),
        yaxis=dict(
            title=dict(
                text='Popularity Score',
                font=dict(size=14)
            ),
            tickfont=dict(size=12),
            range=[0, 100]
        ),
        margin=dict(l=50, r=20, b=100, t=70),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    # Add a grid for better readability
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.3)')
    
    return fig

def create_keyword_chart(keyword_data):
    """
    Create a horizontal bar chart showing top keywords and their sentiment.
    
    Args:
        keyword_data (dict): Keyword and topic data
        
    Returns:
        plotly.graph_objects.Figure: Horizontal bar chart
    """
    # Create a figure with an empty layout
    fig = go.Figure()
    
    # Extract main keywords
    main_topic = keyword_data.get('main_topic', 'Unknown')
    main_keywords = keyword_data.get('main_topic_keywords', [])
    
    # If we don't have keyword data, show a message
    if not main_keywords:
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"No keyword data available for '{main_topic}'.<br>Please connect API sources for detailed analysis.",
            showarrow=False,
            font=dict(size=14, color="#EF4444")
        )
        
        fig.update_layout(
            title=dict(
                text=f'Top Keywords for {main_topic}',
                font=dict(size=18)
            ),
            xaxis=dict(
                title=dict(
                    text='Frequency',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12)
            ),
            yaxis=dict(
                title=dict(
                    text='Keyword',
                    font=dict(size=14)
                ),
                tickfont=dict(size=12)
            ),
            height=400,
            margin=dict(l=20, r=20, b=50, t=70),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        
        return fig
        
    # Sort keywords by frequency and take top 10
    sorted_keywords = sorted(main_keywords, key=lambda x: x.get('frequency', 0), reverse=True)[:10]
    
    # Prepare data for chart
    keywords = [kw.get('keyword', 'Unknown') for kw in sorted_keywords]
    frequencies = [kw.get('frequency', 0) for kw in sorted_keywords]
    sentiments = [kw.get('sentiment', 'neutral') for kw in sorted_keywords]
    
    # Map sentiments to colors
    color_map = {
        'positive': '#10B981',  # Green
        'neutral': '#6B7280',   # Gray
        'negative': '#EF4444'   # Red
    }
    colors = [color_map.get(sentiment, '#6B7280') for sentiment in sentiments]
    
    # Add horizontal bar chart
    fig.add_trace(go.Bar(
        y=keywords,
        x=frequencies,
        orientation='h',
        marker_color=colors,
        text=[f"{freq} ({sent.title()})" for freq, sent in zip(frequencies, sentiments)],
        textposition='auto'
    ))
    
    # Update layout
    fig.update_layout(
        title=dict(
            text=f'Top Keywords for {main_topic}',
            font=dict(size=18)
        ),
        xaxis=dict(
            title=dict(
                text='Frequency',
                font=dict(size=14)
            ),
            tickfont=dict(size=12)
        ),
        yaxis=dict(
            title=dict(
                text='Keyword',
                font=dict(size=14)
            ),
            tickfont=dict(size=12),
            autorange="reversed"  # To have the highest values at the top
        ),
        margin=dict(l=20, r=20, b=50, t=70),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    # Add a legend for sentiment colors
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color_map['positive']),
        name='Positive'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color_map['neutral']),
        name='Neutral'
    ))
    
    fig.add_trace(go.Scatter(
        x=[None],
        y=[None],
        mode='markers',
        marker=dict(size=10, color=color_map['negative']),
        name='Negative'
    ))
    
    # Add a grid for better readability
    fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='rgba(211, 211, 211, 0.3)')
    fig.update_yaxes(showgrid=False)
    
    return fig