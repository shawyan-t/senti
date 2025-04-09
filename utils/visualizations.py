"""
Module for creating visualizations for sentiment analysis.
"""
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import json
import numpy as np
import pycountry

# Helper function to convert country name to ISO code
def country_name_to_code(country_name):
    """
    Convert country name to ISO 3166-1 alpha-3 code for map plotting.
    
    Args:
        country_name (str): Name of the country
        
    Returns:
        str: ISO 3166-1 alpha-3 code or None if not found
    """
    try:
        # Try direct lookup
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
        
        # Try searching by name
        countries = pycountry.countries.search_fuzzy(country_name)
        if countries:
            return countries[0].alpha_3
    except:
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
    # Extract the data
    main_topic = geo_data.get('main_topic', '')
    countries_data = geo_data.get('main_topic_data', [])
    
    # Prepare DataFrame
    df = pd.DataFrame(countries_data)
    
    # Create the 3D globe figure
    fig = go.Figure()
    
    # Add choropleth map layer
    fig.add_trace(go.Choropleth(
        locations=df['country_code'],
        z=df['interest'],
        text=df['country'],
        colorscale='Plasma',
        autocolorscale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar=dict(
            title=dict(
                text='Interest Level',
                font=dict(size=14)
            ),
            tickfont=dict(size=12)
        ),
        name=f'{main_topic} Interest'
    ))
    
    # Update the layout for 3D globe projection
    fig.update_layout(
        title=dict(
            text=f'Global Interest in {main_topic}',
            font=dict(size=20)
        ),
        geo=dict(
            projection_type='orthographic',
            showland=True,
            landcolor='rgb(217, 217, 217)',
            showocean=True,
            oceancolor='rgb(204, 230, 255)',
            showlakes=True,
            lakecolor='rgb(204, 230, 255)',
            showcountries=True,
            countrycolor='rgb(80, 80, 80)',
            countrywidth=0.5,
            showcoastlines=True,
            coastlinecolor='rgb(80, 80, 80)',
            coastlinewidth=0.5
        ),
        width=800,
        height=600,
        margin=dict(t=50, b=0, l=0, r=0),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(0, 0, 0, 0)'
    )
    
    # Add interactive controls for rotating the globe
    fig.update_layout(
        updatemenus=[{
            'buttons': [
                {
                    'args': [{'geo.projection.rotation.lon': -180}],
                    'label': 'Americas',
                    'method': 'relayout'
                },
                {
                    'args': [{'geo.projection.rotation.lon': 0}],
                    'label': 'Europe/Africa',
                    'method': 'relayout'
                },
                {
                    'args': [{'geo.projection.rotation.lon': 90}],
                    'label': 'Asia/Australia',
                    'method': 'relayout'
                }
            ],
            'direction': 'down',
            'showactive': True,
            'x': 0.05,
            'y': 0.05
        }]
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
    # Convert to DataFrame
    df = pd.DataFrame(time_data)
    
    # Convert date strings to datetime
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter based on period
    end_date = df['date'].max()
    if period == 'week':
        start_date = end_date - timedelta(days=7)
    elif period == 'month':
        start_date = end_date - timedelta(days=30)
    elif period == 'year':
        start_date = end_date - timedelta(days=365)
    else:  # 'all'
        start_date = df['date'].min()
    
    df_filtered = df[(df['date'] >= start_date) & (df['date'] <= end_date)]
    
    # Create the figure
    fig = go.Figure()
    
    # Add the main line
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['interest_smoothed'],
        mode='lines',
        name='Trend',
        line=dict(width=3, color='#1E40AF')
    ))
    
    # Add the raw data as scatter points
    fig.add_trace(go.Scatter(
        x=df_filtered['date'],
        y=df_filtered['interest'],
        mode='markers',
        name='Daily Interest',
        marker=dict(size=4, color='#3B82F6', opacity=0.5),
        hoverinfo='y+x'
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
    # Extract main topic and subtopics
    main_topic = keyword_data.get('main_topic', 'Unknown Topic')
    main_keywords = keyword_data.get('main_topic_keywords', [])
    subtopics = keyword_data.get('subtopics', [])
    subtopic_keywords = keyword_data.get('subtopic_keywords', {})
    
    # Prepare data for visualization
    topics = [main_topic] + subtopics
    popularity_values = []
    
    # Calculate average popularity for main topic
    if main_keywords:
        main_popularity = sum(kw['frequency'] for kw in main_keywords) / len(main_keywords)
        popularity_values.append(main_popularity)
    else:
        popularity_values.append(0)
    
    # Calculate popularity for each subtopic
    for subtopic in subtopics:
        subtopic_kws = subtopic_keywords.get(subtopic, [])
        if subtopic_kws:
            subtopic_popularity = sum(kw['frequency'] for kw in subtopic_kws) / len(subtopic_kws)
            popularity_values.append(subtopic_popularity)
        else:
            popularity_values.append(0)
    
    # Create a DataFrame for visualization
    df = pd.DataFrame({
        'Topic': topics,
        'Popularity': popularity_values
    })
    
    # Create color mapping based on whether it's the main topic
    colors = ['#3B82F6' if topic == main_topic else '#60A5FA' for topic in topics]
    
    # Create the figure
    fig = go.Figure(data=[
        go.Bar(
            x=df['Topic'],
            y=df['Popularity'],
            marker_color=colors,
            text=df['Popularity'].round(1),
            textposition='auto'
        )
    ])
    
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
    # Extract main keywords
    main_topic = keyword_data.get('main_topic', 'Unknown')
    main_keywords = keyword_data.get('main_topic_keywords', [])
    
    # Sort keywords by frequency and take top 10
    sorted_keywords = sorted(main_keywords, key=lambda x: x['frequency'], reverse=True)[:10]
    
    # Prepare data for chart
    keywords = [kw['keyword'] for kw in sorted_keywords]
    frequencies = [kw['frequency'] for kw in sorted_keywords]
    sentiments = [kw['sentiment'] for kw in sorted_keywords]
    
    # Map sentiments to colors
    color_map = {
        'positive': '#10B981',  # Green
        'neutral': '#6B7280',   # Gray
        'negative': '#EF4444'   # Red
    }
    colors = [color_map.get(sentiment, '#6B7280') for sentiment in sentiments]
    
    # Create the figure
    fig = go.Figure()
    
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