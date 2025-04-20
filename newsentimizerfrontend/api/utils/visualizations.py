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
    Enhanced conversion of country name to ISO 3166-1 alpha-3 code for map plotting
    with better handling of common variations and special cases.
    
    Args:
        country_name (str): Name of the country
        
    Returns:
        str: ISO 3166-1 alpha-3 code or None if not found
    """
    # Handle None or empty input
    if not country_name:
        return None
        
    # Clean the input
    country_name = country_name.strip()
    
    # Special case mapping for common variations and abbreviations
    special_cases = {
        # Common country name variations
        'usa': 'USA',
        'united states': 'USA',
        'united states of america': 'USA',
        'u.s.': 'USA',
        'u.s.a.': 'USA',
        'america': 'USA',
        'uk': 'GBR',
        'united kingdom': 'GBR',
        'great britain': 'GBR',
        'england': 'GBR',
        'russia': 'RUS',
        'uae': 'ARE',
        'united arab emirates': 'ARE',
        'china': 'CHN',
        'south korea': 'KOR',
        'korea': 'KOR',
        'north korea': 'PRK',
        'taiwan': 'TWN',
        'iran': 'IRN',
        'venezuela': 'VEN',
        'syria': 'SYR',
        'vietnam': 'VNM',
        'laos': 'LAO',
        'ivory coast': 'CIV',
        "côte d'ivoire": 'CIV',
        'cote d\'ivoire': 'CIV',
        'democratic republic of congo': 'COD',
        'dr congo': 'COD',
        'congo-kinshasa': 'COD',
        'republic of congo': 'COG',
        'congo-brazzaville': 'COG',
        'tanzania': 'TZA',
        'myanmar': 'MMR',
        'burma': 'MMR',
        'palestine': 'PSE'
    }
    
    # First, check our special cases dictionary (case insensitive)
    if country_name.lower() in special_cases:
        # Return the alpha-3 code directly if it's in our special cases
        direct_code = special_cases[country_name.lower()]
        # If the code is already alpha-3, return it
        if len(direct_code) == 3:
            return direct_code
        # Otherwise convert alpha-2 to alpha-3
        try:
            country = pycountry.countries.get(alpha_2=direct_code)
            if country:
                return country.alpha_3
        except:
            pass
    
    try:
        # Try exact match by name first
        country = pycountry.countries.get(name=country_name)
        if country:
            return country.alpha_3
        
        # Try by official name
        country = pycountry.countries.get(official_name=country_name)
        if country:
            return country.alpha_3
            
        # Try search by name (this will do fuzzy matching)
        countries = pycountry.countries.search_fuzzy(country_name)
        if countries:
            return countries[0].alpha_3
            
    except Exception as e:
        print(f"Error converting country name '{country_name}': {e}")
        
    # As a last resort, try our own simple fuzzy matching
    try:
        all_countries = list(pycountry.countries)
        for country in all_countries:
            # Check against name or official name if available
            country_names = [country.name.lower()]
            if hasattr(country, 'official_name'):
                country_names.append(country.official_name.lower())
                
            # Check for substantial substring match
            for name in country_names:
                if name in country_name.lower() or country_name.lower() in name:
                    if len(country_name) > 3:  # Avoid matching short strings
                        return country.alpha_3
    except:
        pass
            
    # If all else fails, return None
    print(f"Could not find country code for: {country_name}")
    return None

def create_3d_globe_visualization(geo_data):
    """
    Create a 3D globe visualization showing sentiment across regions
    with enhanced mapping and visual representation.
    
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
            text=["No geographic data available.<br>Sentiment analysis by region will appear here."],
            mode="text",
            textfont=dict(size=14, color="#3B82F6"),
            showlegend=False
        ))
        
        # Update layout for an empty globe
        fig.update_layout(
            title=dict(
                text='Regional Interest in Topic',
                font=dict(size=18, color="#3B82F6")
            ),
            height=600,
            margin=dict(l=0, r=0, b=0, t=40),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            geo=dict(
                projection_type="orthographic",
                showland=True,
                showcountries=True,
                landcolor="lightgray",
                countrycolor='lightgray',
                showocean=True,
                oceancolor='lightblue',
                showlakes=False,
                showrivers=False,
                projection_rotation=dict(lon=0, lat=0, roll=0)
            )
        )
        
        return fig
    
    # If we have data, create sentiment markers for each country
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
    
    # Add sentiment markers
    for country in countries_data:
        fig.add_trace(go.Scattergeo(
            lon=[country['longitude']],
            lat=[country['latitude']],
            text=f"{country['name']}: {country['interest']:.1f}% interest<br>Sentiment: {country['sentiment'].capitalize()}",
            mode='markers',
            marker=dict(
                size=max(5, country['interest'] / 2),  # Adjusted size formula
                color=country['color'],
                opacity=0.8,
                line=dict(width=1, color='black')
            ),
            name=country['name'],
            showlegend=False
        ))
    
    # Add country choropleth layer for better visibility
    # Create a list of country codes and colors
    country_codes = []
    country_colors = []
    
    for country in countries:
        name = country.get('name', '')
        sentiment = country.get('sentiment', 'neutral')
        
        # Get the ISO alpha-3 code
        code = None
        if 'alpha_3' in country:
            code = country['alpha_3']
        else:
            code = country_name_to_code(name)
            
        if code:
            country_codes.append(code)
            
            # Assign color based on sentiment
            if sentiment == 'positive':
                country_colors.append('#10B981')  # Green
            elif sentiment == 'negative':
                country_colors.append('#EF4444')  # Red
            else:
                country_colors.append('#6B7280')  # Gray
    
    # Add choropleth for countries if we have codes
    if country_codes:
        fig.add_trace(go.Choropleth(
            locations=country_codes,
            z=[1] * len(country_codes),  # Dummy values, we're using the colors directly
            colorscale=[[0, 'rgba(0,0,0,0)'], [1, 'rgba(0,0,0,0)']],  # Transparent
            marker_line_color='darkgray',
            marker_line_width=0.5,
            showscale=False,
            customdata=country_colors,
            hoverinfo='none'
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
        name='Positive Sentiment',
        showlegend=True
    ))
    
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='markers',
        marker=dict(size=10, color='#6B7280'),
        name='Neutral Sentiment',
        showlegend=True
    ))
    
    fig.add_trace(go.Scattergeo(
        lon=[None],
        lat=[None],
        mode='markers',
        marker=dict(size=10, color='#EF4444'),
        name='Negative Sentiment',
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
    Create a chart showing interest over time for a topic with improved error handling
    and enhanced visualization.
    
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
            text="No time series data available.<br>Interest over time will appear here.",
            showarrow=False,
            font=dict(size=14, color="#3B82F6")
        )
        
        fig.update_layout(
            title=dict(
                text=f'Interest Over Time ({period.title()})',
                font=dict(size=18, color="#3B82F6")
            ),
            xaxis=dict(
                title=dict(
                    text='Date',
                    font=dict(size=14)
                ),
                showgrid=True,
                gridcolor='lightgray'
            ),
            yaxis=dict(
                title=dict(
                    text='Interest Level',
                    font=dict(size=14)
                ),
                range=[0, 100],
                showgrid=True,
                gridcolor='lightgray'
            ),
            height=400,
            margin=dict(l=40, r=40, t=40, b=40),
            paper_bgcolor='rgba(0, 0, 0, 0)',
            plot_bgcolor='rgba(0, 0, 0, 0)'
        )
        
        return fig
    
    # Convert time_data to pandas DataFrame if it's a list of dictionaries
    if isinstance(time_data, list):
        try:
            # Ensure data has the right format
            for item in time_data:
                if 'date' not in item and 'interest' not in item:
                    raise ValueError("Time data missing required fields")
                    
            df = pd.DataFrame(time_data)
            
            # Ensure date is in datetime format
            if 'date' in df.columns:
                if not pd.api.types.is_datetime64_any_dtype(df['date']):
                    # Try to convert to datetime
                    try:
                        df['date'] = pd.to_datetime(df['date'])
                    except:
                        # Create dummy dates if conversion fails
                        today = pd.Timestamp.now()
                        df['date'] = [today - pd.Timedelta(days=i) for i in range(len(df), 0, -1)]
            else:
                # Create dummy dates if no date column
                today = pd.Timestamp.now()
                df['date'] = [today - pd.Timedelta(days=i) for i in range(len(df), 0, -1)]
                
            # Ensure we have an interest column
            if 'interest' not in df.columns:
                # Use dummy random interest values
                import random
                df['interest'] = [random.randint(10, 90) for _ in range(len(df))]
                
            # Sort by date
            df = df.sort_values('date')
                
        except Exception as e:
            print(f"Error converting time_data to DataFrame: {e}")
            # Create a simple dummy DataFrame
            today = pd.Timestamp.now()
            dates = [today - pd.Timedelta(days=i) for i in range(30, 0, -1)]
            interest = [50 + 15*np.sin(i/5) for i in range(30)]
            df = pd.DataFrame({'date': dates, 'interest': interest})
    else:
        # If it's already a DataFrame, check that it has the necessary columns
        df = time_data
        if 'date' not in df.columns or 'interest' not in df.columns:
            print("DataFrame missing required columns")
            # Create a simple dummy DataFrame
            today = pd.Timestamp.now()
            dates = [today - pd.Timedelta(days=i) for i in range(30, 0, -1)]
            interest = [50 + 15*np.sin(i/5) for i in range(30)]
            df = pd.DataFrame({'date': dates, 'interest': interest})
    
    # Add smoothed interest if not present
    if 'interest_smoothed' not in df.columns:
        try:
            df['interest_smoothed'] = df['interest'].rolling(window=min(7, len(df)//3 or 1), center=True).mean()
            # Fill NaN values at the edges
            df['interest_smoothed'] = df['interest_smoothed'].fillna(df['interest'])
        except Exception as e:
            print(f"Error creating smoothed interest: {e}")
            df['interest_smoothed'] = df['interest']
    
    # Filter data based on period
    try:
        today = pd.Timestamp.now()
        
        if period.lower() == 'week':
            df = df[df['date'] >= (today - pd.Timedelta(days=7))]
        elif period.lower() == 'month':
            df = df[df['date'] >= (today - pd.Timedelta(days=30))]
        elif period.lower() == 'year':
            df = df[df['date'] >= (today - pd.Timedelta(days=365))]
        # 'all' does not filter
            
        # If filtering resulted in empty DataFrame, revert to all data
        if len(df) == 0:
            if isinstance(time_data, list):
                df = pd.DataFrame(time_data)
            else:
                df = time_data
    except Exception as e:
        print(f"Error filtering data by period: {e}")
    
    # Create the main line for raw interest
    try:
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['interest'],
            mode='lines+markers',
            name='Interest Level',
            line=dict(color='#3B82F6', width=1, dash='dot'),
            marker=dict(size=4, color='#3B82F6'),
            opacity=0.6
        ))
        
        # Add the smoothed line
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['interest_smoothed'],
            mode='lines',
            name='Trend',
            line=dict(color='#3B82F6', width=3),
            opacity=1
        ))
        
        # Add range slider
        fig.update_layout(
            xaxis=dict(
                rangeselector=dict(
                    buttons=list([
                        dict(count=7, label="1w", step="day", stepmode="backward"),
                        dict(count=1, label="1m", step="month", stepmode="backward"),
                        dict(count=6, label="6m", step="month", stepmode="backward"),
                        dict(count=1, label="YTD", step="year", stepmode="todate"),
                        dict(count=1, label="1y", step="year", stepmode="backward"),
                        dict(step="all")
                    ])
                ),
                rangeslider=dict(visible=True),
                type="date"
            )
        )
        
        # Try to identify significant events based on peaks and valleys
        # This would typically come from additional analysis
        try:
            # Find peaks (local maxima)
            from scipy.signal import find_peaks
            peaks, _ = find_peaks(df['interest'].values, prominence=10, distance=5)
            
            # Add annotations for peaks
            for peak in peaks[:3]:  # Limit to top 3 peaks
                peak_date = df['date'].iloc[peak]
                peak_interest = df['interest'].iloc[peak]
                
                fig.add_trace(go.Scatter(
                    x=[peak_date],
                    y=[peak_interest],
                    mode='markers',
                    marker=dict(size=10, color='#10B981', line=dict(width=2, color='white')),
                    name=f'Peak Interest: {peak_date.strftime("%b %d")}',
                    showlegend=False
                ))
        except Exception as e:
            print(f"Error identifying peaks: {e}")
    
    except Exception as e:
        print(f"Error creating time series chart: {e}")
        fig.add_annotation(
            x=0.5,
            y=0.5,
            xref="paper",
            yref="paper",
            text=f"Error creating chart: {str(e)}",
            showarrow=False,
            font=dict(size=14, color="#EF4444")
        )
    
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
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)'
        ),
        yaxis=dict(
            title=dict(
                text='Interest Level',
                font=dict(size=14)
            ),
            range=[0, max(df['interest'].max() * 1.1, 100)],
            showgrid=True,
            gridcolor='rgba(211, 211, 211, 0.3)'
        ),
        height=400,
        margin=dict(l=40, r=40, t=40, b=40),
        paper_bgcolor='rgba(0, 0, 0, 0)',
        plot_bgcolor='rgba(250, 250, 250, 0.8)',
        hovermode='closest',
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        )
    )
    
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
            text="Processing topic data...<br>Analyzing related keywords and themes.",
            showarrow=False,
            font=dict(size=14, color="#3B82F6")
        )
        
        fig.update_layout(
            title=dict(
                text='Topic and Subtopic Popularity',
                font=dict(size=18, color="#3B82F6")
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
            font=dict(size=18, color="#3B82F6")
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
            text=f"Processing keyword data for '{main_topic}'...<br>Analyzing topic-related terminology.",
            showarrow=False,
            font=dict(size=14, color="#3B82F6")
        )
        
        fig.update_layout(
            title=dict(
                text=f'Top Keywords for {main_topic}',
                font=dict(size=18, color="#3B82F6")
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
            font=dict(size=18, color="#3B82F6")
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