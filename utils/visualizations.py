import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import random
import numpy as np
import pycountry

def create_global_sentiment_map(data):
    """
    Create a global map visualization showing sentiment by region.
    
    Args:
        data (dict): The filtered analysis data
        
    Returns:
        plotly.graph_objects.Figure: A Plotly map figure
    """
    # Prepare data for visualization
    map_data = []
    
    for analysis_id, analysis in data.items():
        sentiment = analysis.get('sentiment', {}).get('sentiment', 'neutral')
        sentiment_score = analysis.get('sentiment', {}).get('score', 0)
        
        # Get regions for this analysis
        regions = analysis.get('metadata', {}).get('regions', [])
        
        # Map each region to its sentiment
        for region in regions:
            # Try to convert region name to ISO country code for the map
            try:
                country = pycountry.countries.search_fuzzy(region)
                if country:
                    country_code = country[0].alpha_3
                    map_data.append({
                        'country': region,
                        'iso_code': country_code,
                        'sentiment': sentiment,
                        'sentiment_score': sentiment_score
                    })
            except:
                # If we can't find a country code, just use the region name
                map_data.append({
                    'country': region,
                    'iso_code': '',
                    'sentiment': sentiment,
                    'sentiment_score': sentiment_score
                })
    
    if not map_data:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No geographical data available",
            template="plotly_white"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(map_data)
    
    # Calculate average sentiment score per country
    country_sentiment = df.groupby('country')['sentiment_score'].mean().reset_index()
    
    # Create map
    fig = px.choropleth(
        country_sentiment,
        locations=df['iso_code'] if 'iso_code' in df.columns else None,
        locationmode='ISO-3',
        color='sentiment_score',
        hover_name='country',
        color_continuous_scale=px.colors.diverging.RdBu,
        color_continuous_midpoint=0,
        title='Global Sentiment Distribution',
        labels={'sentiment_score': 'Sentiment Score (-1 to 1)'}
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='equirectangular'
        ),
        height=600,
        template="plotly_white"
    )
    
    return fig

def create_sentiment_time_chart(data):
    """
    Create a time series chart showing sentiment trends over time.
    
    Args:
        data (dict): The filtered analysis data
        
    Returns:
        plotly.graph_objects.Figure: A Plotly time series figure
    """
    # Prepare data for visualization
    time_data = []
    
    for analysis_id, analysis in data.items():
        timestamp = analysis.get('timestamp', '')
        sentiment = analysis.get('sentiment', {}).get('sentiment', 'neutral')
        sentiment_score = analysis.get('sentiment', {}).get('score', 0)
        
        if timestamp:
            time_data.append({
                'date': datetime.fromisoformat(timestamp),
                'sentiment': sentiment,
                'sentiment_score': sentiment_score
            })
    
    if not time_data:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No time series data available",
            template="plotly_white"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(time_data)
    
    # Sort by date
    df = df.sort_values('date')
    
    # Create time series chart
    fig = go.Figure()
    
    # Add scatter plot for sentiment scores
    fig.add_trace(go.Scatter(
        x=df['date'],
        y=df['sentiment_score'],
        mode='lines+markers',
        name='Sentiment Score',
        line=dict(color='royalblue'),
        hovertemplate='%{x}<br>Score: %{y:.2f}'
    ))
    
    # Add moving average
    window_size = min(7, len(df))
    if window_size > 1:
        df['ma'] = df['sentiment_score'].rolling(window=window_size).mean()
        fig.add_trace(go.Scatter(
            x=df['date'],
            y=df['ma'],
            mode='lines',
            name=f'{window_size}-Day Moving Average',
            line=dict(color='firebrick', dash='dash'),
            hovertemplate='%{x}<br>MA: %{y:.2f}'
        ))
    
    # Add reference line for neutral sentiment
    fig.add_shape(
        type="line",
        x0=min(df['date']),
        y0=0,
        x1=max(df['date']),
        y1=0,
        line=dict(color="gray", width=1, dash="dot"),
    )
    
    fig.update_layout(
        title='Sentiment Trends Over Time',
        xaxis_title='Date',
        yaxis_title='Sentiment Score (-1 to 1)',
        hovermode='closest',
        height=500,
        template="plotly_white"
    )
    
    return fig

def create_topic_distribution_chart(data):
    """
    Create a chart showing the distribution of topics.
    
    Args:
        data (dict): The filtered analysis data
        
    Returns:
        plotly.graph_objects.Figure: A Plotly chart figure
    """
    # Extract all topics and their sentiment scores
    topic_data = []
    
    for analysis_id, analysis in data.items():
        sentiment_score = analysis.get('sentiment', {}).get('score', 0)
        topics = analysis.get('metadata', {}).get('topics', [])
        
        for topic in topics:
            topic_data.append({
                'topic': topic,
                'sentiment_score': sentiment_score
            })
    
    if not topic_data:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No topic data available",
            template="plotly_white"
        )
    
    # Convert to DataFrame
    df = pd.DataFrame(topic_data)
    
    # Calculate average sentiment score and count per topic
    topic_stats = df.groupby('topic').agg(
        avg_sentiment=('sentiment_score', 'mean'),
        count=('sentiment_score', 'count')
    ).reset_index()
    
    # Sort by count (descending)
    topic_stats = topic_stats.sort_values('count', ascending=False)
    
    # Take top 10 topics
    top_topics = topic_stats.head(10)
    
    # Create horizontal bar chart
    fig = go.Figure()
    
    # Add bars for topic counts
    fig.add_trace(go.Bar(
        y=top_topics['topic'],
        x=top_topics['count'],
        orientation='h',
        name='Frequency',
        marker_color='lightblue',
        hovertemplate='%{y}<br>Count: %{x}<extra></extra>'
    ))
    
    # Create a parallel y-axis for sentiment scores
    fig.update_layout(
        yaxis2=dict(
            overlaying='y',
            side='right',
            showgrid=False,
            zeroline=False
        )
    )
    
    # Add sentiment score markers
    fig.add_trace(go.Scatter(
        y=top_topics['topic'],
        x=top_topics['avg_sentiment'],
        mode='markers',
        name='Avg. Sentiment',
        marker=dict(
            color=top_topics['avg_sentiment'],
            size=12,
            colorscale='RdBu',
            colorbar=dict(title='Sentiment'),
            cmin=-1,
            cmid=0,
            cmax=1
        ),
        yaxis='y2',
        hovertemplate='%{y}<br>Sentiment: %{x:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title='Top Topics by Frequency and Average Sentiment',
        xaxis_title='Count',
        yaxis_title='Topic',
        height=500,
        template="plotly_white",
        hovermode='closest',
        barmode='overlay'
    )
    
    return fig

def create_commodity_price_chart(data):
    """
    Create a chart showing commodity prices and sentiment.
    
    Args:
        data (dict): The filtered analysis data
        
    Returns:
        plotly.graph_objects.Figure: A Plotly chart figure
    """
    # Mock commodity price data - in a real application, this would come from a commodity price API
    # Here we're creating synthetic data that roughly aligns with the date range of our analysis data
    
    # Get the date range from the data
    dates = []
    for analysis_id, analysis in data.items():
        timestamp = analysis.get('timestamp', '')
        if timestamp:
            dates.append(datetime.fromisoformat(timestamp))
    
    if not dates:
        # Return empty figure if no data
        return go.Figure().update_layout(
            title="No commodity data available",
            template="plotly_white"
        )
    
    start_date = min(dates)
    end_date = max(dates)
    
    # Generate date range
    date_range = pd.date_range(start=start_date, end=end_date, freq='D')
    
    # Extract all commodities mentioned in the data
    all_commodities = set()
    for analysis_id, analysis in data.items():
        commodities = analysis.get('metadata', {}).get('commodities', [])
        all_commodities.update(commodities)
    
    # If no commodities found, use a default
    if not all_commodities:
        all_commodities = ['Oil']
    
    # For each commodity, generate a synthetic price series
    commodity_prices = {}
    for commodity in all_commodities:
        # Set up a base price and volatility for known commodities
        if commodity.lower() in ['oil', 'crude oil', 'petroleum']:
            base_price = 70.0
            volatility = 3.0
        elif commodity.lower() in ['gold']:
            base_price = 1800.0
            volatility = 50.0
        elif commodity.lower() in ['gas', 'natural gas']:
            base_price = 3.5
            volatility = 0.2
        else:
            # For unknown commodities, use a generic price range
            base_price = 100.0
            volatility = 5.0
        
        # Generate random walk for price
        np.random.seed(hash(commodity) % 2**32)  # Use commodity name as seed for reproducibility
        steps = np.random.normal(0, volatility, size=len(date_range))
        prices = [base_price]
        for step in steps:
            next_price = max(prices[-1] + step, 0.1)  # Ensure price doesn't go negative
            prices.append(next_price)
        prices = prices[:-1]  # Remove the extra price
        
        commodity_prices[commodity] = prices
    
    # Create the figure
    fig = go.Figure()
    
    # Choose the most frequently mentioned commodity for display
    commodity_counts = {}
    for analysis_id, analysis in data.items():
        commodities = analysis.get('metadata', {}).get('commodities', [])
        for commodity in commodities:
            commodity_counts[commodity] = commodity_counts.get(commodity, 0) + 1
    
    if commodity_counts:
        primary_commodity = max(commodity_counts, key=commodity_counts.get)
    else:
        primary_commodity = list(all_commodities)[0] if all_commodities else 'Oil'
    
    # Add the primary commodity price line
    fig.add_trace(go.Scatter(
        x=date_range,
        y=commodity_prices[primary_commodity],
        mode='lines',
        name=f'{primary_commodity} Price',
        line=dict(color='darkorange', width=2),
        hovertemplate='%{x}<br>Price: $%{y:.2f}'
    ))
    
    # Add sentiment data points for this commodity
    sentiment_dates = []
    sentiment_scores = []
    
    for analysis_id, analysis in data.items():
        commodities = analysis.get('metadata', {}).get('commodities', [])
        
        if primary_commodity in commodities:
            timestamp = analysis.get('timestamp', '')
            sentiment_score = analysis.get('sentiment', {}).get('score', 0)
            
            if timestamp:
                sentiment_dates.append(datetime.fromisoformat(timestamp))
                sentiment_scores.append(sentiment_score)
    
    # Create a secondary y-axis for sentiment scores
    fig.update_layout(
        yaxis2=dict(
            title='Sentiment Score',
            titlefont=dict(color='royalblue'),
            tickfont=dict(color='royalblue'),
            overlaying='y',
            side='right',
            range=[-1, 1]
        )
    )
    
    # Add sentiment markers for the selected commodity
    if sentiment_dates and sentiment_scores:
        fig.add_trace(go.Scatter(
            x=sentiment_dates,
            y=sentiment_scores,
            mode='markers',
            name=f'{primary_commodity} Sentiment',
            marker=dict(
                color=sentiment_scores,
                colorscale='RdBu',
                cmin=-1,
                cmid=0,
                cmax=1,
                size=10
            ),
            yaxis='y2',
            hovertemplate='%{x}<br>Sentiment: %{y:.2f}'
        ))
    
    fig.update_layout(
        title=f'{primary_commodity} Price and Sentiment Correlation',
        xaxis_title='Date',
        yaxis_title='Price (USD)',
        height=500,
        template="plotly_white",
        hovermode='closest'
    )
    
    return fig
