"""
Professional-grade financial sentiment analysis visualization module.
Implements executive dashboard patterns for stock sentiment analysis.
"""
import re
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from datetime import datetime, timedelta
import math
import numpy as np
from scipy import stats
from plotly.subplots import make_subplots
import colorsys

def wilson_confidence_interval(positive, total, confidence=0.95):
    """
    Calculate Wilson confidence interval for proportion.
    More accurate than normal approximation for small samples.
    
    Args:
        positive (int): Number of positive cases
        total (int): Total number of cases
        confidence (float): Confidence level (default 0.95)
        
    Returns:
        tuple: (lower_bound, upper_bound)
    """
    if total == 0:
        return (0, 0)
    
    p = positive / total
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    
    denominator = 1 + z**2 / total
    centre_adjusted_probability = (p + z**2 / (2 * total)) / denominator
    adjusted_standard_deviation = math.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
    
    lower_bound = centre_adjusted_probability - z * adjusted_standard_deviation
    upper_bound = centre_adjusted_probability + z * adjusted_standard_deviation
    
    return (max(0, lower_bound), min(1, upper_bound))

def calculate_sentiment_uncertainty(sentiment_scores, method='bootstrap'):
    """
    Calculate uncertainty metrics for sentiment scores.
    
    Args:
        sentiment_scores (list): List of individual sentiment scores
        method (str): Method to calculate uncertainty ('bootstrap', 'std', 'mad')
        
    Returns:
        dict: Uncertainty metrics
    """
    if not sentiment_scores or len(sentiment_scores) == 0:
        return {'mean': 0, 'std': 0, 'lower_ci': 0, 'upper_ci': 0, 'uncertainty_score': 1.0}
    
    scores = np.array(sentiment_scores)
    mean_score = np.mean(scores)
    
    if method == 'bootstrap' and len(scores) > 1:
        # Bootstrap confidence intervals
        n_bootstrap = min(1000, len(scores) * 10)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            bootstrap_sample = np.random.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        lower_ci = np.percentile(bootstrap_means, 2.5)
        upper_ci = np.percentile(bootstrap_means, 97.5)
        uncertainty_score = (upper_ci - lower_ci) / 2
        
    elif method == 'mad':
        # Median Absolute Deviation - robust to outliers
        median_score = np.median(scores)
        mad = np.median(np.abs(scores - median_score))
        lower_ci = mean_score - 1.96 * mad
        upper_ci = mean_score + 1.96 * mad
        uncertainty_score = mad
        
    else:
        # Standard error method
        std_score = np.std(scores, ddof=1) if len(scores) > 1 else 0
        se = std_score / math.sqrt(len(scores))
        lower_ci = mean_score - 1.96 * se
        upper_ci = mean_score + 1.96 * se
        uncertainty_score = se
    
    return {
        'mean': float(mean_score),
        'std': float(np.std(scores)),
        'lower_ci': float(lower_ci),
        'upper_ci': float(upper_ci),
        'uncertainty_score': float(uncertainty_score),
        'sample_size': len(scores)
    }

def create_sentiment_index_with_uncertainty(analysis_results):
    """
    Create professional sentiment index gauge with uncertainty ribbon.
    Executive View #1: Primary sentiment indicator with confidence bands.
    
    Args:
        analysis_results (dict): Complete analysis results
        
    Returns:
        plotly.graph_objects.Figure: Sentiment index figure
    """
    fig = go.Figure()
    
    # Extract sentiment data
    mathematical_results = analysis_results.get('mathematical_sentiment_analysis', {})
    comprehensive_results = analysis_results.get('comprehensive_results', {})
    
    if not mathematical_results and not comprehensive_results:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No sentiment data available",
            showarrow=False,
            font=dict(size=16, color="#666")
        )
        return fig
    
    # Get main sentiment score and individual scores for uncertainty
    if mathematical_results:
        main_score = mathematical_results.get('composite_score', {}).get('value', 0)
        individual_scores = mathematical_results.get('individual_scores', {})
        sentiment_scores = [
            individual_scores.get('vader_sentiment', 0),
            individual_scores.get('textblob_sentiment', 0),
            individual_scores.get('afinn_sentiment', 0),
            individual_scores.get('roberta_sentiment', 0),
            individual_scores.get('financial_sentiment', 0)
        ]
    else:
        main_score = comprehensive_results.get('sentiment', {}).get('score', 0)
        sentiment_scores = [main_score]  # Limited data
    
    # Calculate uncertainty metrics
    uncertainty = calculate_sentiment_uncertainty(sentiment_scores)
    
    # Convert score from [-1, 1] to [0, 100] scale for display
    display_score = (main_score + 1) * 50
    lower_bound = max(0, (uncertainty['lower_ci'] + 1) * 50)
    upper_bound = min(100, (uncertainty['upper_ci'] + 1) * 50)
    
    # Determine sentiment category and color
    if main_score > 0.1:
        sentiment_label = "POSITIVE"
        primary_color = "#10B981"  # Green
        secondary_color = "rgba(16, 185, 129, 0.3)"
    elif main_score < -0.1:
        sentiment_label = "NEGATIVE"
        primary_color = "#EF4444"  # Red
        secondary_color = "rgba(239, 68, 68, 0.3)"
    else:
        sentiment_label = "NEUTRAL"
        primary_color = "#6B7280"  # Gray
        secondary_color = "rgba(107, 114, 128, 0.3)"
    
    # Create gauge chart
    fig.add_trace(go.Indicator(
        mode="gauge+number+delta",
        value=display_score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': f"SENTIMENT INDEX<br><span style='font-size:0.8em;color:gray'>n={uncertainty['sample_size']} sources</span>"},
        delta={'reference': 50, 'relative': True, 'position': "top"},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': primary_color, 'thickness': 0.8},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': 'rgba(239, 68, 68, 0.2)'},  # Negative zone
                {'range': [30, 70], 'color': 'rgba(107, 114, 128, 0.2)'},  # Neutral zone
                {'range': [70, 100], 'color': 'rgba(16, 185, 129, 0.2)'}   # Positive zone
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': display_score
            }
        }
    ))
    
    # Add uncertainty ribbon (confidence interval) only when valid sample size > 1
    if uncertainty['sample_size'] and uncertainty['sample_size'] > 1:
        fig.add_shape(
            type="rect",
            x0=0.85, y0=lower_bound/100, x1=0.9, y1=upper_bound/100,
            xref="paper", yref="paper",
            fillcolor=secondary_color,
            line=dict(color=primary_color, width=1),
            name="Confidence Interval"
        )
        
        # Add uncertainty score text
        fig.add_annotation(
            x=0.875, y=0.1,
            xref="paper", yref="paper",
            text=f"±{uncertainty['uncertainty_score']:.2f}<br>95% CI",
            showarrow=False,
            font=dict(size=10, color=primary_color),
            align="center"
        )
    
    # Add sentiment label
    fig.add_annotation(
        x=0.5, y=0.15,
        xref="paper", yref="paper",
        text=f"<b>{sentiment_label}</b>",
        showarrow=False,
        font=dict(size=20, color=primary_color)
    )
    
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color="darkblue", size=12)
    )
    
    return fig

def create_polarity_share_bars_with_intervals(analysis_results):
    """
    Create polarity distribution bars with Wilson confidence intervals.
    Executive View #2: Statistical confidence in sentiment distribution.
    
    Args:
        analysis_results (dict): Complete analysis results
        
    Returns:
        plotly.graph_objects.Figure: Polarity bars figure
    """
    fig = go.Figure()
    
    # Extract polarity data
    mathematical_results = analysis_results.get('mathematical_sentiment_analysis', {})
    polarity_dist = mathematical_results.get('polarity_distribution', {})
    
    if not polarity_dist:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No polarity distribution data available",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    # Get distribution data
    positive_pct = polarity_dist.get('positive', 0)
    neutral_pct = polarity_dist.get('neutral', 0)  
    negative_pct = polarity_dist.get('negative', 0)
    
    # Estimate sample size from analysis (prefer real sources length)
    sources = analysis_results.get('sources', [])
    total_units = len(sources) if isinstance(sources, list) and sources else analysis_results.get('unit_count', 0)
    
    # Calculate Wilson intervals for each category
    positive_count = int(positive_pct * total_units / 100)
    neutral_count = int(neutral_pct * total_units / 100)
    negative_count = int(negative_pct * total_units / 100)
    
    pos_lower, pos_upper = wilson_confidence_interval(positive_count, total_units)
    neu_lower, neu_upper = wilson_confidence_interval(neutral_count, total_units)
    neg_lower, neg_upper = wilson_confidence_interval(negative_count, total_units)
    
    # Convert back to percentages
    categories = ['Negative', 'Neutral', 'Positive']
    percentages = [negative_pct, neutral_pct, positive_pct]
    lower_bounds = [neg_lower * 100, neu_lower * 100, pos_lower * 100]
    upper_bounds = [neg_upper * 100, neu_upper * 100, pos_upper * 100]
    colors = ['#EF4444', '#6B7280', '#10B981']
    
    # Create bars with error bars
    fig.add_trace(go.Bar(
        x=categories,
        y=percentages,
        error_y=dict(
            type='data',
            symmetric=False,
            array=[ub - p for ub, p in zip(upper_bounds, percentages)],
            arrayminus=[p - lb for lb, p in zip(lower_bounds, percentages)],
            visible=True,
            color='black',
            thickness=2,
            width=3,
        ),
        marker_color=colors,
        text=[f'{p:.1f}%<br>±{(ub-lb)/2:.1f}%' for p, lb, ub in zip(percentages, lower_bounds, upper_bounds)],
        textposition='outside',
        name='Sentiment Distribution'
    ))
    
    # Add sample size and confidence level info (top-left under title)
    if total_units:
        fig.add_annotation(
            x=0.01, y=1,
            xref="paper", yref="paper",
            xanchor='left', yanchor='top',
            text=f"n={total_units} units",
            showarrow=False,
            font=dict(size=11, color="#666"),
            align="left",
            bgcolor='rgba(0,0,0,0)'
        )
    
    fig.update_layout(
        title=dict(
            text="Polarity Distribution (95% CI)",
            font=dict(size=14)
        ),
        xaxis=dict(title="Sentiment Category", title_standoff=10),
        yaxis=dict(title="Percentage (%)", range=[0, max(upper_bounds) * 1.1], title_standoff=10),
        height=360,
        margin=dict(l=60, r=60, t=80, b=70),
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False
    )

    # Reduce bar text size to avoid overlap
    fig.update_traces(textfont=dict(size=11))
    
    return fig

def create_vad_compass(analysis_results):
    """
    Create VAD (Valence-Arousal-Dominance) compass visualization.
    Executive View #3: Emotional dimensions analysis.
    
    Args:
        analysis_results (dict): Complete analysis results
        
    Returns:
        plotly.graph_objects.Figure: VAD compass figure
    """
    fig = go.Figure()
    
    # Extract VAD data
    mathematical_results = analysis_results.get('mathematical_sentiment_analysis', {})
    vad_scores = mathematical_results.get('vad_analysis', {})
    
    if not vad_scores:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No VAD analysis data available",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    # Get VAD values (assuming they're in 0-100 range, convert to -1 to 1)
    valence = (vad_scores.get('valence', 50) - 50) / 50  # Convert to [-1, 1]
    arousal = (vad_scores.get('arousal', 50) - 50) / 50   # Convert to [-1, 1] 
    dominance = (vad_scores.get('dominance', 50) - 50) / 50 # Convert to [-1, 1]
    
    # Create compass background
    theta = np.linspace(0, 2*np.pi, 100)
    
    # Add concentric circles for reference
    for radius in [0.25, 0.5, 0.75, 1.0]:
        fig.add_trace(go.Scatterpolar(
            r=[radius] * len(theta),
            theta=theta * 180/np.pi,
            mode='lines',
            line=dict(color='lightgray', width=1),
            showlegend=False,
            hoverinfo='skip'
        ))
    
    # Quadrant labels removed per request
    
    # Plot VAD point (using Valence-Arousal as primary axes)
    va_angle = np.arctan2(arousal, valence) * 180/np.pi
    va_magnitude = min(np.sqrt(valence**2 + arousal**2), 1.0)
    
    fig.add_trace(go.Scatterpolar(
        r=[va_magnitude],
        theta=[va_angle],
        mode='markers',
        marker=dict(
            size=20,
            color=dominance,  # Use dominance for color
            colorscale='RdYlGn',
            cmin=-1,
            cmax=1,
            colorbar=dict(title="Dominance", x=1.24, len=0.85, thickness=12, tickfont=dict(size=9)),
            line=dict(color='black', width=2)
        ),
        name='VAD Position',
        hovertemplate='<b>VAD Analysis</b><br>Arousal (r): %{r:.2f}<br>Angle (θ): %{theta:.1f}°<extra></extra>'
    ))

    # Add V/A/D values as a single annotation above the plot to avoid overlap
    fig.add_annotation(
        x=0.5,
        y=1.08,
        xref='paper',
        yref='paper',
        text=f"V: {valence:.2f}  |  A: {arousal:.2f}  |  D: {dominance:.2f}",
        showarrow=False,
        font=dict(size=12, color='#e2e8f0'),
        align='center'
    )
    
    # Add center point reference
    fig.add_trace(go.Scatterpolar(
        r=[0],
        theta=[0],
        mode='markers',
        marker=dict(size=8, color='black'),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    fig.update_layout(
        title=dict(
            text="VAD Emotional Compass",
            font=dict(size=14)
        ),
        polar=dict(
            angularaxis=dict(
                tickvals=[0, 90, 180, 270],
                ticktext=['High<br>Valence', 'High<br>Arousal', 'Low<br>Valence', 'Low<br>Arousal'],
                tickfont=dict(size=8),
                direction='counterclockwise',
                rotation=90
            ),
            domain=dict(x=[0.0, 1.0], y=[0.0, 1.0])
        ),
        height=460,
        margin=dict(l=70, r=120, t=90, b=70),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_source_quality_matrix(analysis_results):
    """
    Create source quality and reliability heatmap.
    Source Transparency #1: Quality assessment of information sources.
    
    Args:
        analysis_results (dict): Complete analysis results
        
    Returns:
        plotly.graph_objects.Figure: Source quality heatmap
    """
    fig = go.Figure()
    
    # Extract source data
    sources = analysis_results.get('sources', [])
    if not sources:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No source quality data available",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    # Analyze sources by type and quality
    source_matrix = {}
    source_types = ['Financial News', 'Social Media', 'Forums', 'Research', 'Other']
    quality_metrics = ['Reliability', 'Sentiment Impact', 'Recency', 'Authority']
    
    # Initialize matrix
    for source_type in source_types:
        source_matrix[source_type] = {metric: [] for metric in quality_metrics}
    
    # Categorize and score sources
    for source in sources:
        url = source.get('url', '')
        title = source.get('title', '')
        sentiment_score = abs(source.get('sentiment', 0))  # Impact magnitude
        
        # Determine source type
        source_type = 'Other'
        if any(domain in url.lower() for domain in ['bloomberg', 'reuters', 'marketwatch', 'cnbc', 'wsj']):
            source_type = 'Financial News'
        elif any(domain in url.lower() for domain in ['reddit', 'twitter', 'facebook', 'linkedin']):
            source_type = 'Social Media'
        elif any(domain in url.lower() for domain in ['seekingalpha', 'fool', 'benzinga']):
            source_type = 'Research'
        elif any(word in url.lower() for word in ['forum', 'discussion', 'board']):
            source_type = 'Forums'
        
        # Calculate quality scores (0-1 scale)
        reliability = 0.9 if source_type == 'Financial News' else 0.7 if source_type == 'Research' else 0.5
        sentiment_impact = min(sentiment_score * 2, 1.0)  # Normalize to 0-1
        recency = 1.0  # Assume recent for simplicity
        authority = 0.8 if source_type == 'Financial News' else 0.6 if source_type == 'Research' else 0.4
        
        source_matrix[source_type]['Reliability'].append(reliability)
        source_matrix[source_type]['Sentiment Impact'].append(sentiment_impact)
        source_matrix[source_type]['Recency'].append(recency)
        source_matrix[source_type]['Authority'].append(authority)
    
    # Calculate average scores for heatmap
    heatmap_data = []
    y_labels = []
    x_labels = quality_metrics
    
    for source_type in source_types:
        row = []
        has_data = False
        for metric in quality_metrics:
            scores = source_matrix[source_type][metric]
            if scores:
                avg_score = np.mean(scores)
                has_data = True
            else:
                avg_score = 0
            row.append(avg_score)
        
        if has_data:  # Only include source types with data
            heatmap_data.append(row)
            y_labels.append(f"{source_type} (n={len(source_matrix[source_type]['Reliability'])})")
    
    if not heatmap_data:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No source data to analyze",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    # Create heatmap
    fig.add_trace(go.Heatmap(
        z=heatmap_data,
        x=x_labels,
        y=y_labels,
        colorscale='RdYlGn',
        zmin=0,
        zmax=1,
        text=[[f'{val:.2f}' for val in row] for row in heatmap_data],
        texttemplate='%{text}',
        textfont=dict(size=12),
        colorbar=dict(
            title="Quality Score"
        ),
        hoverongaps=False,
        hovertemplate='<b>%{y}</b><br>%{x}: %{z:.2f}<extra></extra>'
    ))
    
    fig.update_layout(
        title=dict(
            text="Source Quality Matrix",
            font=dict(size=16)
        ),
        xaxis=dict(title="Quality Metrics"),
        yaxis=dict(title="Source Types"),
        height=300,
        margin=dict(l=120, r=80, t=60, b=40),
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    return fig

def create_rolling_sentiment_timeline(analysis_results):
    """
    Create rolling sentiment timeline with trend analysis.
    Temporal Dynamics #1: Sentiment evolution over time.
    
    Args:
        analysis_results (dict): Complete analysis results
        
    Returns:
        plotly.graph_objects.Figure: Timeline figure
    """
    fig = go.Figure()
    
    # Extract temporal data
    sources = analysis_results.get('sources', [])
    if not sources:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No temporal data available",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    # Create timeline data
    timeline_data = []
    now = datetime.now()
    for source in sources:
        # If a true timestamp exists, prefer it; otherwise, derive a bounded recent timestamp to avoid overlap
        ts = source.get('published_at') or source.get('published_date') or source.get('timestamp')
        if not ts:
            # Skip sources without valid timestamps (no synthetic times)
            continue
        try:
            timestamp = pd.to_datetime(ts)
        except Exception:
            # Skip unparseable timestamps
            continue

        sentiment = float(source.get('sentiment', 0))
        title = source.get('title', 'Unknown')
        if isinstance(title, str) and len(title) > 80:
            title = title[:77] + '…'
        timeline_data.append({'timestamp': timestamp, 'sentiment': sentiment, 'source': title})
    
    # Sort by timestamp
    timeline_data.sort(key=lambda x: x['timestamp'])
    
    if len(timeline_data) < 2:
        fig.add_annotation(
            x=0.5, y=0.5,
            xref="paper", yref="paper",
            text="No sufficient dated sources for timeline",
            showarrow=False,
            font=dict(size=14, color="#666")
        )
        return fig
    
    # Create DataFrame for analysis
    df = pd.DataFrame(timeline_data)
    df = df.sort_values('timestamp')
    
    # Calculate rolling averages
    window_sizes = [3, 6, 12] if len(df) >= 12 else [min(3, len(df)//2)]
    
    for window in window_sizes:
        if window <= len(df):
            df[f'rolling_{window}h'] = df['sentiment'].rolling(window=window, center=True).mean()
    
    # Plot points with decimation to avoid clutter
    colors = ['#10B981' if s > 0.1 else '#EF4444' if s < -0.1 else '#6B7280' for s in df['sentiment']]
    max_points = 300
    show_points = len(df) <= max_points
    if show_points:
        fig.add_trace(go.Scattergl(
            x=df['timestamp'],
            y=df['sentiment'],
            mode='markers',
            marker=dict(
                size=6,
                color=colors,
                opacity=0.5,
                line=dict(width=0)
            ),
            name='Sources',
            hovertemplate='<b>%{text}</b><br>%{x|%b %d, %H:%M}<br>Sentiment: %{y:.3f}<extra></extra>',
            text=df['source']
        ))
    
    # Add rolling averages (always visible)
    for window in window_sizes:
        col_name = f'rolling_{window}h'
        if col_name in df.columns:
            fig.add_trace(go.Scatter(
                x=df['timestamp'],
                y=df[col_name],
                mode='lines',
                name=f'{window}-Source Rolling Avg',
                line=dict(width=3, dash='dash' if window > window_sizes[0] else 'solid'),
                opacity=0.95
            ))
    
    # Add trend line
    if len(df) > 2:
        # Simple linear trend
        x_numeric = [(t - df['timestamp'].min()).total_seconds()/3600 for t in df['timestamp']]
        trend_coef = np.polyfit(x_numeric, df['sentiment'], 1)
        trend_line = np.poly1d(trend_coef)
        trend_direction = "Improving" if trend_coef[0] > 0 else "Declining" if trend_coef[0] < 0 else "Stable"
        
        fig.add_trace(go.Scatter(
            x=df['timestamp'],
            y=trend_line(x_numeric),
            mode='lines',
            name=f'Trend ({trend_direction})',
            line=dict(color='white', width=2, dash='dot'),
            opacity=0.8
        ))

    # Add horizontal reference lines
    fig.add_hline(y=0, line_dash="dot", line_color="gray", opacity=0.5)
    fig.add_hline(y=0.1, line_dash="dot", line_color="green", opacity=0.3)
    fig.add_hline(y=-0.1, line_dash="dot", line_color="red", opacity=0.3)
    
    # Compute dynamic y-axis range to improve vertical spacing while respecting bounds
    try:
        y_min_raw = float(np.nanmin(df['sentiment'].values))
        y_max_raw = float(np.nanmax(df['sentiment'].values))
        y_min = max(-1.1, y_min_raw)
        y_max = min(1.1, y_max_raw)
        y_span = max(0.0, y_max - y_min)
        pad = max(0.05, y_span * 0.2)  # 20% padding, min 0.05
        y_min = max(-1.1, y_min - pad)
        y_max = min(1.1, y_max + pad)
        # Ensure minimum span for readability
        if (y_max - y_min) < 0.3:
            extra = (0.3 - (y_max - y_min)) / 2.0
            y_min = max(-1.1, y_min - extra)
            y_max = min(1.1, y_max + extra)
        # Compute a reasonable tick step (~4-5 ticks)
        y_dtick = max(0.05, round((y_max - y_min) / 5.0, 2))
    except Exception:
        y_min, y_max, y_dtick = -1.1, 1.1, 0.5
    
    fig.update_layout(
        title=dict(
            text="Time",
            x=0.5,
            y=0.98,
            font=dict(size=14)
        ),
        hovermode='x unified',
        xaxis=dict(
            title="",
            showgrid=True,
            gridcolor='lightgray',
            automargin=True,
            nticks=8,
            tickformat="%b %d",
            title_standoff=30,
            rangeslider=dict(
                visible=True,
                thickness=0.10,
                bgcolor='rgba(0,0,0,0.7)',
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            ),
            rangeselector=dict(
                buttons=list([
                    dict(count=24, label="1D", step="hour", stepmode="backward"),
                    dict(count=7, label="1W", step="day", stepmode="backward"),
                    dict(step="all", label="ALL")
                ]),
                x=0.5,
                xanchor='center',
                y=-0.22,
                yanchor='top',
                bgcolor='rgba(0,0,0,0.85)',
                activecolor='#1f2937',
                font=dict(color='#e5e7eb', size=10),
                bordercolor='rgba(255,255,255,0.1)',
                borderwidth=1
            )
        ),
        yaxis=dict(
            title="Sentiment Score",
            range=[-1, 1],
            tickmode='array',
            tickvals=[-1, -0.75, -0.5, -0.25, 0, 0.25, 0.5, 0.75, 1],
            ticktext=['-1.00', '-0.75', '-0.50', '-0.25', '0', '0.25', '0.50', '0.75', '1.00'],
            ticks='outside',
            ticklen=8,
            showgrid=True,
            gridcolor='lightgray'
        ),
        height=420,
        margin=dict(l=60, r=60, t=160, b=140),
        paper_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.10,
            xanchor="center",
            x=0.5,
            font=dict(size=10),
            bgcolor='rgba(0,0,0,0)',
            bordercolor='rgba(255,255,255,0.1)',
            borderwidth=0
        )
    )
    
    return fig

# Legacy function names for backwards compatibility
def create_3d_globe_visualization(geo_data):
    """Legacy wrapper — returns empty annotated figure if no real data provided."""
    try:
        countries = geo_data.get('countries', []) if isinstance(geo_data, dict) else []
        if countries:
            mapped = []
            for c in countries:
                mapped.append({'title': c.get('country', 'Unknown'), 'sentiment': c.get('sentiment_score', 0.0)})
            return create_source_quality_matrix({'sources': mapped})
    except Exception:
        pass
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No globe data available', showarrow=False, font=dict(size=14, color='#666'))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_interest_over_time_chart(time_data, period='year'):
    """Legacy wrapper — no synthetic defaults; returns annotated empty if none."""
    if isinstance(time_data, list) and time_data:
        return create_rolling_sentiment_timeline({'sources': time_data})
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No time series data available', showarrow=False, font=dict(size=14, color='#666'))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_topic_popularity_chart(keyword_data):
    """Legacy wrapper — expects real polarity_distribution; no synthetic defaults."""
    if isinstance(keyword_data, dict):
        pol = keyword_data.get('polarity_distribution')
        if pol:
            return create_polarity_share_bars_with_intervals({'mathematical_sentiment_analysis': {'polarity_distribution': pol}, 'unit_count': keyword_data.get('unit_count', 0)})
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No polarity data available', showarrow=False, font=dict(size=14, color='#666'))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)')
    return fig

def create_keyword_chart(keyword_data):
    """Legacy wrapper — expects real VAD values; no synthetic defaults."""
    if isinstance(keyword_data, dict):
        vad = keyword_data.get('vad_analysis') or {}
        if vad:
            return create_vad_compass({'mathematical_sentiment_analysis': {'vad_analysis': vad}})
    fig = go.Figure()
    fig.add_annotation(x=0.5, y=0.5, xref='paper', yref='paper', text='No VAD data available', showarrow=False, font=dict(size=14, color='#666'))
    fig.update_layout(height=300, margin=dict(l=40, r=40, t=40, b=40), paper_bgcolor='rgba(0,0,0,0)')
    return fig
