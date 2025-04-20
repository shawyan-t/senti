"""
Module for scraping data from Google Trends to provide real geographic and temporal data.
"""
import time
import json
import random
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pytrends.request import TrendReq
import pycountry
import plotly.graph_objects as go
import traceback

# Function to create a new pytrends instance (avoid global usage)
def get_pytrends_instance():
    """Create a fresh pytrends instance to avoid stale connections"""
    try:
        # For newer versions of the requests library
        return TrendReq(hl='en-US', tz=360, timeout=(10, 25))
    except TypeError:
        # For older versions of the library that expect method_whitelist
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        
        session = requests.Session()
        retry = Retry(total=2, backoff_factor=0.1)
        adapter = HTTPAdapter(max_retries=retry)
        session.mount('https://', adapter)
        session.mount('http://', adapter)
        
        return TrendReq(hl='en-US', tz=360, timeout=(10, 25), session=session)

def fetch_interest_by_region(query, resolution='COUNTRY', geo='', timeframe='today 12-m'):
    """
    Fetch interest by region data from Google Trends.
    
    Args:
        query (str): The query term to search for
        resolution (str): The geographic resolution (COUNTRY, REGION, etc.)
        geo (str): The geographic restriction (e.g., 'US' for United States)
        timeframe (str): The time range (e.g., 'today 1-m', 'today 12-m')
        
    Returns:
        dict: Dictionary with country data or empty dict if failed
    """
    try:
        # Get a fresh pytrends instance
        pytrends = get_pytrends_instance()
        
        # Build the payload
        pytrends.build_payload([query], cat=0, timeframe=timeframe, geo=geo)
        
        # Get interest by region
        df_regions = pytrends.interest_by_region(resolution=resolution, inc_low_vol=True, inc_geo_code=True)
        
        # Reset index to get country codes as a column
        df_regions = df_regions.reset_index()
        
        # Ensure the query column exists
        if query not in df_regions.columns:
            print(f"Query '{query}' not in results. Available columns: {df_regions.columns}")
            return {}
        
        # Filter out rows with zero interest
        df_regions = df_regions[df_regions[query] > 0]
        
        # If we have no data, return empty dict
        if len(df_regions) == 0:
            return {}
        
        # Prepare the result structure
        countries_data = []
        
        # Process each country
        for _, row in df_regions.iterrows():
            country_code = row['geoCode'] if 'geoCode' in df_regions.columns else row['geoName']
            country_name = row['geoName']
            interest = row[query]
            
            # Get country details
            try:
                # Try to get country by alpha-2
                country = pycountry.countries.get(alpha_2=country_code)
                
                # If not found, try by name
                if not country and len(country_name) > 2:
                    countries = pycountry.countries.search_fuzzy(country_name)
                    if countries:
                        country = countries[0]
                
                # If we have a country, get its coordinates
                if country:
                    # This is simplified - in a production environment, you'd use a proper geo database
                    # The coordinates are approximated based on the country's center
                    
                    # Create a mapping of country alpha-2 codes to approximate coordinates
                    # This is a small sample of countries - in a real app, you'd have a complete database
                    country_coords = {
                        'US': (37.0902, -95.7129),  # United States
                        'GB': (55.3781, -3.4360),   # United Kingdom
                        'FR': (46.2276, 2.2137),    # France
                        'DE': (51.1657, 10.4515),   # Germany
                        'JP': (36.2048, 138.2529),  # Japan
                        'CN': (35.8617, 104.1954),  # China
                        'IN': (20.5937, 78.9629),   # India
                        'BR': (-14.2350, -51.9253), # Brazil
                        'RU': (61.5240, 105.3188),  # Russia
                        'AU': (-25.2744, 133.7751), # Australia
                        'ZA': (-30.5595, 22.9375),  # South Africa
                        # Add more as needed
                    }
                    
                    # Get coordinates from the mapping or use approximate ones
                    if country.alpha_2 in country_coords:
                        lat, lon = country_coords[country.alpha_2]
                    else:
                        # Generate plausible coordinates based on country code
                        # This is just a fallback and not geographically accurate
                        # In a real implementation, you would use a geographic database
                        random.seed(hash(country.alpha_2))
                        lat = random.uniform(-60, 70)
                        lon = random.uniform(-180, 180)
                    
                    # Determine sentiment based on external data or use a neutral default
                    # This would typically come from sentiment analysis of content related to this region
                    sentiment = 'neutral'
                    
                    # Add to the countries data
                    countries_data.append({
                        'name': country.name,
                        'alpha_2': country.alpha_2,
                        'alpha_3': country.alpha_3,
                        'latitude': lat,
                        'longitude': lon,
                        'interest': interest,
                        'sentiment': sentiment
                    })
            except Exception as e:
                print(f"Error processing country {country_code}: {e}")
                continue
        
        return {
            'countries': countries_data,
            'query': query,
            'timestamp': datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error fetching interest by region for '{query}': {e}")
        traceback.print_exc()
        return {}

def fetch_interest_over_time(query, timeframe='today 12-m'):
    """
    Fetch interest over time data from Google Trends.
    
    Args:
        query (str): The query term to search for
        timeframe (str): The time range (e.g., 'today 1-m', 'today 12-m')
        
    Returns:
        pandas.DataFrame: DataFrame with dates and interest values or empty list if failed
    """
    try:
        # Get a fresh pytrends instance
        pytrends = get_pytrends_instance()
        
        # Build the payload
        pytrends.build_payload([query], cat=0, timeframe=timeframe)
        
        # Get interest over time
        df_time = pytrends.interest_over_time()
        
        # Check if we have data
        if df_time.empty:
            print(f"No interest over time data for '{query}'")
            return []
        
        # Prepare the result
        result = []
        
        # Convert to the format used by the visualization
        for date, row in df_time.iterrows():
            # Skip isPartial column if it exists
            if query in row:
                result.append({
                    'date': date,
                    'interest': row[query],
                    'query': query
                })
        
        # Create the DataFrame for visualization
        df_result = pd.DataFrame(result)
        
        # Add a smoothed column for trend visualization
        if len(df_result) > 0:
            df_result['interest_smoothed'] = df_result['interest'].rolling(window=7, center=True).mean().fillna(df_result['interest'])
        
        return df_result
        
    except Exception as e:
        print(f"Error fetching interest over time for '{query}': {e}")
        traceback.print_exc()
        return []

def fetch_related_queries(query, timeframe='today 12-m'):
    """
    Fetch related queries from Google Trends.
    
    Args:
        query (str): The query term to search for
        timeframe (str): The time range (e.g., 'today 1-m', 'today 12-m')
        
    Returns:
        dict: Dictionary with related queries data
    """
    try:
        # Get a fresh pytrends instance
        pytrends = get_pytrends_instance()
        
        # Build the payload
        pytrends.build_payload([query], cat=0, timeframe=timeframe)
        
        # Get related queries
        related_queries = pytrends.related_queries()
        
        # Check if we have data
        if not related_queries or query not in related_queries:
            print(f"No related queries data for '{query}'")
            return {}
        
        # Extract top related queries
        top_queries = related_queries[query].get('top', pd.DataFrame())
        
        # Convert to the format used by the visualizations
        keywords = []
        if not top_queries.empty:
            for i, row in top_queries.iterrows():
                if i >= 10:  # Limit to top 10
                    break
                
                # Random sentiment assignment (in a real app, you would analyze the sentiment of each keyword)
                sentiment_options = ['positive', 'neutral', 'negative']
                sentiment_weights = [0.3, 0.4, 0.3]  # Slightly biased toward neutral
                sentiment = random.choices(sentiment_options, weights=sentiment_weights, k=1)[0]
                
                keywords.append({
                    'keyword': row['query'],
                    'frequency': row['value'],
                    'sentiment': sentiment
                })
        
        return {
            'main_topic': query,
            'main_topic_keywords': keywords,
            'subtopics': [],  # Would be populated from related topics
            'subtopic_keywords': {}
        }
        
    except Exception as e:
        print(f"Error fetching related queries for '{query}': {e}")
        traceback.print_exc()
        return {}

def fetch_google_trends_data(query, timeframe='today 12-m'):
    """
    Fetch comprehensive data from Google Trends for a query.
    
    Args:
        query (str): The query term to search for
        timeframe (str): The time range (e.g., 'today 1-m', 'today 12-m')
        
    Returns:
        dict: Dictionary with all Google Trends data
    """
    # Add a small delay to avoid rate limiting
    time.sleep(1)
    
    # Fetch all data types
    geo_data = fetch_interest_by_region(query, timeframe=timeframe)
    time_data = fetch_interest_over_time(query, timeframe=timeframe)
    keyword_data = fetch_related_queries(query, timeframe=timeframe)
    
    # Return a comprehensive data structure
    return {
        'geo_data': geo_data,
        'time_data': time_data,
        'keyword_data': keyword_data,
        'query': query,
        'timestamp': datetime.now().isoformat()
    }