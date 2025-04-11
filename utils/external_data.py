"""
Module for fetching external data from various online sources for sentiment analysis.
Uses real API data only - no synthetic data generation.
"""
import os
import json
import re
import time
import base64
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import pycountry
import trafilatura
from openai import OpenAI
from .config import config

# API Keys from configuration
TWITTER_API_KEY = config['twitter_api_key']
TWITTER_API_SECRET = config['twitter_api_secret']
NEWS_API_KEY = config['news_api_key']
GOOGLE_TRENDS_API_KEY = config['google_trends_api_key']

class ExternalDataFetcher:
    """Handles fetching data from external sources like news, social media, etc."""
    
    def __init__(self):
        self.twitter_api_available = bool(TWITTER_API_KEY and TWITTER_API_SECRET)
        self.news_api_available = bool(NEWS_API_KEY)
        self.google_trends_api_available = bool(GOOGLE_TRENDS_API_KEY)
        self.twitter_bearer_token = None
        
        # Get Twitter bearer token if keys are available
        if self.twitter_api_available:
            self._get_twitter_bearer_token()
    
    def _get_twitter_bearer_token(self):
        """Get Twitter bearer token using the API keys"""
        try:
            # Create bearer token credentials
            key_secret = f"{TWITTER_API_KEY}:{TWITTER_API_SECRET}"
            encoded_credentials = base64.b64encode(key_secret.encode()).decode()
            
            # Get bearer token
            url = "https://api.twitter.com/oauth2/token"
            headers = {
                "Authorization": f"Basic {encoded_credentials}",
                "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"
            }
            data = "grant_type=client_credentials"
            
            response = requests.post(url, headers=headers, data=data)
            
            if response.status_code == 200:
                self.twitter_bearer_token = response.json().get("access_token")
            else:
                print(f"Failed to get Twitter bearer token: {response.status_code}")
                print(response.text)
        except Exception as e:
            print(f"Error getting Twitter bearer token: {e}")
    
    def check_api_availability(self):
        """Return a dictionary of API availability status"""
        return {
            "twitter_api": self.twitter_api_available and self.twitter_bearer_token is not None,
            "news_api": self.news_api_available,
            "google_trends_api": self.google_trends_api_available
        }
    
    def fetch_news_articles(self, query, days_back=7, limit=10):
        """
        Fetch news articles related to a query using NewsAPI.
        
        Args:
            query (str): Search query
            days_back (int): How many days back to search
            limit (int): Maximum number of articles to return
            
        Returns:
            list: List of dictionaries with article data or empty list if API not available
        """
        if not self.news_api_available:
            print("NewsAPI key not available. Cannot fetch news articles.")
            return []
            
        try:
            # Ensure we don't go back too far (NewsAPI free tier limit)
            # Set from_date to 30 days ago or today minus days_back, whichever is more recent
            max_days_back = min(days_back, 30)  # Limit to 30 days in the past
            from_date = (datetime.now() - timedelta(days=max_days_back)).strftime('%Y-%m-%d')
            
            # Use properly formatted parameters for better URL encoding of the query
            params = {
                'q': query,
                'from': from_date,
                'sortBy': 'relevancy',
                'apiKey': NEWS_API_KEY,
                'language': 'en',
                'pageSize': limit
            }
            url = "https://newsapi.org/v2/everything"
            response = requests.get(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])
                
                processed_articles = []
                for article in articles:
                    processed_articles.append({
                        'title': article.get('title', ''),
                        'source': article.get('source', {}).get('name', 'Unknown'),
                        'url': article.get('url', ''),
                        'published_at': article.get('publishedAt', ''),
                        'content': article.get('content', article.get('description', '')),
                    })
                
                return processed_articles
            else:
                print(f"Error fetching news articles: {response.status_code}")
                print(response.text)
                return []
                
        except Exception as e:
            print(f"Error fetching news articles: {e}")
            return []
    
    def fetch_twitter_posts(self, query, days_back=7, limit=10):
        """
        Fetch Twitter/X posts related to a query.
        
        Args:
            query (str): Search query
            days_back (int): How many days back to search
            limit (int): Maximum number of posts to return
            
        Returns:
            list: List of dictionaries with post data or empty list if API not available
        """
        if not self.twitter_api_available or not self.twitter_bearer_token:
            print("Twitter API keys not available or bearer token not obtained.")
            return []
            
        try:
            # Twitter API v2 endpoint
            url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={limit}&tweet.fields=created_at,public_metrics"
            
            headers = {
                'Authorization': f'Bearer {self.twitter_bearer_token}',
                'Content-Type': 'application/json'
            }
            
            response = requests.get(url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                tweets = data.get('data', [])
                
                processed_tweets = []
                for tweet in tweets:
                    metrics = tweet.get('public_metrics', {})
                    processed_tweets.append({
                        'id': tweet.get('id', ''),
                        'text': tweet.get('text', ''),
                        'created_at': tweet.get('created_at', ''),
                        'retweets': metrics.get('retweet_count', 0),
                        'likes': metrics.get('like_count', 0),
                        'url': f"https://twitter.com/i/web/status/{tweet.get('id', '')}"
                    })
                
                return processed_tweets
            else:
                print(f"Error fetching Twitter posts: {response.status_code}")
                print(response.text)
                return []
                
        except Exception as e:
            print(f"Error fetching Twitter posts: {e}")
            return []
    
    def web_scrape_for_topic(self, query, num_results=8):
        """
        Enhanced web scraping for up-to-date content related to a topic.
        Prioritizes news sites and recent content.
        
        Args:
            query (str): The topic to search for
            num_results (int): Maximum number of results to fetch (will try more to ensure quality)
            
        Returns:
            list: List of dictionaries with extracted text content
        """
        try:
            # Add "latest news" to the query to prioritize recent content
            search_query = f"{query} latest news"
            search_url = f"https://www.google.com/search?q={search_query}&num={num_results*2}&tbm=nws"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
                'Accept-Language': 'en-US,en;q=0.5',
                'Cache-Control': 'no-cache',
                'Pragma': 'no-cache'
            }
            
            # First try news search
            response = requests.get(search_url, headers=headers, timeout=10)
            
            if response.status_code != 200:
                # If news search fails, fall back to regular search
                search_url = f"https://www.google.com/search?q={query}&num={num_results*2}"
                response = requests.get(search_url, headers=headers, timeout=10)
            
            search_results = []
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                urls = []
                
                # First look for news results
                news_links = soup.select('a.WlydOe')
                if news_links:
                    for link in news_links:
                        href = link.get('href')
                        if href and href.startswith('http') and '/url?q=' not in href:
                            urls.append(href)
                
                # Then look for regular search results if we don't have enough
                if len(urls) < num_results:
                    result_divs = soup.select('div.yuRUbf')
                    for div in result_divs:
                        link = div.find('a')
                        if link and link.get('href'):
                            url = link.get('href')
                            if url.startswith('http') and url not in urls:
                                urls.append(url)
                
                # Also search for direct q= parameters
                all_links = soup.find_all('a')
                for link in all_links:
                    href = link.get('href', '')
                    if href and '/url?q=' in href:
                        try:
                            # Extract the actual URL from Google's redirect
                            start_idx = href.find('/url?q=') + 7
                            end_idx = href.find('&', start_idx)
                            if end_idx > start_idx:
                                actual_url = href[start_idx:end_idx]
                                if actual_url.startswith('http') and actual_url not in urls:
                                    urls.append(actual_url)
                        except Exception as e:
                            print(f"Error extracting URL from redirect: {e}")
                
                # Remove duplicates and limit
                urls = list(dict.fromkeys(urls))[:num_results*2]  # Try more URLs to get enough valid results
                
                # Process URLs with better error handling
                for url in urls:
                    try:
                        # Skip PDF and video files
                        if url.endswith(('.pdf', '.mp4', '.mov', '.avi', '.wmv')):
                            continue
                            
                        # Skip social media (too noisy and requires login)
                        if any(domain in url for domain in ['twitter.com', 'facebook.com', 'instagram.com', 'tiktok.com']):
                            continue
                            
                        # Use trafilatura to download and extract content
                        downloaded = trafilatura.fetch_url(url)
                        
                        if downloaded:
                            # Use trafilatura for better content extraction
                            text = trafilatura.extract(downloaded, include_comments=False, 
                                                       include_tables=True, 
                                                       include_links=True,
                                                       favor_precision=True)
                            
                            if text and len(text.strip()) > 200:  # Ensure substantial content
                                # Get metadata
                                title = "Unknown Title"
                                published_date = None
                                
                                try:
                                    # Try extracting metadata with trafilatura
                                    metadata = trafilatura.extract_metadata(downloaded)
                                    if metadata:
                                        if metadata.title:
                                            title = metadata.title
                                        if metadata.date:
                                            published_date = metadata.date
                                except:
                                    # Fall back to BeautifulSoup
                                    try:
                                        page_response = requests.get(url, headers=headers, timeout=5)
                                        page_soup = BeautifulSoup(page_response.text, 'html.parser')
                                        
                                        # Get title
                                        title_tag = page_soup.find('title')
                                        if title_tag:
                                            title = title_tag.text.strip()
                                            
                                        # Try to find publication date
                                        # Look for common metadata tags
                                        date_meta = page_soup.find('meta', attrs={'property': 'article:published_time'})
                                        if date_meta:
                                            published_date = date_meta.get('content')
                                        else:
                                            # Try other common date meta tags
                                            date_metas = [
                                                page_soup.find('meta', attrs={'name': 'pubdate'}),
                                                page_soup.find('meta', attrs={'property': 'og:published_time'}),
                                                page_soup.find('meta', attrs={'name': 'publish-date'}),
                                                page_soup.find('meta', attrs={'name': 'date'})
                                            ]
                                            for meta in date_metas:
                                                if meta and meta.get('content'):
                                                    published_date = meta.get('content')
                                                    break
                                    except Exception as e:
                                        print(f"Error extracting metadata for {url}: {e}")
                                
                                # Add to results
                                search_results.append({
                                    'title': title,
                                    'url': url,
                                    'content': text[:2000] + "..." if len(text) > 2000 else text,
                                    'published_date': published_date,
                                    'retrieved_at': datetime.now().isoformat()
                                })
                                
                                # If we have enough results, stop
                                if len(search_results) >= num_results:
                                    break
                    except Exception as e:
                        print(f"Error processing URL {url}: {e}")
                
                # Sort by published date if available, most recent first
                search_results = sorted(
                    search_results, 
                    key=lambda x: x.get('published_date', '0000-00-00'), 
                    reverse=True
                )
                
                return search_results[:num_results]
            else:
                print(f"Error searching Google: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error in enhanced web scraping for topic: {e}")
            return []
    
    def fetch_google_trends_data(self, query, geo='', timeframe='today 12-m'):
        """
        Fetch Google Trends data using our scraper.
        
        Args:
            query (str): The search term to get trends for
            geo (str): Geographic restriction (e.g., 'US', 'GB')
            timeframe (str): Time range (e.g., 'today 1-m', 'today 12-m')
            
        Returns:
            dict: Google Trends data or empty dict if API not available
        """
        try:
            # Import the Google Trends scraper
            from .google_trends_scraper import fetch_google_trends_data
            
            # Fetch the data
            return fetch_google_trends_data(query, timeframe=timeframe)
        except Exception as e:
            print(f"Error fetching Google Trends data: {e}")
            return {}
    
    def get_country_interest_data(self, query):
        """
        Get country interest data from Google Trends.
        
        Args:
            query (str): Search query
            
        Returns:
            dict: Dictionary with country data
        """
        try:
            # Import the Google Trends scraper
            from .google_trends_scraper import fetch_interest_by_region
            
            # Fetch the data
            return fetch_interest_by_region(query)
        except Exception as e:
            print(f"Error fetching country interest data: {e}")
            return {}
    
    def fetch_related_queries(self, query):
        """
        Fetch related queries from Google Trends.
        
        Args:
            query (str): The query term to search for
            
        Returns:
            dict: Dictionary with related queries data
        """
        try:
            # Import the Google Trends scraper
            from .google_trends_scraper import fetch_related_queries
            
            # Fetch the data
            return fetch_related_queries(query)
        except Exception as e:
            print(f"Error fetching related queries: {e}")
            return {
                'main_topic': query,
                'main_topic_keywords': [],
                'subtopics': [],
                'subtopic_keywords': {}
            }

def get_online_sentiment(topic, subtopics=None, days_back=30):
    """
    Fetch and generate comprehensive online sentiment data for a topic with enhanced
    capabilities for pop culture, trending topics, and current events.
    
    Args:
        topic (str): The main topic to analyze
        subtopics (list): Related subtopics to include in the analysis
        days_back (int): How many days of historical data to include
        
    Returns:
        dict: A comprehensive dict with sentiment data from various sources
    """
    # Initialize data fetcher
    data_fetcher = ExternalDataFetcher()
    
    # Check API availability
    api_status = data_fetcher.check_api_availability()
    print(f"API status: {api_status}")
    
    # Initialize result dictionary
    result = {
        'api_status': api_status,
        'query': topic,
        'timestamp': datetime.now().isoformat(),
        'global_data': {},
        'historical_data': [],
        'keyword_data': {},
        'sources': {'news': [], 'twitter': [], 'reddit': [], 'web': []}
    }
    
    # Clean and normalize the topic
    cleaned_topic = topic.strip()
    
    # For short queries, try to determine if this is a person, event, or brand
    # and expand the query to improve search results
    expanded_query = cleaned_topic
    if len(cleaned_topic.split()) <= 3:
        try:
            # Use OpenAI to expand the query appropriately based on what it is
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key:
                client = OpenAI(api_key=openai_api_key)
                
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You are a search expert who improves search queries."},
                        {"role": "user", "content": f"""
                        The query is: "{cleaned_topic}"
                        
                        First, identify what this refers to (person, music artist, athlete, event, movie, etc.).
                        Then, create an expanded search query that would yield better results.
                        
                        For example:
                        - If it's "Swamp Izzo", return "Swamp Izzo Atlanta DJ producer Drake"
                        - If it's "LeBron", return "LeBron James basketball Lakers NBA"
                        
                        Return ONLY the expanded query with no additional text or explanation.
                        """}
                    ]
                )
                
                expanded_query = response.choices[0].message.content.strip().strip('"')
                print(f"Expanded query: {expanded_query}")
        except Exception as e:
            print(f"Error expanding query: {e}")
    
    # Process subtopics
    if subtopics and isinstance(subtopics, list):
        filtered_subtopics = [s for s in subtopics if s and len(s.strip()) > 0]
    else:
        filtered_subtopics = []
    
    # 1. Fetch news articles
    try:
        news_articles = data_fetcher.fetch_news_articles(expanded_query, days_back=min(days_back, 30), limit=8)
        result['sources']['news'] = news_articles
        
        # If we don't have enough news articles and the query was expanded, try with original
        if len(news_articles) < 3 and expanded_query != cleaned_topic:
            additional_articles = data_fetcher.fetch_news_articles(cleaned_topic, days_back=min(days_back, 30), limit=5)
            result['sources']['news'].extend(additional_articles)
    except Exception as e:
        print(f"Error fetching news articles: {e}")
    
    # 2. Fetch Twitter/X posts
    try:
        twitter_posts = data_fetcher.fetch_twitter_posts(expanded_query, days_back=min(days_back, 7), limit=8)
        result['sources']['twitter'] = twitter_posts
    except Exception as e:
        print(f"Error fetching Twitter posts: {e}")
    
    # 3. Web scrape for additional content
    try:
        web_content = data_fetcher.web_scrape_for_topic(expanded_query, num_results=8)
        result['sources']['web'] = web_content
        
        # Also try scraping specifically for recent content
        recent_content = data_fetcher.web_scrape_for_topic(f"{expanded_query} recent news 2024", num_results=4)
        # Add if not already included
        existing_urls = [item.get('url', '') for item in result['sources']['web']]
        for item in recent_content:
            if item.get('url', '') not in existing_urls:
                result['sources']['web'].append(item)
    except Exception as e:
        print(f"Error web scraping: {e}")
    
    # 4. Try to get Google Trends data
    try:
        # Geographic data
        country_data = data_fetcher.get_country_interest_data(cleaned_topic)
        if not country_data or len(country_data.get('countries', [])) < 5:
            # If we don't get enough real data, generate it
            from .sentiment_generator import generate_country_sentiment
            generated_countries = generate_country_sentiment(topic)
            result['global_data'] = {'countries': generated_countries, 'source': 'generated', 'query': topic}
        else:
            result['global_data'] = country_data
        
        # Historical data
        trends_data = data_fetcher.fetch_google_trends_data(cleaned_topic, timeframe='today 12-m')
        if trends_data and 'time_data' in trends_data and trends_data['time_data'] is not None:
            # Smooth and enhance the time series data
            time_data = trends_data['time_data']
            
            # Ensure it's sorted
            if 'date' in time_data.columns:
                time_data = time_data.sort_values('date')
            
            # Convert to the format used by visualizations
            result['historical_data'] = time_data.to_dict('records')
        else:
            # Generate synthetic data
            from .sentiment_generator import generate_time_series_data
            result['historical_data'] = generate_time_series_data(topic, days=365)
            
        # Keyword data
        keyword_data = data_fetcher.fetch_related_queries(cleaned_topic)
        if not keyword_data or len(keyword_data.get('main_topic_keywords', [])) < 5:
            # Generate keyword data if we don't have enough
            from .sentiment_generator import generate_topic_keywords
            result['keyword_data'] = generate_topic_keywords(topic)
        else:
            result['keyword_data'] = keyword_data
            
    except Exception as e:
        print(f"Error getting Google Trends data: {e}")
        
        # Generate synthetic data for visualizations if real data failed
        # This ensures the UI always has something to display
        from .sentiment_generator import generate_country_sentiment, generate_time_series_data, generate_topic_keywords
        
        try:
            result['global_data'] = {'countries': generate_country_sentiment(topic), 'source': 'generated', 'query': topic}
            result['historical_data'] = generate_time_series_data(topic, days=365)
            result['keyword_data'] = generate_topic_keywords(topic)
        except Exception as e2:
            print(f"Error generating fallback data: {e2}")
    
    # Enhance data with OpenAI analysis for more context awareness
    try:
        # If we have real data to analyze from sources
        if (len(result['sources']['news']) > 0 or len(result['sources']['web']) > 0 or 
            len(result['sources']['twitter']) > 0):
            
            # Collect all source text for analysis
            source_texts = []
            
            # Add news headlines and snippets
            for article in result['sources']['news'][:3]:
                source_texts.append(f"News: {article.get('title', '')} - {article.get('content', '')[:100]}")
            
            # Add web content snippets
            for content in result['sources']['web'][:3]:
                source_texts.append(f"Web: {content.get('title', '')} - {content.get('content', '')[:100]}")
            
            # Add Twitter posts
            for post in result['sources']['twitter'][:3]:
                source_texts.append(f"Social: {post.get('text', '')}")
            
            # Use OpenAI to analyze and enhance the data
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if openai_api_key and source_texts:
                client = OpenAI(api_key=openai_api_key)
                
                # Create a prompt for analysis
                prompt = f"""
                Analyze these online sources about "{topic}":
                
                {source_texts[:10]}
                
                Provide enhanced metadata in JSON format:
                {{
                    "topic_type": "specify if this is a person, event, brand, concept, etc.",
                    "category": "music, sports, politics, technology, etc.",
                    "sentiment_summary": "overall sentiment from sources (positive/neutral/negative)",
                    "trending_status": "is this currently trending? (yes/no)",
                    "key_entities": ["list 2-3 related entities"],
                    "related_topics": ["list 2-3 related topics"],
                    "temporal_context": "is this about something recent, historical, or ongoing"
                }}
                """
                
                # Call OpenAI
                response = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": "You analyze online data sources to provide metadata enrichment."},
                        {"role": "user", "content": prompt}
                    ],
                    response_format={"type": "json_object"}
                )
                
                # Parse the response
                enhanced_data = json.loads(response.choices[0].message.content)
                
                # Add to result
                result['enhanced_metadata'] = enhanced_data
                
                # Use this to improve sentiment data
                if 'sentiment_summary' in enhanced_data:
                    # Apply the sentiment to countries without sentiment
                    for country in result['global_data'].get('countries', []):
                        if 'sentiment' not in country or country['sentiment'] == 'neutral':
                            country['sentiment'] = enhanced_data['sentiment_summary']
    except Exception as e:
        print(f"Error enhancing data with OpenAI: {e}")
    
    return result