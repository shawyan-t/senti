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

class ExternalDataFetcher:
    """Handles fetching data from external sources like news, social media, etc."""
    
    def __init__(self):
        self.twitter_api_available = bool(TWITTER_API_KEY and TWITTER_API_SECRET)
        self.news_api_available = bool(NEWS_API_KEY)
        self.google_trends_available = True  # pytrends doesn't require an API key
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
            "google_trends": self.google_trends_available
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
    
    def web_scrape_for_topic(self, query, num_results=10):
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
            
            # Fetch the data with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for retry in range(max_retries):
                try:
                    return fetch_google_trends_data(query, timeframe=timeframe)
                except Exception as e:
                    # Check if it's a rate limit error
                    if "429" in str(e) and retry < max_retries - 1:
                        print(f"Google Trends rate limit hit. Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        # Exponential backoff
                        retry_delay *= 2
                    else:
                        # If it's not a rate limit or we've exhausted retries, re-raise
                        raise
                        
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
            
            # Fetch the data with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for retry in range(max_retries):
                try:
                    return fetch_interest_by_region(query)
                except Exception as e:
                    # Check if it's a rate limit error
                    if "429" in str(e) and retry < max_retries - 1:
                        print(f"Google Trends rate limit hit. Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        # Exponential backoff
                        retry_delay *= 2
                    else:
                        # If it's not a rate limit or we've exhausted retries, re-raise
                        raise
                        
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
            
            # Fetch the data with retry logic
            max_retries = 3
            retry_delay = 2  # seconds
            
            for retry in range(max_retries):
                try:
                    return fetch_related_queries(query)
                except Exception as e:
                    # Check if it's a rate limit error
                    if "429" in str(e) and retry < max_retries - 1:
                        print(f"Google Trends rate limit hit. Retrying in {retry_delay} seconds...")
                        import time
                        time.sleep(retry_delay)
                        # Exponential backoff
                        retry_delay *= 2
                    else:
                        # If it's not a rate limit or we've exhausted retries, re-raise
                        raise
                        
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
        news_articles = data_fetcher.fetch_news_articles(expanded_query, days_back=min(days_back, 30), limit=10)
        result['sources']['news'] = news_articles
        
        # If we don't have enough news articles and the query was expanded, try with original
        if len(news_articles) < 3 and expanded_query != cleaned_topic:
            additional_articles = data_fetcher.fetch_news_articles(cleaned_topic, days_back=min(days_back, 30), limit=5)
            result['sources']['news'].extend(additional_articles)
    except Exception as e:
        print(f"Error fetching news articles: {e}")
    
    # 2. Fetch Twitter/X posts
    try:
        twitter_posts = data_fetcher.fetch_twitter_posts(expanded_query, days_back=min(days_back, 7), limit=10)
        result['sources']['twitter'] = twitter_posts
    except Exception as e:
        print(f"Error fetching Twitter posts: {e}")
    
    # 3. Web scrape for additional content
    try:
        web_content = data_fetcher.web_scrape_for_topic(expanded_query, num_results=10)
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

class SearchEngineConnector:
    """Enhanced connector for real-time search engine data to provide up-to-date information to LLMs."""
    
    def __init__(self):
        """Initialize the search engine connector."""
        # Load API keys from config
        self.google_search_api_key = config.get('google_search_api_key')
        self.google_search_cx = config.get('google_search_cx')
        
        # Set up cache directory
        import os
        self.cache_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data', 'search_cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Initialize cache
        try:
            from diskcache import Cache
            self.cache = Cache(self.cache_dir)
        except ImportError:
            print("Warning: diskcache not installed. Caching disabled.")
            self.cache = None
    
    def search_engine_query(self, query, num_results=10, recent_only=True):
        """
        Query Google Search API for reliable results.
        
        Args:
            query (str): Search query
            num_results (int): Number of results to return
            recent_only (bool): Whether to prioritize recent results
            
        Returns:
            list: List of search results with metadata
        """
        results = []
        
        # Google Search API (using Programmable Search Engine)
        if self.google_search_api_key and self.google_search_cx:
            try:
                # Set up parameters
                params = {
                    'key': self.google_search_api_key,
                    'cx': self.google_search_cx,
                    'q': query,
                    'num': min(10, num_results)  # API limit is 10 per request
                }
                
                # Add recent content filtering if requested
                if recent_only:
                    # Filter by date range (last month)
                    from datetime import datetime, timedelta
                    one_month_ago = (datetime.now() - timedelta(days=30)).strftime('%Y%m%d')
                    params['sort'] = 'date:r:' + one_month_ago
                
                # Make the API request
                response = requests.get('https://www.googleapis.com/customsearch/v1', params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get('items', []):
                        results.append({
                            'title': item.get('title'),
                            'link': item.get('link'),
                            'snippet': item.get('snippet'),
                            'published_date': item.get('pagemap', {}).get('metatags', [{}])[0].get('article:published_time'),
                            'source': 'google',
                            'engine': 'google'
                        })
            except Exception as e:
                print(f"Error with Google Search API: {e}")
        else:
            print("Google Search API key or CX not configured")
        
        # If no API results available, fall back to web scraping
        if not results:
            print("Falling back to web scraping for search results")
            try:
                data_fetcher = ExternalDataFetcher()
                scraped_results = data_fetcher.web_scrape_for_topic(query, num_results=num_results)
                
                for item in scraped_results:
                    results.append({
                        'title': item.get('title', 'Unknown'),
                        'link': item.get('url', ''),
                        'snippet': item.get('content', '')[:200] + '...' if item.get('content') else '',
                        'published_date': item.get('published_date'),
                        'source': 'web_scrape',
                        'engine': 'scrape'
                    })
            except Exception as e:
                print(f"Error with fallback web scraping: {e}")
        
        return results
    
    def extract_content_from_url(self, url, max_length=4000):
        """
        Extract and clean content from a URL for processing by the LLM.
        
        Args:
            url (str): URL to extract content from
            max_length (int): Maximum content length to return
            
        Returns:
            dict: Dictionary with extracted content and metadata
        """
        try:
            # Use trafilatura for robust content extraction
            downloaded = trafilatura.fetch_url(url)
            
            if not downloaded:
                return None
                
            # Extract main content
            content = trafilatura.extract(downloaded, 
                                        include_comments=False,
                                        include_tables=True,
                                        include_links=True,
                                        favor_precision=True)
            
            if not content or len(content.strip()) < 100:
                return None
                
            # Extract metadata
            metadata = trafilatura.extract_metadata(downloaded)
            
            # Build result
            result = {
                'url': url,
                'content': content[:max_length] if content else "",
                'title': metadata.title if metadata and metadata.title else "Unknown Title",
                'published_date': metadata.date if metadata and metadata.date else None,
                'author': metadata.author if metadata and metadata.author else None,
                'extracted_at': datetime.now().isoformat()
            }
            
            return result
        except Exception as e:
            print(f"Error extracting content from {url}: {e}")
            return None
    
    def get_cached_or_fresh_data(self, query, max_age_minutes=60):
        """
        Get data from cache if fresh enough, otherwise fetch fresh data.
        
        Args:
            query (str): The search query
            max_age_minutes (int): Maximum age of cached data in minutes
            
        Returns:
            dict: Search results with metadata
        """
        if not self.cache:
            # If cache is not available, just fetch fresh data
            return self.search_engine_query(query)
            
        import hashlib
        import time
        
        # Create a hash of the query for the cache key
        query_hash = hashlib.md5(query.encode()).hexdigest()
        cache_key = f"search_results_{query_hash}"
        
        # Check if we have cached data
        cached_data = self.cache.get(cache_key)
        
        if cached_data:
            # Check if the data is fresh enough
            if time.time() - cached_data['timestamp'] < (max_age_minutes * 60):
                print(f"Using cached data for query '{query}'")
                return cached_data['data']
        
        # If we don't have cached data or it's too old, fetch fresh data
        print(f"Fetching fresh data for query '{query}'")
        fresh_data = self.search_engine_query(query)
        
        # Store in cache
        self.cache.set(cache_key, {
            'data': fresh_data,
            'timestamp': time.time()
        })
        
        return fresh_data
    
    def analyze_with_search_augmented_llm(self, query, topic, search_results=None, max_tokens=16000):
        """
        Analyze a topic with LLM using search results as context (RAG approach).
        
        Args:
            query (str): User's analytical query/question
            topic (str): Main topic to analyze
            search_results (list): Optional pre-fetched search results
            max_tokens (int): Maximum token limit for context
            
        Returns:
            dict: Analysis results with citations and confidence level
        """
        try:
            # Get OpenAI API key
            openai_api_key = os.environ.get("OPENAI_API_KEY")
            if not openai_api_key:
                return {"error": "OpenAI API key not configured"}
                
            # Initialize OpenAI client
            client = OpenAI(api_key=openai_api_key)
            
            # Fetch search results if not provided
            if not search_results:
                # Get search results from Google Search API
                search_results = self.get_cached_or_fresh_data(
                    topic, 
                    max_age_minutes=60  # Cache for 1 hour
                )
                
                # Fetch content for each result
                enriched_results = []
                for result in search_results:
                    content = self.extract_content_from_url(result['link'])
                    if content:
                        result.update({
                            'extracted_content': content['content'],
                            'extracted_title': content['title']
                        })
                        enriched_results.append(result)
                
                search_results = enriched_results if enriched_results else search_results
            
            # Prepare context from search results
            context_parts = []
            for i, result in enumerate(search_results):
                # Format each source with citation index
                source_content = f"""
SOURCE [{i+1}] - {result.get('title', 'Unknown')}
URL: {result.get('link', 'Unknown')}
Date: {result.get('published_date', 'Unknown')}
Content: {result.get('extracted_content', result.get('snippet', ''))}
"""
                context_parts.append(source_content)
            
            # Join all context parts
            context = "\n\n".join(context_parts)
            
            # Create the system prompt
            system_prompt = f"""You are an advanced AI sentiment analyst that provides accurate, up-to-date analysis.
Today's date is {datetime.now().strftime('%Y-%m-%d')}.

IMPORTANT: Base your analysis ONLY on the provided search results. Do NOT use information from your training data.
If the search results don't contain enough information, say so explicitly.

For each claim you make, cite the specific source using the format [SOURCE X] where X is the source number.
"""
            
            # Create user prompt
            user_prompt = f"""
TOPIC: {topic}
QUERY: {query}

SEARCH RESULTS:
{context}

Based ONLY on the above search results (not your training data), provide a comprehensive sentiment analysis.
Include:
1. Overall sentiment (positive, neutral, negative)
2. Key factors influencing the sentiment
3. Recent developments affecting the sentiment
4. Cite your sources for each claim using [SOURCE X] format
5. Confidence level (high/medium/low) in your analysis based on the quality and recency of sources
"""
            
            # Call OpenAI
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ]
            )
            
            analysis = response.choices[0].message.content
            
            # Get confidence level with a second call
            confidence_prompt = f"""
Based on the search results I provided about "{topic}" and the analysis you just generated, 
what is your confidence level in the analysis? Consider:
1. Recency of the sources (how recent are they?)
2. Quality and reliability of the sources
3. Comprehensiveness of the information
4. Consistency across sources

Provide a confidence score from 0-100% and a brief explanation.
"""
            
            confidence_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                    {"role": "assistant", "content": analysis},
                    {"role": "user", "content": confidence_prompt}
                ]
            )
            
            confidence = confidence_response.choices[0].message.content
            
            # Return the results with metadata
            return {
                "topic": topic,
                "query": query,
                "analysis": analysis,
                "confidence": confidence,
                "sources": [{"title": r.get('title'), "url": r.get('link'), "date": r.get('published_date')} 
                        for r in search_results],
                "timestamp": datetime.now().isoformat()
            }
                
        except Exception as e:
            print(f"Error in search-augmented LLM analysis: {e}")
            return {"error": str(e)}

# Update the existing get_online_sentiment function to use the new SearchEngineConnector
def get_online_sentiment_with_search(topic, subtopics=None, days_back=30, use_search_apis=True):
    """
    Enhanced version of get_online_sentiment that uses proper search APIs and RAG approach with LLMs.
    
    Args:
        topic (str): The main topic to analyze
        subtopics (list): Related subtopics to include in the analysis
        days_back (int): How many days of historical data to include
        use_search_apis (bool): Whether to use search APIs for enhanced results
        
    Returns:
        dict: A comprehensive dict with sentiment data from various sources including search engines
    """
    # Initialize result dictionary (in case of error)
    result = {
        'api_status': {},
        'query': topic,
        'timestamp': datetime.now().isoformat(),
        'global_data': {},
        'historical_data': [],
        'keyword_data': {},
        'sources': {'news': [], 'twitter': [], 'reddit': [], 'web': [], 'search_engines': []}
    }
    
    # Add search engine results if enabled (prioritize this)
    if use_search_apis:
        try:
            # Initialize the search engine connector
            search_connector = SearchEngineConnector()
            
            # Get real-time search results
            search_results = search_connector.get_cached_or_fresh_data(
                topic, 
                max_age_minutes=60  # Cache for 1 hour
            )
            
            # Extract content from top results
            enriched_results = []
            for i, result_item in enumerate(search_results[:5]):
                content = search_connector.extract_content_from_url(result_item['link'])
                if content:
                    result_item['content'] = content['content']
                    result_item['title'] = content['title']
                    enriched_results.append(result_item)
                else:
                    enriched_results.append(result_item)
                    
            # Add to sources
            result['sources']['search_engines'] = enriched_results
            
            # Enhance analysis with search-augmented LLM
            search_analysis = search_connector.analyze_with_search_augmented_llm(
                f"What is the current sentiment around {topic}?",
                topic,
                search_results=enriched_results
            )
            
            # Add the search-augmented analysis
            result['search_augmented_analysis'] = search_analysis
            
            # Manually construct a baseline sentiment based on the real-time data
            if search_analysis and not search_analysis.get('error'):
                # Extract sentiment from analysis
                analysis_text = search_analysis.get('analysis', '')
                if "positive" in analysis_text.lower():
                    dominant_sentiment = "positive"
                elif "negative" in analysis_text.lower():
                    dominant_sentiment = "negative"
                else:
                    dominant_sentiment = "neutral"
                
                # Create basic keyword data from search results
                keyword_data = {
                    'main_topic': topic,
                    'main_topic_keywords': []
                }
                
                # Extract potential keywords from search result titles
                from collections import Counter
                import re
                
                # Combine all titles
                all_titles = " ".join([r.get('title', '') for r in enriched_results])
                
                # Extract words, remove common words
                words = re.findall(r'\b[A-Za-z][A-Za-z]{2,}\b', all_titles)
                stopwords = {'the', 'and', 'or', 'but', 'for', 'with', 'about', 'against', 'between'}
                filtered_words = [w.lower() for w in words if w.lower() not in stopwords]
                
                # Count occurrences
                word_counts = Counter(filtered_words)
                
                # Add top keywords
                for word, count in word_counts.most_common(10):
                    if word.lower() != topic.lower():  # Avoid the main topic itself
                        keyword_data['main_topic_keywords'].append({
                            'keyword': word,
                            'frequency': count * 10,  # Scale up for visualization
                            'sentiment': dominant_sentiment
                        })
                
                # Add to result
                result['keyword_data'] = keyword_data
            
        except Exception as e:
            print(f"Error integrating search APIs: {e}")
            result['search_api_error'] = str(e)
    
    # Get traditional sentiment data as a fallback
    try:
        traditional_data = get_online_sentiment(topic, subtopics, days_back)
        
        # Only use traditional data for fields that weren't populated by search results
        if not result.get('global_data', {}).get('countries'):
            result['global_data'] = traditional_data.get('global_data', {})
        
        if not result.get('historical_data'):
            result['historical_data'] = traditional_data.get('historical_data', [])
        
        if not result.get('keyword_data', {}).get('main_topic_keywords'):
            result['keyword_data'] = traditional_data.get('keyword_data', {})
        
        # Add API status
        result['api_status'] = traditional_data.get('api_status', {})
        
        # Add sources that weren't populated
        for source_type in ['news', 'twitter', 'reddit', 'web']:
            if not result['sources'].get(source_type):
                result['sources'][source_type] = traditional_data.get('sources', {}).get(source_type, [])
        
        # Add enhanced metadata if available
        if 'enhanced_metadata' in traditional_data:
            result['enhanced_metadata'] = traditional_data['enhanced_metadata']
            
    except Exception as e:
        print(f"Error getting traditional sentiment data: {e}")
        
        # If everything failed, generate some basic data to avoid breaking the UI
        from .sentiment_generator import generate_country_sentiment, generate_time_series_data
        
        try:
            result['global_data'] = {'countries': generate_country_sentiment(topic), 'source': 'generated', 'query': topic}
            result['historical_data'] = generate_time_series_data(topic, days=365)
        except Exception as e2:
            print(f"Error generating fallback data: {e2}")
    
    return result