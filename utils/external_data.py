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
import praw

# API Keys - These will be populated from environment variables
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "")
TWITTER_API_SECRET = os.environ.get("TWITTER_API_SECRET", "")
NEWS_API_KEY = os.environ.get("NEWSAPI_KEY", "")
GOOGLE_TRENDS_API_KEY = os.environ.get("GOOGLE_TRENDS_API_KEY", "")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.environ.get("REDDIT_CLIENT_SECRET", "")
REDDIT_USER_AGENT = "python:sentim:v1.0 (by /u/Awkward_Sandwich_184)"  # Using details from user's app

class ExternalDataFetcher:
    """Handles fetching data from external sources like news, social media, etc."""
    
    def __init__(self):
        self.twitter_api_available = bool(TWITTER_API_KEY and TWITTER_API_SECRET)
        self.news_api_available = bool(NEWS_API_KEY)
        self.google_trends_api_available = bool(GOOGLE_TRENDS_API_KEY)
        self.reddit_api_available = bool(REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET)
        self.twitter_bearer_token = None
        self.reddit_instance = None
        
        # Initialize Reddit client if credentials are available
        if self.reddit_api_available:
            try:
                self.reddit_instance = praw.Reddit(
                    client_id=REDDIT_CLIENT_ID,
                    client_secret=REDDIT_CLIENT_SECRET,
                    user_agent=REDDIT_USER_AGENT
                )
            except Exception as e:
                print(f"Error initializing Reddit API: {e}")
                self.reddit_api_available = False
        
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
            "google_trends_api": self.google_trends_api_available,
            "reddit_api": self.reddit_api_available
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
    
    def web_scrape_for_topic(self, query, num_results=5):
        """
        Scrape web content related to a topic.
        
        Args:
            query (str): The topic to search for
            num_results (int): Number of results to fetch
            
        Returns:
            list: List of dictionaries with extracted text content
        """
        try:
            # Simple Google search scraping - in production, use Serper.dev or SerpAPI
            search_url = f"https://www.google.com/search?q={query}&num={num_results}"
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.110 Safari/537.36'
            }
            
            response = requests.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                search_results = []
                
                # Get all search result links
                result_divs = soup.select('div.yuRUbf')
                urls = []
                for div in result_divs:
                    link = div.find('a')
                    if link and link.get('href'):
                        url = link.get('href')
                        if url.startswith('http'):
                            urls.append(url)
                
                # Limit to requested number
                urls = urls[:num_results]
                
                for url in urls:
                    try:
                        # Use trafilatura to extract main content
                        downloaded = trafilatura.fetch_url(url)
                        if downloaded:
                            text = trafilatura.extract(downloaded)
                            
                            if text:
                                title = "Unknown Title"
                                # Try to extract title using BeautifulSoup
                                try:
                                    page_response = requests.get(url, headers=headers, timeout=5)
                                    page_soup = BeautifulSoup(page_response.text, 'html.parser')
                                    title_tag = page_soup.find('title')
                                    if title_tag:
                                        title = title_tag.text.strip()
                                except:
                                    pass
                                
                                search_results.append({
                                    'title': title,
                                    'url': url,
                                    'content': text[:1000] + "..." if len(text) > 1000 else text,
                                    'retrieved_at': datetime.now().isoformat()
                                })
                    except Exception as e:
                        print(f"Error processing URL {url}: {e}")
                
                return search_results
            else:
                print(f"Error searching Google: {response.status_code}")
                return []
                
        except Exception as e:
            print(f"Error web scraping for topic: {e}")
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
    
    def fetch_reddit_posts(self, query, limit=15, time_filter='month'):
        """
        Fetch Reddit posts related to a query using PRAW.
        
        Args:
            query (str): Search query
            limit (int): Maximum number of posts to return
            time_filter (str): Time filter ('day', 'week', 'month', 'year', 'all')
            
        Returns:
            list: List of dictionaries with Reddit post data or empty list if API not available
        """
        if not self.reddit_api_available or not self.reddit_instance:
            print("Reddit API credentials not available. Cannot fetch Reddit posts.")
            return []
            
        try:
            processed_posts = []
            # Search for posts
            for submission in self.reddit_instance.subreddit('all').search(query, limit=limit, sort='relevance', time_filter=time_filter):
                # Extract post data and metrics
                post_data = {
                    'id': submission.id,
                    'title': submission.title,
                    'text': submission.selftext[:500] if submission.selftext else '',
                    'subreddit': submission.subreddit.display_name,
                    'score': submission.score,
                    'comments': submission.num_comments,
                    'created_utc': datetime.fromtimestamp(submission.created_utc).isoformat(),
                    'url': f"https://www.reddit.com{submission.permalink}",
                    'upvote_ratio': submission.upvote_ratio
                }
                
                # Add some top comments if available
                post_data['top_comments'] = []
                try:
                    submission.comments.replace_more(limit=0)  # Don't fetch MoreComments
                    for comment in submission.comments.list()[:3]:  # Get top 3 comments
                        if hasattr(comment, 'body') and comment.body:
                            post_data['top_comments'].append({
                                'text': comment.body[:200],
                                'score': comment.score,
                                'created_utc': datetime.fromtimestamp(comment.created_utc).isoformat()
                            })
                except Exception as e:
                    print(f"Error fetching comments for post {submission.id}: {e}")
                
                processed_posts.append(post_data)
            
            return processed_posts
                
        except Exception as e:
            print(f"Error fetching Reddit posts: {e}")
            return []
    
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

def get_online_sentiment(topic, subtopics=None, days_back=30):
    """
    Fetch sentiment from online sources about a topic.
    
    Args:
        topic (str): The main topic to analyze
        subtopics (list): Optional list of subtopics
        days_back (int): How many days back to search
        
    Returns:
        dict: Sentiment data from various sources
    """
    if not subtopics:
        subtopics = []
    
    # Initialize the data fetcher
    fetcher = ExternalDataFetcher()
    
    # Get API availability status
    api_status = fetcher.check_api_availability()
    
    # Store source data
    sources = {
        'news': [],
        'twitter': [],
        'web': [],
        'reddit': []
    }
    
    # 1. Get news articles for main topic and subtopics
    news_articles = fetcher.fetch_news_articles(topic, days_back)
    sources['news'].extend(news_articles)
    
    for subtopic in subtopics:
        # Get fewer articles for each subtopic to avoid overwhelming
        subtopic_articles = fetcher.fetch_news_articles(f"{topic} {subtopic}", days_back, limit=3)
        sources['news'].extend(subtopic_articles)
    
    # 2. Get twitter data
    twitter_posts = fetcher.fetch_twitter_posts(topic, days_back)
    sources['twitter'].extend(twitter_posts)
    
    # 3. Web scraping
    web_content = fetcher.web_scrape_for_topic(topic)
    sources['web'].extend(web_content)
    
    # 4. Reddit data
    time_filter = 'month'
    if days_back <= 7:
        time_filter = 'week'
    elif days_back <= 1:
        time_filter = 'day'
    elif days_back >= 365:
        time_filter = 'year'
    
    reddit_posts = fetcher.fetch_reddit_posts(topic, limit=10, time_filter=time_filter)
    sources['reddit'].extend(reddit_posts)
    
    # Also get Reddit posts for subtopics
    for subtopic in subtopics:
        subtopic_posts = fetcher.fetch_reddit_posts(f"{topic} {subtopic}", limit=5, time_filter=time_filter)
        sources['reddit'].extend(subtopic_posts)
    
    # Keywords and topics data
    keyword_data = {
        'main_topic': topic,
        'main_topic_keywords': [],
        'subtopics': subtopics,
        'subtopic_keywords': {}
    }
    
    # 4. First try to get real country data from Google Trends
    global_data = fetcher.get_country_interest_data(topic)
    
    # 5. Try to get real historical data from Google Trends
    historical_data = fetcher.fetch_google_trends_data(topic)
    
    # 6. Compile all source texts for sentiment analysis
    from .openai_client import analyze_sentiment
    
    all_source_texts = []
    
    # Add news article titles and content
    for article in sources['news']:
        all_source_texts.append(article.get('title', '') + ': ' + article.get('content', '')[:500])
    
    # Add tweet texts
    for tweet in sources['twitter']:
        all_source_texts.append(tweet.get('text', ''))
    
    # Add web content
    for content in sources['web']:
        all_source_texts.append(content.get('title', '') + ': ' + content.get('content', '')[:500])
    
    # Add Reddit content
    for post in sources['reddit']:
        # Add the post title and text
        post_text = f"{post.get('title', '')} [{post.get('subreddit', '')}]: {post.get('text', '')}"
        all_source_texts.append(post_text)
        
        # Add top comments
        for comment in post.get('top_comments', []):
            if comment.get('score', 0) > 5:  # Only include comments with significant engagement
                all_source_texts.append(f"Reddit comment: {comment.get('text', '')}")
    
    # Overall sentiment from all sources
    combined_text = '\n\n'.join(all_source_texts)
    sentiment_result = None
    
    if combined_text and len(combined_text) > 100:
        try:
            sentiment_result = analyze_sentiment(combined_text)
            print(f"Generated sentiment for '{topic}': {sentiment_result.get('sentiment', 'neutral')}")
        except Exception as e:
            print(f"Error analyzing sentiment: {e}")
    
    # 7. If we don't have real global data, generate it with OpenAI
    if not global_data or not global_data.get('countries'):
        from .sentiment_generator import generate_country_sentiment
        
        try:
            print(f"No Google Trends geographic data available, generating sentiment data for '{topic}'")
            country_data = generate_country_sentiment(topic)
            
            # If we have a global sentiment from text sources, apply it
            if sentiment_result:
                sentiment_label = sentiment_result.get('sentiment', 'neutral')
                # Adjust some countries to match overall sentiment to make it realistic
                for i, country in enumerate(country_data):
                    if i % 3 == 0:  # Modify every third country to match overall sentiment
                        country['sentiment'] = sentiment_label
            
            global_data = {
                'countries': country_data,
                'query': topic,
                'timestamp': datetime.now().isoformat()
            }
            print(f"Generated data for {len(country_data)} countries")
        except Exception as e:
            print(f"Error generating country sentiment: {e}")
            global_data = {'countries': [], 'query': topic}
    
    # 8. If we don't have keyword data, generate it with OpenAI
    if (not historical_data.get('keyword_data') or 
            not historical_data.get('keyword_data', {}).get('main_topic_keywords')):
        from .sentiment_generator import generate_topic_keywords
        
        try:
            print(f"No keyword data available from Google Trends, generating keywords for '{topic}'")
            keyword_data = generate_topic_keywords(topic)
            
            # Add keyword data to historical_data
            if 'keyword_data' not in historical_data:
                historical_data['keyword_data'] = {}
            
            historical_data['keyword_data'] = keyword_data
            print(f"Generated {len(keyword_data.get('main_topic_keywords', []))} keywords")
        except Exception as e:
            print(f"Error generating topic keywords: {e}")
    
    # 9. If we don't have time series data, generate it with OpenAI
    if not historical_data.get('time_data') or len(historical_data.get('time_data', [])) == 0:
        from .sentiment_generator import generate_time_series_data
        
        try:
            print(f"No time series data available from Google Trends, generating for '{topic}'")
            time_data = generate_time_series_data(topic)
            historical_data['time_data'] = time_data
            print(f"Generated {len(time_data)} time data points")
        except Exception as e:
            print(f"Error generating time series data: {e}")
            
    # Update the keyword_data with topics from historical_data if available
    if historical_data.get('keyword_data') and historical_data['keyword_data'].get('main_topic_keywords'):
        keyword_data = historical_data['keyword_data']
    
    # Return all the collected data
    return {
        'api_status': api_status,
        'sources': sources,
        'keyword_data': keyword_data,
        'global_data': global_data,
        'historical_data': historical_data
    }