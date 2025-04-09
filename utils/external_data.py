"""
Module for fetching external data from various online sources for sentiment analysis.
"""
import os
import json
import re
import time
import random
from datetime import datetime, timedelta
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import pycountry
import trafilatura

# API Keys - These will be populated from environment variables
TWITTER_API_KEY = os.environ.get("TWITTER_API_KEY", "")
REDDIT_CLIENT_ID = os.environ.get("REDDIT_CLIENT_ID", "")
REDDIT_SECRET = os.environ.get("REDDIT_SECRET", "")
NEWS_API_KEY = os.environ.get("NEWS_API_KEY", "")

class ExternalDataFetcher:
    """Handles fetching data from external sources like news, social media, etc."""
    
    def __init__(self):
        self.twitter_api_available = bool(TWITTER_API_KEY)
        self.reddit_api_available = bool(REDDIT_CLIENT_ID and REDDIT_SECRET)
        self.news_api_available = bool(NEWS_API_KEY)
    
    def check_api_availability(self):
        """Return a dictionary of API availability status"""
        return {
            "twitter_api": self.twitter_api_available,
            "reddit_api": self.reddit_api_available,
            "news_api": self.news_api_available
        }
    
    def fetch_news_articles(self, query, days_back=7, limit=10):
        """
        Fetch news articles related to a query using NewsAPI.
        
        Args:
            query (str): Search query
            days_back (int): How many days back to search
            limit (int): Maximum number of articles to return
            
        Returns:
            list: List of dictionaries with article data
        """
        if not self.news_api_available:
            return self._get_fallback_news_data(query)
            
        try:
            from_date = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
            
            url = f"https://newsapi.org/v2/everything?q={query}&from={from_date}&sortBy=relevancy&apiKey={NEWS_API_KEY}&language=en&pageSize={limit}"
            response = requests.get(url)
            
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
                return self._get_fallback_news_data(query)
                
        except Exception as e:
            print(f"Error fetching news articles: {e}")
            return self._get_fallback_news_data(query)
    
    def fetch_reddit_posts(self, query, days_back=7, limit=10):
        """
        Fetch Reddit posts related to a query.
        
        Args:
            query (str): Search query
            days_back (int): How many days back to search
            limit (int): Maximum number of posts to return
            
        Returns:
            list: List of dictionaries with post data
        """
        if not self.reddit_api_available:
            return self._get_fallback_reddit_data(query)
            
        try:
            # Get Reddit OAuth token
            auth = requests.auth.HTTPBasicAuth(REDDIT_CLIENT_ID, REDDIT_SECRET)
            data = {
                'grant_type': 'client_credentials',
                'username': 'sentimizer_app',  # Replace with actual username if available
                'password': 'sentimizer_password'  # Replace with actual password if available
            }
            headers = {'User-Agent': 'Sentimizer/0.1'}
            
            token_response = requests.post(
                'https://www.reddit.com/api/v1/access_token',
                auth=auth,
                data=data,
                headers=headers
            )
            
            if token_response.status_code != 200:
                return self._get_fallback_reddit_data(query)
                
            token = token_response.json().get('access_token', '')
            
            if not token:
                return self._get_fallback_reddit_data(query)
            
            # Use the token to fetch posts
            headers['Authorization'] = f'Bearer {token}'
            search_url = f'https://oauth.reddit.com/search?q={query}&sort=relevance&t=week&limit={limit}'
            
            search_response = requests.get(search_url, headers=headers)
            
            if search_response.status_code == 200:
                data = search_response.json()
                posts = data.get('data', {}).get('children', [])
                
                processed_posts = []
                for post in posts:
                    post_data = post.get('data', {})
                    processed_posts.append({
                        'title': post_data.get('title', ''),
                        'subreddit': post_data.get('subreddit', ''),
                        'url': f"https://www.reddit.com{post_data.get('permalink', '')}",
                        'created_at': datetime.fromtimestamp(post_data.get('created_utc', 0)).isoformat(),
                        'score': post_data.get('score', 0),
                        'comments': post_data.get('num_comments', 0),
                        'content': post_data.get('selftext', '')
                    })
                
                return processed_posts
            else:
                print(f"Error fetching Reddit posts: {search_response.status_code}")
                return self._get_fallback_reddit_data(query)
                
        except Exception as e:
            print(f"Error fetching Reddit posts: {e}")
            return self._get_fallback_reddit_data(query)
    
    def fetch_twitter_posts(self, query, days_back=7, limit=10):
        """
        Fetch Twitter/X posts related to a query.
        
        Args:
            query (str): Search query
            days_back (int): How many days back to search
            limit (int): Maximum number of posts to return
            
        Returns:
            list: List of dictionaries with post data
        """
        if not self.twitter_api_available:
            return self._get_fallback_twitter_data(query)
            
        try:
            # Twitter API v2 endpoint
            url = f"https://api.twitter.com/2/tweets/search/recent?query={query}&max_results={limit}&tweet.fields=created_at,public_metrics"
            
            headers = {
                'Authorization': f'Bearer {TWITTER_API_KEY}',
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
                return self._get_fallback_twitter_data(query)
                
        except Exception as e:
            print(f"Error fetching Twitter posts: {e}")
            return self._get_fallback_twitter_data(query)
    
    def web_scrape_for_topic(self, query, num_results=5):
        """
        Scrape web content related to a topic using a web search API or direct scraping.
        
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
                result_links = soup.select('.yuRUbf a')
                urls = [link.get('href') for link in result_links if link.get('href').startswith('http')]
                
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
    
    def fetch_historical_data(self, topic, start_date=None, end_date=None):
        """
        Fetch historical data about a topic for temporal analysis.
        This would typically use a paid API, but for now we'll create realistic-looking synthetic data.
        
        Args:
            topic (str): The topic to analyze
            start_date (datetime): Start date for analysis
            end_date (datetime): End date for analysis
            
        Returns:
            pandas.DataFrame: DataFrame with dates and interest scores
        """
        if not start_date:
            start_date = datetime.now() - timedelta(days=365)  # Default to 1 year
        if not end_date:
            end_date = datetime.now()
            
        # Generate date range
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Create a base trend with some seasonality and a general trend
        days = len(date_range)
        
        # Base interest with trend
        np.random.seed(42)  # For reproducible results
        
        # Create a more realistic trend with:
        # 1. General trend direction (up, down, or stable)
        trend_direction = np.random.choice([-0.0001, 0, 0.0001])
        base_trend = np.linspace(0, trend_direction * days, days)
        
        # 2. Add weekly seasonality (e.g., more interest on weekdays)
        weekly = 0.1 * np.sin(np.arange(days) * (2 * np.pi / 7))
        
        # 3. Add monthly seasonality
        monthly = 0.2 * np.sin(np.arange(days) * (2 * np.pi / 30))
        
        # 4. Add a few "spike" events
        spikes = np.zeros(days)
        num_spikes = int(days / 60)  # Roughly one spike every two months
        spike_indices = np.random.choice(np.arange(days), size=num_spikes, replace=False)
        spike_heights = np.random.uniform(0.5, 1.5, size=num_spikes)
        for idx, height in zip(spike_indices, spike_heights):
            # Create a small window of impact around the spike
            window = 5
            for j in range(-window, window + 1):
                if 0 <= idx + j < days:
                    # Diminishing effect as we move away from spike center
                    spikes[idx + j] = height * (1 - abs(j) / window)
        
        # 5. Add some random noise
        noise = 0.1 * np.random.normal(0, 1, days)
        
        # Combine all components and normalize to 0-100 range
        interest = base_trend + weekly + monthly + spikes + noise
        interest = 50 + 40 * (interest - np.min(interest)) / (np.max(interest) - np.min(interest))
        
        # Create the DataFrame
        df = pd.DataFrame({
            'date': date_range,
            'interest': interest
        })
        
        # Add a smoothed version for trend visualization
        df['interest_smoothed'] = df['interest'].rolling(window=7, center=True).mean().fillna(df['interest'])
        
        return df
    
    def fetch_global_interest_data(self, topic, subtopics=[]):
        """
        Fetch global interest data about topics for geographical analysis.
        For now, we'll create synthetic data but in production this would use Google Trends API
        or a similar service.
        
        Args:
            topic (str): The main topic
            subtopics (list): Additional subtopics to include
            
        Returns:
            dict: Dictionary with country data
        """
        # Get all countries
        all_countries = []
        for country in pycountry.countries:
            try:
                all_countries.append({
                    'name': country.name,
                    'alpha_2': country.alpha_2,
                    'alpha_3': country.alpha_3
                })
            except AttributeError:
                # Some countries might not have all attributes
                pass
        
        # Create interest values for every country
        np.random.seed(hash(topic) % 10000)  # Different seed for each topic
        
        # Main topic interest by country
        topic_data = []
        
        for country in all_countries:
            # Generate interest with some regional clustering effect
            # We'll use the 3-letter country code's ord values to create regional similarities
            country_seed = sum(ord(c) for c in country['alpha_3'])
            regional_factor = np.sin(country_seed * 0.1) * 0.5 + 0.5  # 0 to 1 value
            
            interest = np.random.normal(50 * regional_factor, 20)
            interest = max(0, min(100, interest))  # Clamp to 0-100
            
            topic_data.append({
                'country': country['name'],
                'country_code': country['alpha_3'],
                'interest': interest
            })
        
        # Generate subtopic data
        subtopic_data = {}
        for subtopic in subtopics:
            subtopic_data[subtopic] = []
            
            # Use a different seed for each subtopic
            np.random.seed(hash(subtopic) % 10000)
            
            for country in all_countries:
                country_seed = sum(ord(c) for c in country['alpha_3'])
                regional_factor = np.sin(country_seed * 0.1 + hash(subtopic) * 0.01) * 0.5 + 0.5
                
                interest = np.random.normal(50 * regional_factor, 20)
                interest = max(0, min(100, interest))  # Clamp to 0-100
                
                subtopic_data[subtopic].append({
                    'country': country['name'],
                    'country_code': country['alpha_3'],
                    'interest': interest
                })
        
        # Combine everything
        result = {
            'main_topic': topic,
            'main_topic_data': topic_data,
            'subtopics': subtopics,
            'subtopic_data': subtopic_data
        }
        
        return result
    
    def fetch_keyword_data(self, topic, subtopics=[]):
        """
        Fetch keyword and discussion data for a topic and its subtopics.
        
        Args:
            topic (str): The main topic
            subtopics (list): List of subtopics
            
        Returns:
            dict: Dictionary with keyword and discussion data
        """
        # In a real implementation, this would use social monitoring APIs
        # For now, we'll return structured data that simulates keyword extraction
        
        # Create a list of potential keywords related to the topic
        potential_keywords = [
            "innovation", "growth", "strategy", "challenge", "solution",
            "market", "global", "sustainable", "technology", "digital",
            "transformation", "opportunity", "crisis", "development", "impact",
            "future", "leader", "change", "disruption", "trend"
        ]
        
        np.random.seed(hash(topic) % 10000)
        
        # Select a subset of keywords for main topic
        num_keywords = min(8, len(potential_keywords))
        main_keywords = np.random.choice(potential_keywords, num_keywords, replace=False).tolist()
        
        # Generate frequency for each keyword
        main_keyword_data = []
        for keyword in main_keywords:
            frequency = np.random.randint(30, 100)
            main_keyword_data.append({
                'keyword': keyword,
                'frequency': frequency,
                'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 
                                             p=[0.4, 0.4, 0.2])  # More weighted toward positive/neutral
            })
        
        # Sort by frequency
        main_keyword_data = sorted(main_keyword_data, key=lambda x: x['frequency'], reverse=True)
        
        # Generate subtopic keywords
        subtopic_keywords = {}
        for subtopic in subtopics:
            np.random.seed(hash(subtopic) % 10000)
            
            # Select keywords for this subtopic
            num_keywords = min(5, len(potential_keywords))
            selected_keywords = np.random.choice(potential_keywords, num_keywords, replace=False).tolist()
            
            subtopic_keyword_data = []
            for keyword in selected_keywords:
                frequency = np.random.randint(20, 90)
                subtopic_keyword_data.append({
                    'keyword': keyword,
                    'frequency': frequency,
                    'sentiment': np.random.choice(['positive', 'neutral', 'negative'], 
                                                 p=[0.3, 0.4, 0.3])
                })
            
            # Sort by frequency
            subtopic_keyword_data = sorted(subtopic_keyword_data, key=lambda x: x['frequency'], reverse=True)
            subtopic_keywords[subtopic] = subtopic_keyword_data
        
        return {
            'main_topic': topic,
            'main_topic_keywords': main_keyword_data,
            'subtopics': subtopics,
            'subtopic_keywords': subtopic_keywords
        }
    
    # Fallback methods for when API keys are not available
    def _get_fallback_news_data(self, query):
        """Generate fallback news data when API is not available"""
        current_date = datetime.now()
        sources = ["The Daily News", "Global Times", "Tech Insider", "Financial Review", 
                   "Science Today", "World Report", "Business Daily", "Market Watch"]
                   
        articles = []
        for i in range(5):  # Return 5 simulated articles
            date = current_date - timedelta(days=i)
            articles.append({
                'title': f"{query.title()} - Latest Developments and Analysis",
                'source': sources[i % len(sources)],
                'url': f"https://example.com/news/{query.lower().replace(' ', '-')}-article-{i}",
                'published_at': date.isoformat(),
                'content': f"This simulated article discusses {query} in detail..."
            })
        
        return articles
    
    def _get_fallback_reddit_data(self, query):
        """Generate fallback Reddit data when API is not available"""
        current_date = datetime.now()
        subreddits = ["worldnews", "technology", "business", "science", "politics", 
                      "economics", "futurology", "environment"]
                      
        posts = []
        for i in range(5):  # Return 5 simulated posts
            date = current_date - timedelta(days=i)
            posts.append({
                'title': f"Discussion about {query.title()} and its implications",
                'subreddit': subreddits[i % len(subreddits)],
                'url': f"https://www.reddit.com/r/{subreddits[i % len(subreddits)]}/comments/{i}",
                'created_at': date.isoformat(),
                'score': random.randint(10, 5000),
                'comments': random.randint(5, 500),
                'content': f"This simulated Reddit post discusses {query}..."
            })
        
        return posts
    
    def _get_fallback_twitter_data(self, query):
        """Generate fallback Twitter data when API is not available"""
        current_date = datetime.now()
        tweets = []
        
        for i in range(5):  # Return 5 simulated tweets
            date = current_date - timedelta(days=i, hours=random.randint(0, 23))
            tweets.append({
                'id': f"{10000000 + i}",
                'text': f"Interesting developments in {query}. #analysis #trends",
                'created_at': date.isoformat(),
                'retweets': random.randint(5, 2000),
                'likes': random.randint(10, 5000),
                'url': f"https://twitter.com/i/web/status/{10000000 + i}"
            })
        
        return tweets

# Helper function to get sentiment from all sources
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
    if subtopics is None:
        subtopics = []
        
    data_fetcher = ExternalDataFetcher()
    
    # Check API availability
    api_status = data_fetcher.check_api_availability()
    
    # Fetch data from all available sources
    news_data = data_fetcher.fetch_news_articles(topic, days_back=days_back)
    reddit_data = data_fetcher.fetch_reddit_posts(topic, days_back=days_back)
    twitter_data = data_fetcher.fetch_twitter_posts(topic, days_back=days_back)
    web_data = data_fetcher.web_scrape_for_topic(topic, num_results=5)
    
    # Fetch historical trend data
    historical_data = data_fetcher.fetch_historical_data(
        topic, 
        start_date=datetime.now() - timedelta(days=days_back),
        end_date=datetime.now()
    )
    
    # Fetch global interest data
    global_data = data_fetcher.fetch_global_interest_data(topic, subtopics)
    
    # Fetch keyword data
    keyword_data = data_fetcher.fetch_keyword_data(topic, subtopics)
    
    # Format the data as a uniform structure
    result = {
        'topic': topic,
        'subtopics': subtopics,
        'sources': {
            'news': news_data,
            'reddit': reddit_data,
            'twitter': twitter_data,
            'web': web_data
        },
        'historical_data': historical_data.to_dict(orient='records'),
        'global_data': global_data,
        'keyword_data': keyword_data,
        'api_status': api_status
    }
    
    return result