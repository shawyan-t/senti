"""
Module for generating sentiment data when external API data is unavailable.
Uses OpenAI to generate realistic sentiment scores based on search topics.
"""
import os
import json
import random
import pycountry
from datetime import datetime
from openai import OpenAI
from transformers import pipeline
import numpy as np
from sentence_transformers import SentenceTransformer
from umap import UMAP

# Initialize OpenAI client with API key
openai_api_key = os.environ.get("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Important note: the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user

# Load the HuggingFace emotion pipeline once
_emotion_classifier = None

def get_emotion_classifier():
    global _emotion_classifier
    if _emotion_classifier is None:
        _emotion_classifier = pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base", top_k=None)
    return _emotion_classifier

# Map model emotions to Plutchik's 8 emotions
PLUTCHIK_MAP = {
    'joy': 'Joy',
    'trust': 'Trust',
    'fear': 'Fear',
    'surprise': 'Surprise',
    'sadness': 'Sadness',
    'disgust': 'Disgust',
    'anger': 'Anger',
    'anticipation': 'Anticipation',
}
# The model may not output all 8, so we need to map or fill in zeros
PLUTCHIK_EMOTIONS = ['Joy', 'Trust', 'Fear', 'Surprise', 'Sadness', 'Disgust', 'Anger', 'Anticipation']

def generate_country_sentiment(topic, countries=None):
    """
    Generate realistic sentiment for countries based on the topic.
    Uses OpenAI to assess likely sentiment toward a topic in different regions.
    
    Args:
        topic (str): The topic to analyze
        countries (list): List of countries to analyze, or None to use default major countries
        
    Returns:
        list: List of dictionaries with country data and sentiment
    """
    if not countries:
        # Default set of major countries
        countries = [
            "United States", "China", "India", "Brazil", "Russia", 
            "United Kingdom", "Germany", "France", "Japan", "Australia",
            "Canada", "Mexico", "South Africa", "Nigeria", "Saudi Arabia",
            "Italy", "Spain", "South Korea"
        ]
    
    # Prepare coordinates for each country
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
        'CA': (56.1304, -106.3468), # Canada
        'MX': (23.6345, -102.5528), # Mexico
        'NG': (9.0820, 8.6753),     # Nigeria
        'SA': (23.8859, 45.0792),   # Saudi Arabia
        'IT': (41.8719, 12.5674),   # Italy
        'ES': (40.4637, -3.7492),   # Spain
        'KR': (35.9078, 127.7669),  # South Korea
    }
    
    # Use OpenAI to generate realistic sentiment for each country
    try:
        # Build prompt for OpenAI
        prompt = f"""
        For the topic "{topic}", generate a realistic sentiment analysis for the following countries, 
        based on cultural, social, economic and political factors. Consider:

        1. How is this topic typically viewed in each country?
        2. Are there strong cultural or political positions on this topic in these regions?
        3. What recent events might influence sentiment toward this topic?

        For each country, provide a sentiment label (positive, neutral, or negative) and an interest score (0-100) 
        indicating how much this topic is discussed or searched for in that country.

        Countries: {", ".join(countries)}

        Return your response in this JSON format:
        {{
            "countries": [
                {{
                    "name": "Country Name",
                    "sentiment": "positive/neutral/negative",
                    "interest": interest_score (0-100),
                    "rationale": "Brief explanation of sentiment and interest level"
                }},
                ...
            ]
        }}

        Base your analysis on real cultural and geopolitical knowledge. Ensure the sentiments and interest levels vary realistically based on actual regional differences.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert geopolitical and cultural sentiment analyzer."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse response
        sentiment_data = json.loads(response.choices[0].message.content)
        
        # Process and structure the results
        country_data = []
        
        for country_info in sentiment_data.get("countries", []):
            country_name = country_info.get("name", "")
            # Find country code
            try:
                # Try exact match first
                country_obj = pycountry.countries.get(name=country_name)
                
                # If not found, try fuzzy search
                if not country_obj:
                    countries_found = pycountry.countries.search_fuzzy(country_name)
                    if countries_found:
                        country_obj = countries_found[0]
                
                if country_obj:
                    alpha_2 = country_obj.alpha_2
                    alpha_3 = country_obj.alpha_3
                    
                    # Get coordinates
                    if alpha_2 in country_coords:
                        lat, lon = country_coords[alpha_2]
                    else:
                        # Use random but plausible coordinates if not in our mapping
                        random.seed(hash(alpha_2))
                        lat = random.uniform(-60, 70)
                        lon = random.uniform(-180, 180)
                    
                    # Add to country data
                    country_data.append({
                        'name': country_name,
                        'alpha_2': alpha_2,
                        'alpha_3': alpha_3,
                        'latitude': lat,
                        'longitude': lon,
                        'sentiment': country_info.get("sentiment", "neutral"),
                        'interest': country_info.get("interest", 50),
                        'rationale': country_info.get("rationale", "")
                    })
            except Exception as e:
                print(f"Error processing country {country_name}: {e}")
                continue
        
        return country_data
    
    except Exception as e:
        print(f"Error generating country sentiment: {e}")
        return []

def generate_topic_keywords(topic):
    """
    Generate realistic keywords related to the topic with sentiment scores.
    Uses OpenAI to generate keywords that would likely be associated with a topic.
    
    Args:
        topic (str): The topic to analyze
        
    Returns:
        dict: Dictionary with keyword data
    """
    try:
        # Build prompt for OpenAI
        prompt = f"""
        For the topic "{topic}", generate a list of 10-15 related keywords or search terms that people would likely use.
        For each keyword, provide:
        1. A frequency score (1-100) indicating how commonly this term is associated with the topic
        2. A sentiment classification (positive, neutral, or negative)

        Return your response in this JSON format:
        {{
            "main_topic": "{topic}",
            "main_topic_keywords": [
                {{
                    "keyword": "term1",
                    "frequency": frequency_score,
                    "sentiment": "positive/neutral/negative"
                }},
                ...
            ],
            "subtopics": ["subtopic1", "subtopic2", "subtopic3"],
            "subtopic_keywords": {{
                "subtopic1": [
                    {{
                        "keyword": "term1",
                        "frequency": frequency_score,
                        "sentiment": "positive/neutral/negative"
                    }},
                    ...
                ],
                ...
            }}
        }}

        Ensure the frequency scores and sentiments are realistic based on actual search patterns and cultural attitudes.
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in search trends and keyword analysis."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        
        # Parse response
        keyword_data = json.loads(response.choices[0].message.content)
        return keyword_data
    
    except Exception as e:
        print(f"Error generating topic keywords: {e}")
        return {
            "main_topic": topic,
            "main_topic_keywords": [],
            "subtopics": [],
            "subtopic_keywords": {}
        }

def generate_time_series_data(topic, days=180):
    """
    Generate realistic time series data for interest in a topic over time.
    Uses OpenAI to create a plausible pattern of interest over time.
    
    Args:
        topic (str): The topic to analyze
        days (int): Number of days of data to generate
        
    Returns:
        list: List of dictionaries with date and interest value
    """
    try:
        # Build prompt for OpenAI
        prompt = f"""
        For the topic "{topic}", analyze how interest in this topic has likely changed over the past {days} days.
        Consider:
        1. Seasonal patterns if relevant
        2. Recent events that might have caused spikes or drops in interest
        3. General trend (increasing, decreasing, stable)

        Describe the pattern of interest, noting any significant points (e.g., "Sharp spike in March due to...").
        
        Then, list 5-7 key dates and the approximate interest level on those dates (0-100).
        """
        
        # Call OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are an expert in search trends and topic analysis."},
                {"role": "user", "content": prompt}
            ]
        )
        
        # Use the description to build a realistic time series
        description = response.choices[0].message.content
        
        # Generate timestamps
        import pandas as pd
        import numpy as np
        from datetime import timedelta
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Basic trend with some randomness
        base_interest = np.random.normal(50, 10, size=len(date_range))
        
        # Add trend based on description
        if "increasing" in description.lower():
            trend = np.linspace(0, 20, len(date_range))
            base_interest += trend
        elif "decreasing" in description.lower():
            trend = np.linspace(20, 0, len(date_range))
            base_interest += trend
            
        # Look for specific events in the description
        import re
        events = re.findall(r"(spike|drop|increase|decrease|peak|rise|fall).*?(January|February|March|April|May|June|July|August|September|October|November|December)", description)
        
        for event in events:
            event_type, month = event
            # Find approximate location in the date range for this month
            month_idx = [i for i, date in enumerate(date_range) if date.strftime('%B') == month]
            if month_idx:
                # Add spike or drop
                if event_type in ['spike', 'increase', 'peak', 'rise']:
                    base_interest[month_idx[len(month_idx)//2]:] += np.random.uniform(10, 30)
                else:
                    base_interest[month_idx[len(month_idx)//2]:] -= np.random.uniform(10, 30)
        
        # Ensure values are within 0-100 range
        base_interest = np.clip(base_interest, 0, 100)
        
        # Convert to the expected format
        result = []
        for i, date in enumerate(date_range):
            result.append({
                'date': date,
                'interest': float(base_interest[i]),
                'query': topic
            })
        
        # Create smoothed version (7-day rolling average)
        df = pd.DataFrame(result)
        df['interest_smoothed'] = df['interest'].rolling(window=7, center=True).mean().fillna(df['interest'])
        
        # Convert back to dictionaries
        return df.to_dict('records')
    
    except Exception as e:
        print(f"Error generating time series data: {e}")
        return []

def generate_emotion_analysis(text):
    """
    Generate an 8-dimensional emotion analysis for a given text using HuggingFace.
    Returns a list of dicts: [{emotion: 'Joy', score: ...}, ...] with scores summing to 1.0
    """
    classifier = get_emotion_classifier()
    # Run the classifier
    results = classifier(text)
    # results: list of dicts with 'label' and 'score'
    # Map to Plutchik
    emotion_scores = {e: 0.0 for e in PLUTCHIK_EMOTIONS}
    for r in results[0]:
        label = r['label'].lower()
        # Map model label to Plutchik if possible
        if label in PLUTCHIK_MAP:
            plutchik = PLUTCHIK_MAP[label]
            emotion_scores[plutchik] = r['score']
        # Optionally, map similar emotions (e.g., 'optimism' to 'anticipation')
        elif label == 'optimism':
            emotion_scores['Anticipation'] = r['score']
        elif label == 'love':
            emotion_scores['Trust'] += r['score'] * 0.5
            emotion_scores['Joy'] += r['score'] * 0.5
        # Add more mappings as needed
    # Normalize
    total = sum(emotion_scores.values())
    if total > 0:
        for k in emotion_scores:
            emotion_scores[k] /= total
    # Return as list of dicts
    return [{"emotion": k, "score": emotion_scores[k]} for k in PLUTCHIK_EMOTIONS]

_embedding_model = None

def get_embedding_model():
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embedding_model

def generate_embeddings(texts):
    """
    Generate embeddings for a list of texts or a single string using sentence-transformers,
    and process them through UMAP to get 3D coordinates.
    Returns a list of 3D embedding vectors (list of floats).
    """
    model = get_embedding_model()
    if isinstance(texts, str):
        texts = [texts]
    # Generate high-dimensional embeddings
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    
    # Process through UMAP to get 3D coordinates
    umap = UMAP(n_components=3, n_neighbors=15, min_dist=0.1, random_state=42)
    embeddings_3d = umap.fit_transform(embeddings)
    
    return embeddings_3d.tolist()