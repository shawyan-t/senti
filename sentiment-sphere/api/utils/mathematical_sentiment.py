"""
Mathematical Sentiment Analysis Engine
Provides rigorous, statistically-backed sentiment analysis using multiple models.
"""

import numpy as np
import scipy.stats as stats
from scipy.spatial.distance import cosine
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime, timedelta
import json
import logging

# Lexicon-based analyzers
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from afinn import Afinn

# Transformer-based analyzers
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# Performance optimization
from numba import jit
import warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class SentimentScore:
    """Statistical sentiment score with confidence metrics"""
    value: float
    confidence_interval: Tuple[float, float]
    statistical_significance: float
    method: str
    sample_size: int = 1

@dataclass
class EmotionVector:
    """Mathematical representation of emotional state"""
    coordinates: np.ndarray  # 8D Plutchik vector
    magnitude: float
    dominant_emotions: List[str]
    entropy: float
    coherence: float

@dataclass
class TemporalAnalysis:
    """Time-based sentiment analysis"""
    trend_slope: float
    trend_significance: float
    momentum: float
    acceleration: float
    baseline_comparison: float

class MathematicalSentimentAnalyzer:
    """
    Advanced sentiment analyzer using multiple mathematical models
    for statistical rigor and performance optimization.
    """
    
    def __init__(self):
        """Initialize all sentiment analysis models"""
        logger.info("Initializing Mathematical Sentiment Analyzer...")
        
        # Lexicon-based analyzers
        self.vader = SentimentIntensityAnalyzer()
        self.afinn = Afinn()
        
        # Transformer models (cached for performance)
        self._init_transformer_models()
        
        # Plutchik emotion mapping
        self.plutchik_emotions = [
            'joy', 'trust', 'fear', 'surprise', 
            'sadness', 'disgust', 'anger', 'anticipation'
        ]
        
        # Statistical parameters
        self.confidence_level = 0.95
        self.bootstrap_samples = 1000
        
        logger.info("Mathematical Sentiment Analyzer initialized successfully")

    def _init_transformer_models(self):
        """Initialize transformer-based sentiment models"""
        try:
            # Use smaller, faster models for real-time analysis
            self.sentiment_pipeline = pipeline(
                "sentiment-analysis",
                model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                tokenizer="cardiffnlp/twitter-roberta-base-sentiment-latest",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.emotion_pipeline = pipeline(
                "text-classification",
                model="j-hartmann/emotion-english-distilroberta-base",
                device=0 if torch.cuda.is_available() else -1,
                top_k=None
            )
            
        except Exception as e:
            logger.warning(f"Could not load transformer models: {e}")
            self.sentiment_pipeline = None
            self.emotion_pipeline = None

    def _calculate_confidence_interval(self, scores: np.ndarray, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate confidence interval using bootstrap sampling"""
        if len(scores) < 2:
            return (scores[0], scores[0])
            
        # Bootstrap sampling for confidence interval
        bootstrap_means = []
        n_samples = len(scores)
        
        for _ in range(self.bootstrap_samples):
            bootstrap_sample = np.random.choice(scores, size=n_samples, replace=True)
            bootstrap_means.append(np.mean(bootstrap_sample))
        
        alpha = 1 - confidence
        lower_percentile = (alpha/2) * 100
        upper_percentile = (1 - alpha/2) * 100
        
        ci_lower = np.percentile(bootstrap_means, lower_percentile)
        ci_upper = np.percentile(bootstrap_means, upper_percentile)
        
        return (ci_lower, ci_upper)

    def _lexicon_analysis(self, text: str) -> Dict[str, float]:
        """Perform lexicon-based sentiment analysis with financial context enhancement"""
        results = {}
        
        # VADER Analysis
        vader_scores = self.vader.polarity_scores(text)
        results['vader'] = vader_scores['compound']
        
        # TextBlob Analysis  
        blob = TextBlob(text)
        results['textblob'] = blob.sentiment.polarity
        
        # AFINN Analysis - fix the input type
        try:
            afinn_score = self.afinn.score(text)  # AFINN expects string, not list
            # Normalize AFINN score to [-1, 1] range
            word_count = max(len(text.split()), 1)
            normalized_afinn = afinn_score / word_count if word_count > 0 else 0
            results['afinn'] = max(-1, min(1, normalized_afinn))  # Clamp to [-1, 1]
        except Exception as e:
            logger.warning(f"AFINN analysis failed: {e}")
            results['afinn'] = 0.0
        
        # Financial sentiment analysis - interpret financial language
        financial_sentiment = self._analyze_financial_sentiment(text)
        results['financial_context'] = financial_sentiment
        
        return results

    def _analyze_financial_sentiment(self, text: str) -> float:
        """Analyze financial sentiment using context-aware keyword patterns"""
        text_lower = text.lower()
        financial_score = 0.0
        
        # Positive financial indicators (bullish)
        positive_patterns = {
            'growth': ['increased', 'boosted', 'gained', 'surged', 'jumped', 'soared', 'climbed', 'rose', 'rally'],
            'performance': ['outperformed', 'beat expectations', 'exceeded', 'strong', 'robust', 'solid'],
            'financial_health': ['profitable', 'revenue growth', 'earnings beat', 'upgraded', 'buy rating'],
            'market_action': ['acquisition', 'merger', 'expansion', 'investment', 'partnership'],
            'percentages': []  # Will be calculated dynamically
        }
        
        # Negative financial indicators (bearish)  
        negative_patterns = {
            'decline': ['decreased', 'fell', 'dropped', 'plunged', 'tumbled', 'declined', 'slumped', 'cut'],
            'performance': ['underperformed', 'missed expectations', 'weak', 'disappointing', 'poor'],
            'financial_trouble': ['loss', 'debt', 'bankruptcy', 'layoffs', 'downgraded', 'sell rating'],
            'market_concern': ['investigation', 'lawsuit', 'scandal', 'regulatory', 'fine'],
            'percentages': []  # Will be calculated dynamically
        }
        
        # Count positive and negative indicators
        positive_count = 0
        negative_count = 0
        
        for category, patterns in positive_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    positive_count += 1
        
        for category, patterns in negative_patterns.items():
            for pattern in patterns:
                if pattern in text_lower:
                    negative_count += 1
        
        # Look for percentage changes (very important for financial sentiment)
        import re
        percentage_matches = re.findall(r'(\w+).*?(\d+(?:\.\d+)?)\s*%', text_lower)
        for action, percentage in percentage_matches:
            pct = float(percentage)
            if any(pos_word in action for pos_word in ['up', 'gain', 'boost', 'increas', 'rise']):
                # Positive percentage change
                positive_count += min(3, pct / 10)  # Scale: 10% = 1 point, 30%+ = 3 points
            elif any(neg_word in action for neg_word in ['down', 'loss', 'declin', 'drop', 'fall']):
                # Negative percentage change  
                negative_count += min(3, pct / 10)
        
        # Calculate financial sentiment score
        if positive_count > 0 or negative_count > 0:
            net_sentiment = positive_count - negative_count
            # Normalize to [-1, 1] with more sensitivity for financial content
            max_possible = max(positive_count + negative_count, 1)
            financial_score = max(-1, min(1, net_sentiment / max_possible))
            
            # Amplify financial sentiment (financial news should have stronger signals)
            financial_score *= 1.5  # Make financial sentiment more pronounced
            financial_score = max(-1, min(1, financial_score))  # Keep in bounds
        
        return financial_score

    def _transformer_analysis(self, text: str) -> Dict[str, float]:
        """Perform transformer-based sentiment analysis"""
        results = {}
        
        if self.sentiment_pipeline is not None:
            try:
                # Sentiment analysis
                sentiment_result = self.sentiment_pipeline(text)
                
                # Convert to standardized scale [-1, 1]
                if sentiment_result[0]['label'].upper() == 'POSITIVE':
                    results['roberta_sentiment'] = sentiment_result[0]['score']
                elif sentiment_result[0]['label'].upper() == 'NEGATIVE':
                    results['roberta_sentiment'] = -sentiment_result[0]['score']
                else:  # NEUTRAL
                    results['roberta_sentiment'] = 0.0
                    
            except Exception as e:
                logger.warning(f"Transformer sentiment analysis failed: {e}")
                results['roberta_sentiment'] = 0.0
        
        return results

    def _emotion_vector_analysis(self, text: str) -> EmotionVector:
        """Generate mathematical emotion vector using Plutchik's model"""
        emotion_scores = np.zeros(8)  # 8 Plutchik emotions
        
        if self.emotion_pipeline is not None:
            try:
                emotion_results = self.emotion_pipeline(text)
                
                # Map detected emotions to Plutchik model
                emotion_map = {
                    'joy': 0, 'trust': 1, 'fear': 2, 'surprise': 3,
                    'sadness': 4, 'disgust': 5, 'anger': 6, 'anticipation': 7
                }
                
                for emotion_result in emotion_results:
                    emotion_name = emotion_result['label'].lower()
                    if emotion_name in emotion_map:
                        emotion_scores[emotion_map[emotion_name]] = emotion_result['score']
                
            except Exception as e:
                logger.warning(f"Emotion analysis failed: {e}")
        
        # Normalize emotion vector
        if np.sum(emotion_scores) > 0:
            emotion_scores = emotion_scores / np.sum(emotion_scores)
        else:
            # Default neutral emotional state
            emotion_scores = np.ones(8) / 8
        
        # Calculate mathematical properties
        magnitude = np.linalg.norm(emotion_scores)
        
        # Calculate emotion entropy (uncertainty)
        entropy = -np.sum(emotion_scores * np.log2(emotion_scores + 1e-10))
        
        # Calculate emotional coherence (how focused the emotions are)
        coherence = 1.0 - (entropy / np.log2(8))  # Normalized entropy
        
        # Get dominant emotions (top 3)
        dominant_indices = np.argsort(emotion_scores)[-3:][::-1]
        dominant_emotions = [self.plutchik_emotions[i] for i in dominant_indices if emotion_scores[i] > 0.1]
        
        return EmotionVector(
            coordinates=emotion_scores,
            magnitude=magnitude,
            dominant_emotions=dominant_emotions,
            entropy=entropy,
            coherence=coherence
        )

    def _calculate_sentiment_entropy(self, scores: Dict[str, float]) -> float:
        """Calculate sentiment entropy - measure of uncertainty"""
        # Convert scores to probabilities
        score_values = np.array(list(scores.values()))
        
        # Shift to positive range and normalize
        shifted_scores = score_values + 1  # [-1,1] -> [0,2]
        probabilities = shifted_scores / np.sum(shifted_scores)
        
        # Calculate entropy
        entropy = -np.sum(probabilities * np.log2(probabilities + 1e-10))
        return entropy

    def _calculate_polarization_index(self, scores: Dict[str, float]) -> float:
        """Calculate how polarized/divisive the sentiment is"""
        score_values = np.array(list(scores.values()))
        
        # Calculate standard deviation (measure of disagreement)
        polarization = np.std(score_values)
        
        # Normalize to [0, 1] range
        return min(1.0, polarization)

    def analyze_mathematical_sentiment(self, text: str, historical_data: Optional[List[Dict]] = None) -> Dict[str, Any]:
        """
        Perform comprehensive mathematical sentiment analysis
        
        Args:
            text: Text to analyze
            historical_data: Historical sentiment data for temporal analysis
            
        Returns:
            Comprehensive mathematical analysis results
        """
        logger.info("Starting mathematical sentiment analysis...")
        
        # Step 1: Multi-model sentiment analysis
        lexicon_scores = self._lexicon_analysis(text)
        transformer_scores = self._transformer_analysis(text)
        
        # Combine all scores
        all_scores = {**lexicon_scores, **transformer_scores}
        score_array = np.array(list(all_scores.values()))
        
        # Step 2: Calculate consensus score with statistical confidence
        consensus_score = np.mean(score_array)
        confidence_interval = self._calculate_confidence_interval(score_array)
        
        # Statistical significance (how confident we are this isn't random)
        t_stat, p_value = stats.ttest_1samp(score_array, 0)  # Test against neutral
        statistical_significance = 1 - p_value if p_value < 1 else 0
        
        # Step 3: Calculate entropy and polarization
        sentiment_entropy = self._calculate_sentiment_entropy(all_scores)
        polarization_index = self._calculate_polarization_index(all_scores)
        
        # Step 4: Emotion vector analysis
        emotion_vector = self._emotion_vector_analysis(text)
        
        # Step 5: Temporal analysis (if historical data available)
        temporal_analysis = None
        if historical_data and len(historical_data) > 1:
            temporal_analysis = self._analyze_temporal_trends(historical_data, consensus_score)
        
        # Step 6: Compile comprehensive results
        results = {
            "mathematical_sentiment_analysis": {
                "composite_score": {
                    "value": float(consensus_score),
                    "confidence_interval": [float(ci) for ci in confidence_interval],
                    "statistical_significance": float(statistical_significance)
                },
                
                "multi_model_validation": {
                    "lexicon_based": {
                        "vader_score": float(lexicon_scores.get('vader', 0)),
                        "textblob_score": float(lexicon_scores.get('textblob', 0)),
                        "afinn_score": float(lexicon_scores.get('afinn', 0)),
                        "consensus": float(np.mean([lexicon_scores.get(k, 0) for k in ['vader', 'textblob', 'afinn']]))
                    },
                    "transformer_based": {
                        "roberta_sentiment": float(transformer_scores.get('roberta_sentiment', 0)),
                        "consensus": float(np.mean([v for v in transformer_scores.values()]))
                    },
                    "all_models_consensus": float(consensus_score)
                },
                
                "uncertainty_metrics": {
                    "sentiment_entropy": float(sentiment_entropy),
                    "polarization_index": float(polarization_index),
                    "model_agreement": float(1.0 - np.std(score_array))  # Higher = more agreement
                }
            },
            
            "emotion_vector_analysis": {
                "plutchik_coordinates": {
                    emotion: float(score) for emotion, score in 
                    zip(self.plutchik_emotions, emotion_vector.coordinates)
                },
                "dominant_emotions": emotion_vector.dominant_emotions,
                "emotion_entropy": float(emotion_vector.entropy),
                "emotional_coherence": float(emotion_vector.coherence),
                
                "emotion_mathematics": {
                    "vector_magnitude": float(emotion_vector.magnitude),
                    "emotional_distance_from_neutral": float(
                        np.linalg.norm(emotion_vector.coordinates - np.ones(8)/8)
                    ),
                    "primary_emotion_vector": emotion_vector.coordinates[:3].tolist()
                }
            }
        }
        
        # Add temporal analysis if available
        if temporal_analysis:
            results["temporal_analysis"] = temporal_analysis
        
        logger.info("Mathematical sentiment analysis completed")
        return results

    def _analyze_temporal_trends(self, historical_data: List[Dict], current_score: float) -> Dict[str, Any]:
        """Analyze temporal sentiment trends with statistical rigor"""
        try:
            # Extract historical scores and timestamps
            scores = []
            timestamps = []
            
            for data_point in historical_data:
                if 'sentiment_score' in data_point and 'timestamp' in data_point:
                    scores.append(data_point['sentiment_score'])
                    timestamps.append(pd.to_datetime(data_point['timestamp']))
            
            if len(scores) < 3:
                return {"error": "Insufficient historical data for temporal analysis"}
            
            # Create time series
            ts_data = pd.Series(scores, index=timestamps).sort_index()
            
            # Calculate trend slope using linear regression
            x_numeric = np.arange(len(ts_data))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x_numeric, ts_data.values)
            
            # Calculate momentum (recent rate of change)
            if len(ts_data) >= 5:
                recent_scores = ts_data.tail(5).values
                momentum = np.mean(np.diff(recent_scores))
            else:
                momentum = slope
            
            # Calculate acceleration (change in momentum)
            if len(ts_data) >= 10:
                first_half_momentum = np.mean(np.diff(ts_data.head(len(ts_data)//2).values))
                second_half_momentum = np.mean(np.diff(ts_data.tail(len(ts_data)//2).values))
                acceleration = second_half_momentum - first_half_momentum
            else:
                acceleration = 0.0
            
            # Compare to baseline (30-day average if available)
            baseline_comparison = 0.0
            if len(ts_data) >= 30:
                baseline = ts_data.tail(30).mean()
                baseline_comparison = current_score - baseline
            
            return {
                "trend_slope": float(slope),
                "trend_significance": float(1 - p_value if p_value < 1 else 0),
                "trend_r_squared": float(r_value**2),
                "momentum": float(momentum),
                "acceleration": float(acceleration),
                "baseline_comparison": float(baseline_comparison),
                "data_points": len(scores)
            }
            
        except Exception as e:
            logger.error(f"Temporal analysis failed: {e}")
            return {"error": f"Temporal analysis failed: {str(e)}"}

# Global instance for performance
_mathematical_analyzer = None

def get_mathematical_analyzer() -> MathematicalSentimentAnalyzer:
    """Get singleton instance of mathematical analyzer for performance"""
    global _mathematical_analyzer
    if _mathematical_analyzer is None:
        _mathematical_analyzer = MathematicalSentimentAnalyzer()
    return _mathematical_analyzer
