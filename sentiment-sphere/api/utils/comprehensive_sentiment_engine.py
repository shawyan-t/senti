"""
Comprehensive Sentiment/Affect Engine
Implementation of the end-to-end pipeline specification for general web sentiment analysis.
Following the mathematical blueprint with proper aggregation, weighting, and calibration.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import re
import json
import logging
import uuid
from scipy.stats import entropy, norm
from scipy.spatial.distance import jensenshannon
import hashlib
from collections import defaultdict
import math

logger = logging.getLogger(__name__)

@dataclass
class CanonicalUnit:
    """Canonical unit of content after normalization"""
    text: str
    source_domain: str
    url: str
    author: Optional[str]
    platform_stats: Dict[str, float]  # upvotes, karma, views
    publish_time: datetime
    last_edit_time: Optional[datetime]
    first_seen_time: datetime
    language: str
    cluster_id: str
    thread_depth: int  # For Reddit
    retrieval_score: float
    length: int
    unit_id: str

@dataclass
class UnitScores:
    """Raw classifier scores for a unit"""
    # Sentiment distribution
    sentiment_probs: np.ndarray  # [neg, neu, pos]
    sentiment_scalar: float  # s_u^(0) in [-1,1]
    
    # Emotion probabilities (Plutchik/Ekman)
    emotion_probs: Dict[str, float]  # joy, sad, anger, fear, disgust, surprise
    
    # VAD regression
    vad: np.ndarray  # [valence, arousal, dominance] in [0,1]^3
    
    # Tone/style
    subjectivity: float
    politeness: float
    formality: float
    assertiveness: float
    
    # Special cases
    sarcasm_prob: float
    toxicity_prob: float
    stance_prob: float  # σ_u = p_pro - p_anti
    uncertainty_prob: float
    misinformation_risk: float
    language_quality: float
    
    # Model uncertainty
    predictive_entropy: float
    epistemic_variance: float

@dataclass
class WeightedUnit:
    """Unit with all computed weights"""
    unit: CanonicalUnit
    scores: UnitScores
    
    # Individual weights
    w_fresh: float
    w_novelty: float
    w_source: float
    w_tree: float
    w_length: float
    w_duplicate: float
    
    # Final weight
    final_weight: float
    
    # Corrected scores
    sentiment_sarc_corrected: float
    sentiment_tox_corrected: float

class ComprehensiveSentimentEngine:
    """
    Implementation of the comprehensive sentiment analysis pipeline.
    Follows the mathematical specification exactly.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or self._default_config()
        self._historical_centroids = {}
        self._domain_scores = {}
        
    def _wilson_confidence_interval(self, successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
        """Calculate Wilson confidence interval for binomial proportion"""
        if total == 0:
            return (0.0, 0.0)
        
        z = norm.ppf((1 + confidence) / 2)  # Z-score for confidence level
        p = successes / total
        
        # Wilson score interval formula
        denominator = 1 + z**2 / total
        center = (p + z**2 / (2 * total)) / denominator
        margin = z * np.sqrt((p * (1 - p) + z**2 / (4 * total)) / total) / denominator
        
        lower = max(0.0, center - margin)
        upper = min(1.0, center + margin)
        
        return (lower, upper)
        
    def _default_config(self) -> Dict[str, Any]:
        """Default configuration following the specification"""
        return {
            # Freshness decay
            'tau_general': 7.0,  # days
            'tau_social': 2.0,   # days  
            'tau_evergreen': 30.0,  # days
            'w_min_freshness': 0.2,
            
            # Sarcasm correction
            'beta_sarcasm': 0.5,
            'gamma_sarcasm': 0.2,
            
            # Toxicity dampening
            'lambda_toxicity': 0.3,
            
            # Novelty
            'alpha_novelty': 1.0,
            
            # Aggregation
            'trim_percent': 0.1,  # 10% trimmed mean
            'tukey_c0': 4.685,
            
            # EMA for trends
            'ema_alpha': 0.4,
            
            # Reddit thread decay
            'rho_thread': 0.7,
            
            # Confidence calibration
            'confidence_alpha0': 2.0,
            'confidence_alpha1': 1.5,
            'confidence_alpha2': 0.8,
        }
    
    def process_query(self, units: List[CanonicalUnit]) -> Dict[str, Any]:
        """
        Main pipeline: process all units for a query and return comprehensive results
        """
        if not units:
            return self._empty_result()
        
        # Step 1: Run unit-level classifiers
        scored_units = []
        for unit in units:
            scores = self._classify_unit(unit)
            scored_units.append((unit, scores))
        
        # Step 2: Apply unit-level corrections and weights
        weighted_units = []
        for unit, scores in scored_units:
            weighted_unit = self._compute_unit_weights(unit, scores, units)
            weighted_units.append(weighted_unit)
        
        # Step 3: Cluster-level robust aggregation
        clusters = self._group_by_cluster(weighted_units)
        cluster_summaries = {}
        for cluster_id, cluster_units in clusters.items():
            cluster_summaries[cluster_id] = self._aggregate_cluster(cluster_units)
        
        # Step 4: Query-level aggregation
        result = self._aggregate_query(cluster_summaries, weighted_units)
        
        return result
    
    def _classify_unit(self, unit: CanonicalUnit) -> UnitScores:
        """
        Run all unit-level classifiers. 
        In production, this would call actual ML models.
        For now, implementing mathematical frameworks based on text analysis.
        """
        text = unit.text.lower()
        
        # 1. Sentiment distribution using multiple signals
        sentiment_probs = self._compute_sentiment_distribution(text)
        sentiment_scalar = sentiment_probs[2] - sentiment_probs[0]  # pos - neg
        
        # 2. Emotion probabilities (simplified Plutchik mapping)
        emotion_probs = self._compute_emotion_probabilities(text)
        
        # 3. VAD regression (valence, arousal, dominance)
        vad = self._compute_vad(text, sentiment_scalar)
        
        # 4. Tone/style analysis
        subjectivity = self._compute_subjectivity(text)
        politeness = self._compute_politeness(text)
        formality = self._compute_formality(text)
        assertiveness = self._compute_assertiveness(text)
        
        # 5. Special classifiers
        sarcasm_prob = self._detect_sarcasm(text)
        toxicity_prob = self._detect_toxicity(text)
        stance_prob = self._compute_stance(text)
        uncertainty_prob = self._detect_uncertainty(text)
        misinformation_risk = self._assess_misinformation_risk(text)
        language_quality = self._assess_language_quality(text)
        
        # 6. Model uncertainty
        predictive_entropy = entropy(sentiment_probs)
        epistemic_variance = 0.1  # Would come from model ensemble/MC-dropout
        
        return UnitScores(
            sentiment_probs=sentiment_probs,
            sentiment_scalar=sentiment_scalar,
            emotion_probs=emotion_probs,
            vad=vad,
            subjectivity=subjectivity,
            politeness=politeness,
            formality=formality,
            assertiveness=assertiveness,
            sarcasm_prob=sarcasm_prob,
            toxicity_prob=toxicity_prob,
            stance_prob=stance_prob,
            uncertainty_prob=uncertainty_prob,
            misinformation_risk=misinformation_risk,
            language_quality=language_quality,
            predictive_entropy=predictive_entropy,
            epistemic_variance=epistemic_variance
        )
    
    def _compute_sentiment_distribution(self, text: str) -> np.ndarray:
        """Compute [neg, neu, pos] probabilities using comprehensive lexical analysis"""
        words = text.lower().split()
        if not words:
            return np.array([0.33, 0.34, 0.33])
        
        # Expanded positive indicators with weights
        pos_patterns = {
            r'\b(excellent|outstanding|exceptional|magnificent|superb|brilliant|incredible|amazing|fantastic|wonderful)\b': 3.0,
            r'\b(great|good|nice|beautiful|perfect|awesome|impressive|remarkable|stellar|phenomenal)\b': 2.0,
            r'\b(positive|better|improved|successful|victory|win|benefit|advantage|profit|gain)\b': 1.5,
            r'\b(like|enjoy|love|appreciate|pleased|happy|satisfied|delighted|thrilled)\b': 1.0
        }
        
        # Expanded negative indicators with weights  
        neg_patterns = {
            r'\b(terrible|awful|horrible|disgusting|appalling|dreadful|atrocious|abysmal|catastrophic)\b': 3.0,
            r'\b(bad|worst|stupid|ugly|disappointing|failure|disaster|crisis|problem|issue)\b': 2.0,
            r'\b(hate|dislike|reject|oppose|decline|fall|drop|loss|damage|harm)\b': 1.5,
            r'\b(sad|angry|frustrated|annoyed|concerned|worried|troubled|upset)\b': 1.0
        }
        
        # Calculate weighted scores
        pos_score = 0
        for pattern, weight in pos_patterns.items():
            pos_score += len(re.findall(pattern, text)) * weight
            
        neg_score = 0  
        for pattern, weight in neg_patterns.items():
            neg_score += len(re.findall(pattern, text)) * weight
        
        # Intensifiers and diminishers
        intensifiers = len(re.findall(r'\b(very|extremely|incredibly|absolutely|totally|really|quite|rather|significantly)\b', text))
        diminishers = len(re.findall(r'\b(slightly|somewhat|barely|hardly|scarcely|a bit|a little)\b', text))
        
        # Negations with context window
        negation_contexts = re.findall(r'\b(not|no|never|nothing|neither|nor|isn\'t|aren\'t|wasn\'t|weren\'t|don\'t|doesn\'t|didn\'t|won\'t|wouldn\'t|can\'t|couldn\'t|shouldn\'t)\s+\w+', text)
        negations = len(negation_contexts)
        
        # Apply modifiers
        intensity_multiplier = 1 + (intensifiers * 0.3) - (diminishers * 0.2)
        pos_score *= intensity_multiplier
        neg_score *= intensity_multiplier
        
        # Apply negation effect (flip sentiment in negation contexts)
        if negations > 0:
            neg_flip = min(negations * 0.5, 1.0)  # Cap flip effect
            original_pos, original_neg = pos_score, neg_score
            pos_score = original_pos * (1 - neg_flip) + original_neg * neg_flip * 0.7
            neg_score = original_neg * (1 - neg_flip) + original_pos * neg_flip * 0.7
        
        # Normalize word count
        word_count = len(words)
        pos_score = pos_score / word_count
        neg_score = neg_score / word_count
        
        # Convert to probabilities with neutral baseline
        neutral_baseline = 0.3  # Higher baseline for neutral
        total = pos_score + neg_score + neutral_baseline
        
        if total == 0:
            return np.array([0.33, 0.34, 0.33])
            
        probs = np.array([neg_score/total, neutral_baseline/total, pos_score/total])
        return probs / probs.sum()  # Ensure normalization
    
    def _compute_emotion_probabilities(self, text: str) -> Dict[str, float]:
        """Map text to comprehensive Plutchik-8 emotions using weighted lexical indicators"""
        # Plutchik's 8 basic emotions with comprehensive pattern matching
        emotion_patterns = {
            'joy': {
                r'\b(joy|joyful|happy|happiness|cheerful|elated|ecstatic|blissful|delighted|gleeful)\b': 3.0,
                r'\b(pleased|glad|content|satisfied|upbeat|optimistic|positive|bright|sunny)\b': 2.0,
                r'\b(good|nice|great|wonderful|excellent|amazing|fantastic|awesome|brilliant)\b': 1.5,
                r'\b(smile|laugh|celebrate|party|fun|enjoyable|pleasant|lovely)\b': 1.0
            },
            'sadness': {
                r'\b(sad|sadness|depressed|depression|miserable|sorrowful|grief|mourning|melancholy)\b': 3.0,
                r'\b(unhappy|downhearted|dejected|despondent|gloomy|blue|down|low)\b': 2.0,
                r'\b(disappointed|regret|remorse|loss|pain|hurt|ache|suffer)\b': 1.5,
                r'\b(cry|tears|weep|sob|mourn|lament|sigh|frown)\b': 1.0
            },
            'anger': {
                r'\b(anger|angry|furious|rage|wrath|fury|livid|irate|incensed|outraged)\b': 3.0,
                r'\b(mad|annoyed|irritated|frustrated|agitated|vexed|pissed|ticked)\b': 2.0,
                r'\b(hate|hatred|loathe|despise|resent|bitter|hostile|aggressive)\b': 2.5,
                r'\b(damn|hell|stupid|idiot|ridiculous|absurd|unfair|wrong)\b': 1.0
            },
            'fear': {
                r'\b(fear|afraid|scared|terrified|frightened|petrified|horrified|panic)\b': 3.0,
                r'\b(anxious|anxiety|worried|nervous|uneasy|apprehensive|concerned)\b': 2.0,
                r'\b(threat|danger|risk|unsafe|insecure|vulnerable|exposed)\b': 1.5,
                r'\b(avoid|escape|hide|retreat|flee|run|emergency|alarm)\b': 1.0
            },
            'disgust': {
                r'\b(disgusted|disgusting|revolted|revolting|repulsed|repulsive|sickened)\b': 3.0,
                r'\b(gross|nasty|foul|vile|loathsome|abhorrent|repugnant|offensive)\b': 2.5,
                r'\b(sick|nauseous|queasy|distasteful|unpleasant|awful|terrible)\b': 2.0,
                r'\b(reject|refuse|decline|turn away|avoid|shun)\b': 1.0
            },
            'surprise': {
                r'\b(surprised|surprise|shocked|shock|amazed|astonished|stunned|astounded)\b': 3.0,
                r'\b(unexpected|sudden|abrupt|startled|bewildered|perplexed|confused)\b': 2.0,
                r'\b(wow|whoa|omg|incredible|unbelievable|remarkable|extraordinary)\b': 2.5,
                r'\b(wonder|curious|puzzled|baffled|mystified)\b': 1.0
            },
            'trust': {
                r'\b(trust|trustworthy|reliable|dependable|faithful|loyal|devoted)\b': 3.0,
                r'\b(confidence|confident|secure|safe|certain|sure|belief|believe)\b': 2.0,
                r'\b(support|backing|endorse|approve|accept|welcome|embrace)\b': 1.5,
                r'\b(friend|friendship|ally|partner|team|together|unity)\b': 1.0
            },
            'anticipation': {
                r'\b(anticipate|anticipation|expect|expectation|hope|hopeful|eager|excited)\b': 3.0,
                r'\b(future|tomorrow|next|coming|approaching|imminent|pending)\b': 2.0,
                r'\b(plan|planning|prepare|ready|waiting|await|look forward)\b': 1.5,
                r'\b(soon|shortly|eventually|potential|possibility|opportunity)\b': 1.0
            }
        }
        
        emotions = {}
        total_weighted_score = 0
        
        # Calculate weighted emotion scores
        for emotion, patterns in emotion_patterns.items():
            emotion_score = 0
            for pattern, weight in patterns.items():
                matches = len(re.findall(pattern, text.lower()))
                emotion_score += matches * weight
            
            emotions[emotion] = emotion_score
            total_weighted_score += emotion_score
        
        # Normalize to probabilities
        if total_weighted_score == 0:
            # Return uniform distribution if no emotional indicators found
            return {emotion: 1/8 for emotion in emotions}
        
        # Convert to probabilities
        emotion_probs = {emotion: score/total_weighted_score for emotion, score in emotions.items()}
        
        # Apply minimum baseline to prevent zero probabilities
        min_prob = 0.02  # 2% minimum for each emotion
        for emotion in emotion_probs:
            emotion_probs[emotion] = max(emotion_probs[emotion], min_prob)
        
        # Renormalize after applying minimums
        total_prob = sum(emotion_probs.values())
        emotion_probs = {emotion: prob/total_prob for emotion, prob in emotion_probs.items()}
        
        return emotion_probs
    
    def _compute_vad(self, text: str, sentiment: float) -> np.ndarray:
        """Compute Valence, Arousal, Dominance scores"""
        # Valence maps closely to sentiment
        valence = (sentiment + 1) / 2  # Map [-1,1] to [0,1]
        
        # Arousal indicators
        arousal_words = len(re.findall(r'\b(excited|energetic|intense|wild|crazy|dynamic|explosive)\b', text))
        calm_words = len(re.findall(r'\b(calm|peaceful|quiet|serene|relaxed|still)\b', text))
        arousal = 0.5 + 0.3 * (arousal_words - calm_words) / max(len(text.split()), 1)
        arousal = np.clip(arousal, 0, 1)
        
        # Dominance indicators
        dominant_words = len(re.findall(r'\b(control|power|strong|confident|command|lead|boss)\b', text))
        submissive_words = len(re.findall(r'\b(weak|submit|follow|obey|helpless|powerless)\b', text))
        dominance = 0.5 + 0.3 * (dominant_words - submissive_words) / max(len(text.split()), 1)
        dominance = np.clip(dominance, 0, 1)
        
        return np.array([valence, arousal, dominance])
    
    def _compute_subjectivity(self, text: str) -> float:
        """Compute subjectivity score [0,1] - higher means more subjective"""
        words = len(text.split())
        if words == 0:
            return 0.5
        
        # Subjective indicators with weights
        subjective_patterns = {
            r'\b(i think|i feel|i believe|in my opinion|personally|i guess|i suppose|seems to me)\b': 3.0,
            r'\b(seems|appears|looks like|sounds|feels|smells|tastes)\b': 2.0,
            r'\b(probably|likely|possibly|maybe|perhaps|might|could|should)\b': 1.5,
            r'\b(love|hate|like|dislike|prefer|enjoy|appreciate|despise)\b': 2.5,
            r'\b(beautiful|ugly|amazing|terrible|wonderful|awful|fantastic|horrible)\b': 2.0
        }
        
        # Objective indicators with weights  
        objective_patterns = {
            r'\b(data|research|study|analysis|report|statistics|evidence|facts|findings)\b': 3.0,
            r'\b(according to|based on|research shows|studies indicate|data suggests)\b': 2.5,
            r'\b(measured|calculated|observed|recorded|documented|verified)\b': 2.0,
            r'\b(\d+%|\d+\.\d+|statistics|numbers|figures|metrics|results)\b': 1.5,
            r'\b(conclude|determine|establish|prove|demonstrate|confirm)\b': 1.0
        }
        
        subj_score = 0
        for pattern, weight in subjective_patterns.items():
            subj_score += len(re.findall(pattern, text.lower())) * weight
            
        obj_score = 0
        for pattern, weight in objective_patterns.items():
            obj_score += len(re.findall(pattern, text.lower())) * weight
        
        # Normalize by word count
        subj_ratio = subj_score / words
        obj_ratio = obj_score / words
        
        # Convert to 0-1 scale with logistic function
        if obj_ratio == 0 and subj_ratio == 0:
            return 0.5  # Neutral default
        
        raw_score = subj_ratio / (subj_ratio + obj_ratio)
        return np.clip(raw_score, 0, 1)
    
    def _compute_politeness(self, text: str) -> float:
        """Compute politeness score with comprehensive patterns"""
        words = len(text.split())
        if words == 0:
            return 0.5
        
        # Polite indicators with weights
        polite_patterns = {
            r'\b(please|kindly|would you|could you|may i|excuse me)\b': 3.0,
            r'\b(thank you|thanks|grateful|appreciate|apologize|sorry)\b': 2.5,
            r'\b(sir|madam|mr\.|mrs\.|ms\.|dr\.|professor)\b': 2.0,
            r'\b(respectfully|humbly|cordially|sincerely|regards)\b': 2.0,
            r'\b(welcome|blessing|honor|privilege|delighted)\b': 1.5
        }
        
        # Rude/impolite indicators with weights
        rude_patterns = {
            r'\b(fuck|shit|damn|hell|ass|bitch|bastard|idiot|stupid)\b': 3.0,
            r'\b(shut up|go away|get lost|screw you|piss off)\b': 2.5,
            r'\b(whatever|duh|obviously|seriously|ridiculous)\b': 1.5,
            r'\b(lame|sucks|crap|jerk|loser|moron)\b': 2.0
        }
        
        polite_score = 0
        for pattern, weight in polite_patterns.items():
            polite_score += len(re.findall(pattern, text.lower())) * weight
            
        rude_score = 0
        for pattern, weight in rude_patterns.items():
            rude_score += len(re.findall(pattern, text.lower())) * weight
        
        # Calculate politeness ratio
        if polite_score == 0 and rude_score == 0:
            return 0.6  # Slightly polite default
        
        total = polite_score + rude_score
        politeness = polite_score / total if total > 0 else 0.5
        
        return np.clip(politeness, 0, 1)
    
    def _compute_formality(self, text: str) -> float:
        """Compute formality score based on language patterns"""
        words = text.split()
        if not words:
            return 0.5
        
        # Formal indicators
        formal_patterns = {
            r'\b(furthermore|moreover|therefore|consequently|nevertheless|however|additionally)\b': 3.0,
            r'\b(thus|hence|accordingly|subsequently|notwithstanding|albeit)\b': 2.5,
            r'\b(regarding|concerning|pursuant|whereas|herein|thereof|whereby)\b': 2.0,
            r'\b(utilize|commence|terminate|demonstrate|facilitate|implement)\b': 1.5,
            r'\b(analysis|examination|investigation|assessment|evaluation)\b': 1.0
        }
        
        # Informal indicators
        informal_patterns = {
            r'\b(gonna|wanna|gotta|kinda|sorta|yeah|nah|yep|nope)\b': 3.0,
            r'\b(lol|omg|wtf|btw|fyi|tbh|imo|afaik)\b': 2.5,
            r'\b(awesome|cool|crazy|nuts|wild|sick|sweet)\b': 2.0,
            r'\b(stuff|things|guys|dude|bro|hey|hi|bye)\b': 1.5
        }
        
        formal_score = 0
        for pattern, weight in formal_patterns.items():
            formal_score += len(re.findall(pattern, text.lower())) * weight
            
        informal_score = 0
        for pattern, weight in informal_patterns.items():
            informal_score += len(re.findall(pattern, text.lower())) * weight
        
        # Sentence length factor (longer = more formal)
        sentences = [s.strip() for s in re.split(r'[.!?]', text) if s.strip()]
        if sentences:
            avg_sentence_length = np.mean([len(s.split()) for s in sentences])
            length_factor = (avg_sentence_length - 8) / 20  # Normalize around 8 words
        else:
            length_factor = 0
        
        # Word length factor (longer words = more formal)
        avg_word_length = np.mean([len(w) for w in words])
        word_length_factor = (avg_word_length - 4) / 10  # Normalize around 4 chars
        
        # Calculate base formality
        if formal_score == 0 and informal_score == 0:
            base_formality = 0.5
        else:
            total = formal_score + informal_score
            base_formality = formal_score / total if total > 0 else 0.5
        
        # Combine with structural factors
        final_formality = base_formality + (length_factor * 0.2) + (word_length_factor * 0.1)
        return np.clip(final_formality, 0, 1)
    
    def _compute_assertiveness(self, text: str) -> float:
        """Compute assertiveness score with nuanced patterns"""
        words = len(text.split())
        if words == 0:
            return 0.5
        
        # Assertive indicators
        assertive_patterns = {
            r'\b(must|will|shall|need to|have to|require|demand|insist)\b': 3.0,
            r'\b(should|ought to|supposed to|expected to|necessary|essential)\b': 2.5,
            r'\b(definitely|certainly|absolutely|clearly|obviously|undoubtedly)\b': 2.0,
            r'\b(always|never|all|every|none|everyone|nobody|everything|nothing)\b': 2.0,
            r'\b(command|order|direct|instruct|tell|force|make)\b': 1.5
        }
        
        # Tentative/hedging indicators  
        tentative_patterns = {
            r'\b(maybe|perhaps|might|could|possibly|probably|potentially)\b': 2.5,
            r'\b(seems|appears|looks like|sounds like|feels like|suggests)\b': 2.0,
            r'\b(i think|i believe|i guess|i suppose|in my opinion)\b': 1.5,
            r'\b(somewhat|rather|quite|fairly|relatively|partially)\b': 1.0,
            r'\b(try|attempt|hope|wish|would like|prefer)\b': 1.0
        }
        
        assertive_score = 0
        for pattern, weight in assertive_patterns.items():
            assertive_score += len(re.findall(pattern, text.lower())) * weight
            
        tentative_score = 0
        for pattern, weight in tentative_patterns.items():
            tentative_score += len(re.findall(pattern, text.lower())) * weight
        
        # Calculate assertiveness ratio
        if assertive_score == 0 and tentative_score == 0:
            return 0.5  # Neutral default
        
        total = assertive_score + tentative_score
        assertiveness = assertive_score / total if total > 0 else 0.5
        
        return np.clip(assertiveness, 0, 1)
    
    def _detect_sarcasm(self, text: str) -> float:
        """Detect sarcasm probability"""
        # Sarcasm indicators
        quotes = len(re.findall(r'"[^"]*"', text))  # Quoted text
        caps = len(re.findall(r'\b[A-Z]{2,}\b', text))  # ALL CAPS
        contradictions = len(re.findall(r'\b(yeah right|sure|obviously|totally)\b', text))
        
        sarcasm_score = (quotes + caps + contradictions) / max(len(text.split()), 1)
        return np.clip(sarcasm_score * 2, 0, 1)
    
    def _detect_toxicity(self, text: str) -> float:
        """Detect toxicity probability"""
        toxic_words = len(re.findall(r'\b(hate|kill|die|stupid|idiot|moron|bitch|asshole)\b', text))
        threats = len(re.findall(r'\b(i will|gonna kill|destroy you|fuck you)\b', text))
        
        toxicity = (toxic_words + threats * 2) / max(len(text.split()), 1)
        return np.clip(toxicity * 3, 0, 1)
    
    def _compute_stance(self, text: str) -> float:
        """Compute stance (pro - anti) for general topics"""
        pro_words = len(re.findall(r'\b(support|favor|agree|approve|endorse|back)\b', text))
        anti_words = len(re.findall(r'\b(oppose|against|disagree|reject|condemn|criticize)\b', text))
        
        stance = (pro_words - anti_words) / max(len(text.split()), 1)
        return np.clip(stance * 2, -1, 1)
    
    def _detect_uncertainty(self, text: str) -> float:
        """Detect uncertainty/hedging"""
        uncertain_words = len(re.findall(r'\b(maybe|perhaps|might|could|possibly|rumor|allegedly|supposedly)\b', text))
        return np.clip(uncertain_words / max(len(text.split()), 1) * 3, 0, 1)
    
    def _assess_misinformation_risk(self, text: str) -> float:
        """Assess misinformation risk (conservative scoring)"""
        # Look for absolutist claims without evidence
        absolute_claims = len(re.findall(r'\b(always|never|all|none|everyone|nobody) \w+', text))
        evidence_words = len(re.findall(r'\b(study|research|data|evidence|source|according)\b', text))
        
        if evidence_words > 0:
            return 0.1  # Low risk if evidence cited
        
        risk = absolute_claims / max(len(text.split()), 1)
        return np.clip(risk * 2, 0, 0.7)  # Cap at 0.7 to avoid over-penalization
    
    def _assess_language_quality(self, text: str) -> float:
        """Assess language quality/perplexity proxy"""
        # Simple heuristics for quality
        words = text.split()
        if not words:
            return 0.0
        
        # Vocabulary diversity
        unique_words = len(set(words))
        diversity = unique_words / len(words)
        
        # Grammar proxy (ratio of function words)
        function_words = len(re.findall(r'\b(the|and|or|but|in|on|at|to|for|of|with|by)\b', text))
        grammar_score = function_words / len(words)
        
        # Spelling errors proxy (repeated characters)
        spelling_errors = len(re.findall(r'(.)\1{2,}', text))  # 3+ repeated chars
        error_rate = spelling_errors / len(words)
        
        quality = diversity * 0.5 + grammar_score * 0.3 + (1 - error_rate) * 0.2
        return np.clip(quality, 0, 1)
    
    def _compute_unit_weights(self, unit: CanonicalUnit, scores: UnitScores, all_units: List[CanonicalUnit]) -> WeightedUnit:
        """Compute all weight components for a unit following the specification"""
        
        # 3.1 Sarcasm-aware sentiment correction
        s_sarc = self._apply_sarcasm_correction(scores.sentiment_scalar, scores.sarcasm_prob)
        
        # 3.2 Toxicity-aware correction  
        s_tox = s_sarc * (1 - self.config['lambda_toxicity'] * scores.toxicity_prob)
        
        # 3.3 Freshness weight
        w_fresh = self._compute_freshness_weight(unit)
        
        # 3.4 Novelty weight
        w_novelty = self._compute_novelty_weight(unit)
        
        # 3.5 Source credibility weight
        w_source = self._compute_source_weight(unit, scores)
        
        # Reddit tree weight
        w_tree = self._compute_tree_weight(unit) if unit.thread_depth > 0 else 1.0
        
        # 3.6 Length control
        w_length = self._compute_length_weight(unit, all_units)
        
        # Duplication control
        w_duplicate = self._compute_duplicate_weight(unit, all_units)
        
        # 3.7 Final weight
        final_weight = w_fresh * w_novelty * w_source * w_tree * w_length * w_duplicate
        final_weight = np.clip(final_weight, 0.001, 1.0)  # ε = 0.001
        
        return WeightedUnit(
            unit=unit,
            scores=scores,
            w_fresh=w_fresh,
            w_novelty=w_novelty,
            w_source=w_source,
            w_tree=w_tree,
            w_length=w_length,
            w_duplicate=w_duplicate,
            final_weight=final_weight,
            sentiment_sarc_corrected=s_sarc,
            sentiment_tox_corrected=s_tox
        )
    
    def _apply_sarcasm_correction(self, sentiment: float, sarcasm_prob: float) -> float:
        """Apply sarcasm correction using Option A from specification"""
        beta = self.config['beta_sarcasm']
        gamma = self.config['gamma_sarcasm']
        
        attenuation = (1 - beta * sarcasm_prob) * sentiment
        flip_component = gamma * sarcasm_prob * np.sign(sentiment) * abs(sentiment)
        
        return attenuation - flip_component
    
    def _compute_freshness_weight(self, unit: CanonicalUnit) -> float:
        """Compute freshness weight with exponential decay"""
        delta_t = (datetime.now() - unit.publish_time).total_seconds() / (24 * 3600)  # days
        
        # Choose tau based on content type (simplified)
        tau = self.config['tau_general']
        if 'reddit' in unit.source_domain.lower():
            tau = self.config['tau_social']
        
        w_fresh = np.exp(-delta_t / tau)
        return max(w_fresh, self.config['w_min_freshness'])
    
    def _compute_novelty_weight(self, unit: CanonicalUnit) -> float:
        """Compute novelty weight (simplified - would use embeddings in production)"""
        # For now, use text length and unique word ratio as proxy
        words = unit.text.split()
        unique_ratio = len(set(words)) / max(len(words), 1)
        
        # Simulate novelty score
        novelty = min(1, self.config['alpha_novelty'] * unique_ratio)
        return novelty
    
    def _compute_source_weight(self, unit: CanonicalUnit, scores: UnitScores) -> float:
        """Compute source credibility weight"""
        # Domain score (simplified)
        domain_score = self._get_domain_score(unit.source_domain)
        
        # Bot probability (simplified - assume human)
        p_bot = 0.1
        
        # Engagement component
        votes = unit.platform_stats.get('upvotes', 0)
        v_95 = np.percentile([u.platform_stats.get('upvotes', 0) for u in [unit]], 95)
        engagement = np.log(1 + votes) / np.log(1 + max(v_95, 1))
        
        # Retrieval and quality components
        retrieval_prior = 1 / (1 + np.exp(-(2 * unit.retrieval_score - 1)))  # sigmoid
        quality_prior = 1 / (1 + np.exp(-(2 * scores.language_quality - 1)))
        
        w_source = domain_score * (1 - p_bot) * engagement * retrieval_prior * quality_prior
        return np.clip(w_source, 0, 1)
    
    def _get_domain_score(self, domain: str) -> float:
        """Get domain credibility score"""
        if domain in self._domain_scores:
            return self._domain_scores[domain]
        
        # Simple heuristic scoring
        if any(trusted in domain for trusted in ['reuters', 'ap.org', 'bbc', 'nyt']):
            score = 0.9
        elif any(social in domain for social in ['reddit', 'twitter', 'facebook']):
            score = 0.6
        elif domain.endswith('.edu') or domain.endswith('.gov'):
            score = 0.8
        else:
            score = 0.7  # Default
        
        self._domain_scores[domain] = score
        return score
    
    def _compute_tree_weight(self, unit: CanonicalUnit) -> float:
        """Compute Reddit tree weight"""
        rho = self.config['rho_thread']
        return 1 / (1 + np.exp(-(3 - 0.5 * unit.thread_depth)))  # Sigmoid decay
    
    def _compute_length_weight(self, unit: CanonicalUnit, all_units: List[CanonicalUnit]) -> float:
        """Compute length penalty weight"""
        lengths = [u.length for u in all_units]
        L_ref = np.median(lengths) if lengths else 100
        
        return min(1, np.sqrt(unit.length / max(L_ref, 1)))
    
    def _compute_duplicate_weight(self, unit: CanonicalUnit, all_units: List[CanonicalUnit]) -> float:
        """Compute duplicate clustering weight"""
        # Count units in same cluster from same domain
        same_cluster_domain = sum(1 for u in all_units 
                                if u.cluster_id == unit.cluster_id and u.source_domain == unit.source_domain)
        
        return 1 / np.sqrt(max(same_cluster_domain, 1))
    
    def _group_by_cluster(self, weighted_units: List[WeightedUnit]) -> Dict[str, List[WeightedUnit]]:
        """Group weighted units by cluster ID"""
        clusters = defaultdict(list)
        for wu in weighted_units:
            clusters[wu.unit.cluster_id].append(wu)
        return dict(clusters)
    
    def _aggregate_cluster(self, cluster_units: List[WeightedUnit]) -> Dict[str, Any]:
        """Robust aggregation within cluster using Tukey biweight"""
        if not cluster_units:
            return {}
        
        # Extract sentiment values and weights
        sentiments = np.array([wu.sentiment_tox_corrected for wu in cluster_units])
        weights = np.array([wu.final_weight for wu in cluster_units])
        
        # Tukey biweight aggregation
        cluster_sentiment = self._tukey_biweight(sentiments, weights)
        
        # Aggregate other metrics using weighted means
        total_weight = weights.sum()
        
        # Emotion aggregation
        emotions = defaultdict(list)
        for wu in cluster_units:
            for emotion, prob in wu.scores.emotion_probs.items():
                emotions[emotion].append(prob * wu.final_weight)
        
        cluster_emotions = {emotion: sum(probs)/total_weight for emotion, probs in emotions.items()}
        
        # VAD aggregation
        vad_weighted = np.array([wu.scores.vad * wu.final_weight for wu in cluster_units])
        cluster_vad = vad_weighted.sum(axis=0) / total_weight
        
        return {
            'sentiment': cluster_sentiment,
            'emotions': cluster_emotions,
            'vad': cluster_vad,
            'weight': total_weight,
            'unit_count': len(cluster_units),
            'representative_text': max(cluster_units, key=lambda x: x.final_weight).unit.text[:200]
        }
    
    def _tukey_biweight(self, values: np.ndarray, weights: np.ndarray) -> float:
        """Tukey biweight (bisquare) robust estimation"""
        if len(values) == 0:
            return 0.0
        
        if len(values) == 1:
            return values[0]
        
        # Initial estimate (weighted median)
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        
        cumsum = np.cumsum(sorted_weights)
        median_idx = np.searchsorted(cumsum, cumsum[-1] / 2)
        median = sorted_values[min(median_idx, len(sorted_values) - 1)]
        
        # MAD estimation
        deviations = np.abs(values - median)
        mad = np.median(deviations)
        
        if mad == 0:
            return median
        
        # Tukey biweight
        c0 = self.config['tukey_c0']
        u = (values - median) / (c0 * mad)
        
        # Biweight function: ψ(u) = u(1-u²)² for |u| < 1, else 0
        mask = np.abs(u) < 1
        psi = np.zeros_like(u)
        psi[mask] = u[mask] * (1 - u[mask]**2)**2
        
        # Weight function: w(u) = (1-u²)² for |u| < 1, else 0  
        w = np.zeros_like(u)
        w[mask] = (1 - u[mask]**2)**2
        
        numerator = np.sum(weights * values * w)
        denominator = np.sum(weights * w)
        
        if denominator == 0:
            return median
        
        return numerator / denominator
    
    def _aggregate_query(self, cluster_summaries: Dict[str, Dict], weighted_units: List[WeightedUnit]) -> Dict[str, Any]:
        """Final query-level aggregation following specification section 5"""
        if not cluster_summaries:
            return self._empty_result()
        
        # 5.1 Primary sentiment score
        cluster_weights = np.array([cs['weight'] for cs in cluster_summaries.values()])
        cluster_sentiments = np.array([cs['sentiment'] for cs in cluster_summaries.values()])
        
        S = np.sum(cluster_weights * cluster_sentiments) / np.sum(cluster_weights)
        
        # Confidence calculation
        within_cluster_var = np.mean([self._compute_cluster_variance(cs) for cs in cluster_summaries.values()])
        between_cluster_var = np.var(cluster_sentiments, ddof=1) if len(cluster_sentiments) > 1 else 0
        avg_entropy = np.mean([wu.scores.predictive_entropy for wu in weighted_units])
        
        confidence = 1 / (1 + np.exp(-(
            self.config['confidence_alpha0'] - 
            self.config['confidence_alpha1'] * np.sqrt(between_cluster_var) - 
            self.config['confidence_alpha2'] * avg_entropy
        )))
        
        # 5.2 Disagreement/polarization
        all_sentiments = np.array([wu.sentiment_tox_corrected for wu in weighted_units])
        all_weights = np.array([wu.final_weight for wu in weighted_units])
        
        disagreement = np.sum(all_weights * (all_sentiments - S)**2) / np.sum(all_weights)
        
        # Polarity shares with Wilson confidence intervals
        total_units = len(all_sentiments)
        pos_units = np.sum(all_sentiments > 0)
        neg_units = np.sum(all_sentiments < 0)
        neu_units = total_units - pos_units - neg_units
        
        pi_pos = np.sum(all_weights[all_sentiments > 0]) / np.sum(all_weights)
        pi_neg = np.sum(all_weights[all_sentiments < 0]) / np.sum(all_weights)
        pi_neu = 1 - pi_pos - pi_neg
        
        # Wilson confidence intervals for polarity shares
        pi_pos_ci = self._wilson_confidence_interval(pos_units, total_units)
        pi_neg_ci = self._wilson_confidence_interval(neg_units, total_units)
        pi_neu_ci = self._wilson_confidence_interval(neu_units, total_units)
        
        # 5.3 Emotion/tone summaries
        total_weight = sum(cs['weight'] for cs in cluster_summaries.values())
        
        # Aggregate emotions
        emotion_sums = defaultdict(float)
        for cs in cluster_summaries.values():
            for emotion, prob in cs['emotions'].items():
                emotion_sums[emotion] += prob * cs['weight']
        
        avg_emotions = {emotion: total/total_weight for emotion, total in emotion_sums.items()}
        
        # Aggregate VAD
        vad_weighted = np.array([cs['vad'] * cs['weight'] for cs in cluster_summaries.values()])
        avg_vad = vad_weighted.sum(axis=0) / total_weight
        
        # 5.4 Sarcasm/toxicity rates
        sarcasm_rate = np.sum([wu.scores.sarcasm_prob * wu.final_weight for wu in weighted_units]) / np.sum(all_weights)
        toxicity_rate = np.sum([wu.scores.toxicity_prob * wu.final_weight for wu in weighted_units]) / np.sum(all_weights)
        
        # 5.5 Freshness & novelty
        freshness_score = np.sum([wu.w_fresh * wu.final_weight for wu in weighted_units]) / np.sum(all_weights)
        novelty_score = np.sum([wu.w_novelty * wu.final_weight for wu in weighted_units]) / np.sum(all_weights)
        
        # Calculate additional tone metrics 
        subjectivity_scores = [wu.scores.subjectivity * wu.final_weight for wu in weighted_units]
        politeness_scores = [wu.scores.politeness * wu.final_weight for wu in weighted_units]
        formality_scores = [wu.scores.formality * wu.final_weight for wu in weighted_units]
        assertiveness_scores = [wu.scores.assertiveness * wu.final_weight for wu in weighted_units]
        
        avg_subjectivity = np.sum(subjectivity_scores) / np.sum(all_weights)
        avg_politeness = np.sum(politeness_scores) / np.sum(all_weights)
        avg_formality = np.sum(formality_scores) / np.sum(all_weights)
        avg_assertiveness = np.sum(assertiveness_scores) / np.sum(all_weights)
        
        # Stance calculation (if applicable)
        stance_scores = [wu.scores.stance_prob * wu.final_weight for wu in weighted_units]
        avg_stance = np.sum(stance_scores) / np.sum(all_weights)
        
        # Top emotions sorted by weight
        top_emotions = sorted(avg_emotions.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Build comprehensive result following exact specification format
        result = {
            # Core sentiment with confidence and disagreement
            'sentiment': {
                'score': float(S),  # S ∈ [-1,1]
                'confidence': float(confidence),  # ∈ [0,1]
                'disagreement': float(disagreement),  # D
                'polarity': {
                    'positive': float(pi_pos),  # π_+
                    'negative': float(pi_neg),  # π_-  
                    'neutral': float(pi_neu),   # π_0
                    'wilson_ci': {
                        'positive': [float(pi_pos_ci[0]), float(pi_pos_ci[1])],
                        'negative': [float(pi_neg_ci[0]), float(pi_neg_ci[1])], 
                        'neutral': [float(pi_neu_ci[0]), float(pi_neu_ci[1])]
                    }
                }
            },
            
            # Emotion analysis with top emotions
            'emotion': avg_emotions,  # \bar{\mathbf{e}}
            'top_emotions': [emotion for emotion, _ in top_emotions],
            
            # VAD analysis
            'vad': {  # \bar{\mathbf{v}}
                'valence': float(avg_vad[0]),
                'arousal': float(avg_vad[1]),
                'dominance': float(avg_vad[2])
            },
            
            # Tone analysis
            'tone': {
                'subjectivity': float(avg_subjectivity),    # \bar{p}^{\text{subj}}
                'politeness': float(avg_politeness),        # \bar{p}^{\text{polite}}
                'formality': float(avg_formality),          # \bar{p}^{\text{form}}
                'assertiveness': float(avg_assertiveness)   # \bar{p}^{\text{assert}}
            },
            
            # Special rates
            'sarcasm_rate': float(sarcasm_rate),     # \bar{p}^{\text{sarc}}
            'toxicity_rate': float(toxicity_rate),   # \bar{p}^{\text{tox}}
            'stance_score': float(avg_stance),       # \bar{\sigma}
            
            # Temporal signals
            'freshness_score': float(freshness_score),  # F
            'novelty_score': float(novelty_score),
            'topic_shift_jsd': 0.0,  # Would calculate JSD in production
            
            # Trend dynamics (simplified for single analysis)
            'trend': {
                'momentum': 0.0,     # M
                'acceleration': 0.0, # A
                'current_score': float(S)  # S_t
            },
            
            # Platform breakdown
            'breakdown': {
                'platform_shares': {
                    'user_input': total_weight,  # All weight from user input
                    'search_results': float(len([u for u in weighted_units if u.unit.source_domain != 'user_input']))
                },
                'domain_counts': len(set(wu.unit.source_domain for wu in weighted_units)),
                'weight_distribution': [float(wu.final_weight) for wu in weighted_units]
            },
            
            # Evidence quality and coverage
            'coverage': {
                'total_units': len(weighted_units),
                'total_clusters': len(cluster_summaries),
                'total_weight': float(total_weight),
                'language_coverage': {'en': len(weighted_units)},  # Would expand in production
                'time_span_hours': 0.0,  # Single point analysis
                'weight_concentration': float(max([wu.final_weight for wu in weighted_units]) / np.mean([wu.final_weight for wu in weighted_units]))
            },
            
            # Explanations with cluster details
            'explanations': {
                'top_clusters': self._get_detailed_clusters(cluster_summaries, weighted_units, limit=3),
                'key_phrases': self._extract_key_phrases(weighted_units),
                'weight_factors': self._explain_weight_factors(weighted_units)
            }
        }
        
        return result
    
    def _compute_cluster_variance(self, cluster_summary: Dict) -> float:
        """Compute within-cluster variance (simplified)"""
        return 0.1  # Would compute from actual unit variances
    
    def _get_detailed_clusters(self, cluster_summaries: Dict, weighted_units: List[WeightedUnit], limit: int = 3) -> List[Dict]:
        """Get detailed top clusters with explanations"""
        sorted_clusters = sorted(cluster_summaries.items(), 
                               key=lambda x: x[1]['weight'], reverse=True)
        
        top_clusters = []
        for cluster_id, summary in sorted_clusters[:limit]:
            # Find units in this cluster
            cluster_units = [wu for wu in weighted_units if wu.unit.cluster_id == cluster_id]
            
            # Explain why this cluster was weighted highly
            explanations = []
            if summary['weight'] > np.mean([cs['weight'] for cs in cluster_summaries.values()]):
                explanations.append("above average weight ✓")
            
            avg_freshness = np.mean([wu.w_fresh for wu in cluster_units])
            if avg_freshness > 0.7:
                explanations.append("very recent ✓")
            
            avg_quality = np.mean([wu.scores.language_quality for wu in cluster_units])
            if avg_quality > 0.6:
                explanations.append("high language quality ✓")
                
            avg_toxicity = np.mean([wu.scores.toxicity_prob for wu in cluster_units])
            if avg_toxicity < 0.2:
                explanations.append("low toxicity ✓")
            
            top_clusters.append({
                'cluster_id': cluster_id,
                'weight': float(summary['weight']),
                'sentiment': float(summary['sentiment']),
                'unit_count': summary['unit_count'],
                'representative_text': summary['representative_text'][:150] + "...",
                'why_important': explanations,
                'avg_freshness': float(avg_freshness),
                'avg_quality': float(avg_quality),
                'sources': list(set(wu.unit.source_domain for wu in cluster_units))
            })
        
        return top_clusters
    
    def _extract_key_phrases(self, weighted_units: List[WeightedUnit]) -> List[Dict]:
        """Extract key phrases that drove the sentiment analysis"""
        # Simple keyword extraction based on weights and sentiment indicators
        key_phrases = []
        
        for wu in sorted(weighted_units, key=lambda x: x.final_weight, reverse=True)[:5]:
            text = wu.unit.text.lower()
            
            # Find sentiment-bearing phrases
            pos_phrases = re.findall(r'\b(?:great|excellent|amazing|fantastic|wonderful|brilliant)\b[^.!?]*', text)
            neg_phrases = re.findall(r'\b(?:terrible|awful|horrible|bad|disappointing|failed)\b[^.!?]*', text)
            
            for phrase in pos_phrases[:2]:
                key_phrases.append({
                    'phrase': phrase.strip()[:100],
                    'sentiment_impact': '+',
                    'weight': float(wu.final_weight),
                    'source': wu.unit.source_domain
                })
            
            for phrase in neg_phrases[:2]:
                key_phrases.append({
                    'phrase': phrase.strip()[:100],
                    'sentiment_impact': '-',
                    'weight': float(wu.final_weight),
                    'source': wu.unit.source_domain
                })
        
        return key_phrases[:10]  # Top 10 key phrases
    
    def _explain_weight_factors(self, weighted_units: List[WeightedUnit]) -> Dict[str, Any]:
        """Explain what factors contributed to unit weighting"""
        total_units = len(weighted_units)
        
        # Analyze weight distribution
        weights = [wu.final_weight for wu in weighted_units]
        
        return {
            'weight_range': {
                'min': float(min(weights)),
                'max': float(max(weights)),
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights))
            },
            'factor_analysis': {
                'high_freshness_units': len([wu for wu in weighted_units if wu.w_fresh > 0.8]),
                'high_novelty_units': len([wu for wu in weighted_units if wu.w_novelty > 0.7]),
                'high_source_units': len([wu for wu in weighted_units if wu.w_source > 0.8]),
                'low_toxicity_units': len([wu for wu in weighted_units if wu.scores.toxicity_prob < 0.1])
            },
            'weight_concentration': {
                'top_10_percent_weight_share': float(np.sum(sorted(weights, reverse=True)[:max(1, total_units//10)]) / np.sum(weights)),
                'gini_coefficient': self._calculate_gini(weights)
            }
        }
    
    def _calculate_gini(self, weights: List[float]) -> float:
        """Calculate Gini coefficient for weight distribution inequality"""
        if len(weights) <= 1:
            return 0.0
        
        sorted_weights = sorted(weights)
        n = len(sorted_weights)
        cumsum = np.cumsum(sorted_weights)
        
        return 1 - (2 / n) * (n + 1 - 2 * np.sum(cumsum) / cumsum[-1])
    
    def _get_top_clusters(self, cluster_summaries: Dict, limit: int = 3) -> List[Dict]:
        """Legacy function - kept for compatibility"""
        return self._get_detailed_clusters(cluster_summaries, [], limit)
    
    def _empty_result(self) -> Dict[str, Any]:
        """Return empty result structure matching full result format"""
        return {
            'sentiment': {'score': 0.0, 'confidence': 0.0, 'disagreement': 0.0, 
                         'polarity': {'positive': 0.0, 'negative': 0.0, 'neutral': 1.0,
                                    'wilson_ci': {'positive': [0.0, 0.0], 'negative': [0.0, 0.0], 'neutral': [0.8, 1.0]}}},
            'emotion': {},
            'top_emotions': [],
            'vad': {'valence': 0.5, 'arousal': 0.5, 'dominance': 0.5},
            'tone': {'subjectivity': 0.5, 'politeness': 0.5, 'formality': 0.5, 'assertiveness': 0.5},
            'sarcasm_rate': 0.0,
            'toxicity_rate': 0.0,
            'stance_score': 0.0,
            'freshness_score': 0.0,
            'novelty_score': 0.0,
            'topic_shift_jsd': 0.0,
            'trend': {'momentum': 0.0, 'acceleration': 0.0, 'current_score': 0.0},
            'breakdown': {'platform_shares': {}, 'domain_counts': 0, 'weight_distribution': []},
            'coverage': {
                'total_units': 0,
                'total_clusters': 0,
                'total_weight': 0.0,
                'language_coverage': {'en': 0},
                'time_span_hours': 0.0,
                'weight_concentration': 0.0
            },
            'explanations': {'top_clusters': [], 'key_phrases': [], 'weight_factors': {}}
        }

# Global instance
_comprehensive_engine = None

def get_comprehensive_engine() -> ComprehensiveSentimentEngine:
    """Get singleton instance of comprehensive engine"""
    global _comprehensive_engine
    if _comprehensive_engine is None:
        _comprehensive_engine = ComprehensiveSentimentEngine()
    return _comprehensive_engine