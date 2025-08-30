"""
Enhanced Analysis Engine
Provides context-aware summaries focusing on recent developments with mathematical backing.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
import re
import logging
from openai import OpenAI
from .config import config
from .mathematical_sentiment import get_mathematical_analyzer

logger = logging.getLogger(__name__)

# Initialize OpenAI client
client = OpenAI(api_key=config['openai_api_key'])

@dataclass
class RecentEvent:
    """Structure for recent events with relevance scoring"""
    title: str
    date: datetime
    relevance_score: float
    impact_score: float
    source_url: str
    summary: str

@dataclass
class ContextualMetrics:
    """Contextual analysis metrics"""
    recency_weight: float
    relevance_score: float
    factual_density: float
    complexity_score: float
    information_entropy: float

class EnhancedAnalysisEngine:
    """
    Advanced analysis engine that focuses on recent developments
    with mathematical validation and contextual intelligence.
    """
    
    def __init__(self):
        self.math_analyzer = get_mathematical_analyzer()
        self.max_summary_length = 300  # Focus on concise, relevant summaries
        
    def _calculate_recency_weights(self, events: List[Dict], decay_rate: float = 0.1) -> np.ndarray:
        """Calculate time-decay weights for recent events"""
        if not events:
            return np.array([])
            
        current_time = datetime.now()
        weights = []
        
        for event in events:
            try:
                event_time = datetime.fromisoformat(event.get('date', str(current_time)))
                days_ago = (current_time - event_time).days
                
                # Exponential decay: more recent = higher weight
                weight = np.exp(-decay_rate * days_ago)
                weights.append(weight)
                
            except Exception:
                weights.append(0.1)  # Low weight for unparseable dates
                
        return np.array(weights)
    
    def _calculate_information_entropy(self, text: str) -> float:
        """Calculate information entropy of text content"""
        # Character-level entropy
        chars = list(text.lower())
        char_counts = {}
        
        for char in chars:
            char_counts[char] = char_counts.get(char, 0) + 1
        
        total_chars = len(chars)
        if total_chars == 0:
            return 0.0
            
        # Calculate entropy
        entropy = 0.0
        for count in char_counts.values():
            p = count / total_chars
            if p > 0:
                entropy -= p * np.log2(p)
                
        return entropy
    
    def _calculate_factual_density(self, text: str) -> float:
        """Calculate ratio of factual content to opinion/speculation"""
        # Look for factual indicators
        factual_indicators = [
            r'\d{4}',  # Years
            r'\d+%',   # Percentages  
            r'\$\d+',  # Dollar amounts
            r'\d+\.\d+', # Decimals/measurements
            r'according to', r'reported', r'announced', r'confirmed',
            r'data shows', r'statistics', r'study found', r'research indicates'
        ]
        
        opinion_indicators = [
            r'i think', r'believe', r'feel', r'opinion', r'seems',
            r'probably', r'might', r'could be', r'appears to be',
            r'speculation', r'rumor', r'allegedly'
        ]
        
        factual_count = sum(len(re.findall(pattern, text.lower())) for pattern in factual_indicators)
        opinion_count = sum(len(re.findall(pattern, text.lower())) for pattern in opinion_indicators)
        
        total_indicators = factual_count + opinion_count
        if total_indicators == 0:
            return 0.5  # Neutral if no indicators found
            
        return factual_count / total_indicators
    
    def _calculate_complexity_score(self, text: str) -> float:
        """Calculate linguistic complexity of text"""
        sentences = text.split('.')
        words = text.split()
        
        if not sentences or not words:
            return 0.0
        
        # Average sentence length
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Vocabulary diversity (unique words / total words)
        unique_words = len(set(word.lower().strip('.,!?;:') for word in words))
        vocab_diversity = unique_words / len(words) if words else 0
        
        # Complex word ratio (words with 3+ syllables, approximated by 7+ characters)
        complex_words = sum(1 for word in words if len(word) > 7)
        complex_ratio = complex_words / len(words) if words else 0
        
        # Combine metrics (normalized to [0, 1])
        complexity = (
            min(avg_sentence_length / 20, 1) * 0.4 +  # Sentence length component
            vocab_diversity * 0.3 +                   # Vocabulary diversity
            min(complex_ratio * 3, 1) * 0.3           # Complex words component
        )
        
        return complexity
    
    def generate_enhanced_summary(
        self, 
        content: str, 
        search_results: List[Dict] = None,
        historical_context: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Generate enhanced summary focusing on recent developments with mathematical backing
        """
        logger.info("Generating enhanced summary with mathematical validation...")
        
        # Calculate contextual metrics
        factual_density = self._calculate_factual_density(content)
        complexity_score = self._calculate_complexity_score(content)
        information_entropy = self._calculate_information_entropy(content)
        
        # Process recent events from search results
        recent_events = []
        recency_weights = np.array([1.0])  # Default weight
        
        if search_results:
            # Extract and weight recent events
            for result in search_results[:5]:  # Top 5 most relevant
                try:
                    # Estimate relevance based on search ranking and recency
                    relevance = 1.0 - (search_results.index(result) * 0.15)  # Decreasing relevance
                    
                    recent_events.append({
                        'title': result.get('title', ''),
                        'date': datetime.now().isoformat(),  # Approximate for search results
                        'relevance_score': relevance,
                        'snippet': result.get('snippet', ''),
                        'url': result.get('link', '')
                    })
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            if recent_events:
                recency_weights = self._calculate_recency_weights(recent_events)
        
        # Generate focused summary using GPT-4o with mathematical context
        summary_prompt = self._create_enhanced_summary_prompt(
            content, recent_events, factual_density, complexity_score
        )
        
        try:
            summary_response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": self._get_summary_system_prompt()},
                    {"role": "user", "content": summary_prompt}
                ],
                max_tokens=400,  # Enforce conciseness
                temperature=0.3  # Lower temperature for more factual summaries
            )
            
            enhanced_summary = summary_response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            # Fallback to truncated content
            enhanced_summary = content[:300] + "..." if len(content) > 300 else content
        
        # Calculate final metrics
        contextual_metrics = {
            "factual_density": factual_density,
            "complexity_score": complexity_score, 
            "information_entropy": information_entropy,
            "recent_events_count": len(recent_events),
            "average_recency_weight": float(np.mean(recency_weights)) if len(recency_weights) > 0 else 0.0
        }
        
        return {
            "enhanced_summary": enhanced_summary,
            "word_count": len(content.split()),
            "contextual_metrics": contextual_metrics,
            "recent_events_identified": len(recent_events),
            "mathematical_backing": {
                "factual_confidence": factual_density,
                "information_content": information_entropy,
                "linguistic_complexity": complexity_score
            }
        }
    
    def _create_enhanced_summary_prompt(
        self, 
        content: str, 
        recent_events: List[Dict],
        factual_density: float,
        complexity_score: float
    ) -> str:
        """Create targeted prompt for enhanced summary generation"""
        
        recent_events_context = ""
        if recent_events:
            recent_events_context = "\n\nRECENT RELEVANT EVENTS:\n"
            for i, event in enumerate(recent_events[:3], 1):
                recent_events_context += f"{i}. {event['title']}: {event['snippet']}\n"
        
        return f"""
        Analyze the following content and provide a concise, mathematically-informed summary focusing on recent developments and quantifiable insights.
        
        CONTENT ANALYSIS METRICS:
        - Factual Density: {factual_density:.3f} (higher = more factual)
        - Complexity Score: {complexity_score:.3f} (higher = more sophisticated)
        
        REQUIREMENTS:
        1. Maximum 2-3 sentences (under 300 characters)
        2. Focus on recent developments (last 30 days when available)
        3. Include quantifiable metrics when present (percentages, numbers, dates)
        4. Prioritize factual information over speculation
        5. Mention key numerical insights or trends if available
        
        {recent_events_context}
        
        CONTENT TO ANALYZE:
        {content[:2000]}...
        
        Provide a summary that captures the most recent and quantifiable developments:
        """
    
    def _get_summary_system_prompt(self) -> str:
        """System prompt for enhanced summary generation"""
        return """
        You are an expert analyst specializing in creating concise, mathematically-informed summaries.
        Your summaries focus on recent developments, quantifiable metrics, and factual content.
        
        Key principles:
        1. Prioritize recent developments (mentioned dates, current events)
        2. Include specific numbers, percentages, and measurements when available
        3. Focus on facts over opinions or speculation
        4. Keep summaries under 300 characters while maintaining information density
        5. Use precise, analytical language
        
        Always structure summaries to highlight the most recent and quantifiable information first.
        """
    
    def analyze_content_structure(self, content: str, file_type: str = "text") -> Dict[str, Any]:
        """
        Analyze document structure for better processing
        """
        structure_analysis = {
            "total_length": len(content),
            "word_count": len(content.split()),
            "sentence_count": len(content.split('.')),
            "paragraph_count": len(content.split('\n\n')),
            "file_type": file_type
        }
        
        if file_type == "pdf":
            # Look for PDF-specific structure
            structure_analysis["has_headers"] = bool(re.search(r'^[A-Z\s]{10,}$', content, re.MULTILINE))
            structure_analysis["has_page_numbers"] = bool(re.search(r'\d+\s*$', content, re.MULTILINE))
            
        elif file_type == "csv":
            # Analyze CSV structure
            lines = content.split('\n')
            structure_analysis["row_count"] = len(lines)
            if lines:
                structure_analysis["column_count"] = len(lines[0].split(','))
                structure_analysis["has_header"] = not lines[0].replace(',', '').replace(' ', '').isdigit()
        
        return structure_analysis

# Global instance for performance
_enhanced_analyzer = None

def get_enhanced_analyzer() -> EnhancedAnalysisEngine:
    """Get singleton instance of enhanced analyzer"""
    global _enhanced_analyzer
    if _enhanced_analyzer is None:
        _enhanced_analyzer = EnhancedAnalysisEngine()
    return _enhanced_analyzer