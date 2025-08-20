"""
Natural Language Processing Components

Core NLP functionality for analyzing text content, extracting patterns,
and understanding contextual information in vote messages.
"""

import re
import logging
from dataclasses import dataclass
from typing import Dict, List, Set, Any
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ContentAnalysisResult:
    """Result of comprehensive content analysis."""
    
    sentiment_score: float = 0.0      # -1 to 1
    urgency_level: float = 0.0        # 0 to 1
    certainty_level: float = 0.0      # 0 to 1
    emotional_intensity: float = 0.0  # 0 to 1
    technical_complexity: int = 0
    word_count: int = 0
    unique_words: int = 0
    readability_score: float = 0.0


class PatternLibrary:
    """Library of semantic patterns for vote interpretation."""
    
    def __init__(self):
        self.patterns = {
            'strong_approve': [
                r'\b(strongly?|enthusiastically?) (approve|support|endorse)\b',
                r'\b(absolutely|definitely) (yes|approve)\b',
                r'\b(excellent|perfect|ideal) (proposal|idea)\b'
            ],
            'conditional_approve': [
                r'\bapprove (if|provided|assuming)\b',
                r'\bsupport (with|under) conditions?\b',
                r'\byes,? but (only if|provided)\b'
            ],
            'weak_approve': [
                r'\bi guess (yes|okay|approve)\b',
                r'\b(reluctantly?|hesitantly?) approve\b',
                r'\b(probably|maybe) yes\b'
            ],
            'strong_reject': [
                r'\b(strongly?|absolutely) (reject|oppose)\b',
                r'\b(terrible|awful|horrible) (idea|proposal)\b',
                r'\b(definitely|absolutely) no\b'
            ],
            'conditional_reject': [
                r'\breject (unless|until|if not)\b',
                r'\bno,? but would (support|approve) if\b',
                r'\bcannot approve (without|unless)\b'
            ],
            'weak_reject': [
                r'\b(probably|likely) no\b',
                r'\b(hesitant|reluctant) to support\b',
                r'\bhave (serious )?concerns?\b'
            ],
            'abstain': [
                r'\b(abstain|neutral|undecided)\b',
                r'\bneed more (information|time)\b',
                r'\bno (strong )?opinion\b'
            ],
            'clarification': [
                r'\bwhat (about|if|does)\b',
                r'\bhow (will|would|does)\b',
                r'\bcan (you|someone) (clarify|explain)\b'
            ]
        }
        
        self._compile_patterns()
    
    def _compile_patterns(self):
        """Compile regex patterns for efficiency."""
        self.compiled_patterns = {}
        for category, pattern_list in self.patterns.items():
            self.compiled_patterns[category] = [
                re.compile(pattern, re.IGNORECASE) for pattern in pattern_list
            ]


class ContextualAnalyzer:
    """Analyzes contextual information from text content."""
    
    def __init__(self):
        self.sentiment_keywords = {
            'positive': {'good', 'great', 'excellent', 'perfect', 'wonderful', 'brilliant'},
            'negative': {'bad', 'terrible', 'awful', 'horrible', 'wrong', 'problematic'},
            'uncertainty': {'maybe', 'perhaps', 'possibly', 'uncertain', 'unclear'},
            'certainty': {'definitely', 'certainly', 'absolutely', 'sure', 'confident'},
            'urgency': {'urgent', 'immediately', 'asap', 'quickly', 'critical', 'emergency'}
        }
    
    def analyze_sentiment(self, text: str) -> float:
        """Analyze sentiment score from -1 to 1."""
        words = set(text.lower().split())
        positive = len(words.intersection(self.sentiment_keywords['positive']))
        negative = len(words.intersection(self.sentiment_keywords['negative']))
        total = positive + negative
        return (positive - negative) / total if total > 0 else 0.0
    
    def analyze_certainty(self, text: str) -> float:
        """Analyze certainty level from 0 to 1."""
        words = set(text.lower().split())
        uncertainty = len(words.intersection(self.sentiment_keywords['uncertainty']))
        certainty = len(words.intersection(self.sentiment_keywords['certainty']))
        total = uncertainty + certainty
        return certainty / total if total > 0 else 0.5
    
    def analyze_urgency(self, text: str) -> float:
        """Analyze urgency level from 0 to 1."""
        words = set(text.lower().split())
        urgency_count = len(words.intersection(self.sentiment_keywords['urgency']))
        return min(1.0, urgency_count / 3)


class NaturalLanguageProcessor:
    """Main NLP processor for vote text analysis."""
    
    def __init__(self):
        self.pattern_library = PatternLibrary()
        self.contextual_analyzer = ContextualAnalyzer()
    
    def analyze_content(self, text: str) -> ContentAnalysisResult:
        """Perform comprehensive content analysis."""
        words = text.split()
        word_set = set(w.lower() for w in words)
        
        return ContentAnalysisResult(
            sentiment_score=self.contextual_analyzer.analyze_sentiment(text),
            urgency_level=self.contextual_analyzer.analyze_urgency(text),
            certainty_level=self.contextual_analyzer.analyze_certainty(text),
            emotional_intensity=self._calculate_emotional_intensity(text),
            technical_complexity=self._calculate_technical_complexity(word_set),
            word_count=len(words),
            unique_words=len(word_set),
            readability_score=self._calculate_readability(words)
        )
    
    def find_pattern_matches(self, text: str) -> Dict[str, float]:
        """Find pattern matches with confidence scores."""
        matches = {}
        
        for category, patterns in self.pattern_library.compiled_patterns.items():
            confidence = 0.0
            for pattern in patterns:
                if pattern.search(text):
                    confidence = max(confidence, 0.8)  # Base confidence for pattern match
            
            if confidence > 0:
                matches[category] = confidence
        
        return matches
    
    def extract_conditions(self, text: str) -> List[str]:
        """Extract conditional statements."""
        conditions = []
        condition_patterns = [
            r'if ([^,.!?]+)',
            r'provided (?:that )?([^,.!?]+)',
            r'assuming ([^,.!?]+)',
            r'unless ([^,.!?]+)'
        ]
        
        for pattern in condition_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            conditions.extend(m.strip() for m in matches if m.strip())
        
        return conditions
    
    def extract_concerns(self, text: str) -> List[str]:
        """Extract expressed concerns."""
        concerns = []
        concern_patterns = [
            r'concern(?:ed)? (?:about |with )?([^,.!?]+)',
            r'worry(?:ied)? (?:about |that )?([^,.!?]+)',
            r'problem (?:with |is )?([^,.!?]+)'
        ]
        
        for pattern in concern_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concerns.extend(m.strip() for m in matches if m.strip())
        
        return concerns
    
    def extract_alternatives(self, text: str) -> List[str]:
        """Extract alternative suggestions."""
        alternatives = []
        alt_patterns = [
            r'(?:alternative|instead|rather),? ([^,.!?]+)',
            r'suggest(?:ion)? (?:is |that )?([^,.!?]+)',
            r'what if ([^,.!?]+)'
        ]
        
        for pattern in alt_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            alternatives.extend(m.strip() for m in matches if m.strip())
        
        return alternatives
    
    def extract_questions(self, text: str) -> List[str]:
        """Extract questions from text."""
        sentences = re.split(r'[.!?]+', text)
        questions = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if (sentence.endswith('?') or 
                sentence.lower().startswith(('what', 'how', 'why', 'when', 'where', 'who')) or
                'question' in sentence.lower()):
                questions.append(sentence)
        
        return questions
    
    def _calculate_emotional_intensity(self, text: str) -> float:
        """Calculate emotional intensity based on linguistic markers."""
        # Simple implementation - could be enhanced with more sophisticated analysis
        intensity_markers = ['!', '!!', '!!!', 'very', 'extremely', 'absolutely', 'totally']
        count = sum(text.count(marker) for marker in intensity_markers)
        return min(1.0, count / 5)
    
    def _calculate_technical_complexity(self, words: Set[str]) -> int:
        """Calculate technical complexity score."""
        technical_terms = {
            'algorithm', 'implementation', 'architecture', 'database', 'api',
            'framework', 'infrastructure', 'optimization', 'scalability'
        }
        return len(words.intersection(technical_terms))
    
    def _calculate_readability(self, words: List[str]) -> float:
        """Calculate simple readability score."""
        if not words:
            return 0.0
        
        avg_word_length = sum(len(word) for word in words) / len(words)
        # Simple readability: lower average word length = higher readability
        return max(0.0, 1.0 - (avg_word_length - 4) / 10)