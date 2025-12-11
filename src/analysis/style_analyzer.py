"""Text style analysis module for NLP metrics."""
import re
from typing import Dict, List
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np


class StyleAnalyzer:
    """Analyzes text style and linguistic patterns."""

    def __init__(self):
        """Initialize the style analyzer."""
        # Download required NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            pass  # Already downloaded

    def analyze(self, text: str) -> Dict:
        """Perform comprehensive style analysis on text.

        Args:
            text: Input text to analyze

        Returns:
            Dict containing various style metrics
        """
        # Tokenize
        sentences = sent_tokenize(text)
        words = word_tokenize(text.lower())

        # Filter out punctuation for word-based metrics
        words_only = [w for w in words if w.isalnum()]

        # Calculate all metrics
        metrics = {
            'lexical_diversity': self._lexical_diversity(words_only),
            'perplexity_proxy': self._perplexity_proxy(words_only),
            'sentence_complexity': self._sentence_complexity(sentences),
            'readability': self._readability_scores(text, sentences, words_only),
            'vocabulary_richness': self._vocabulary_richness(words_only),
            'punctuation_patterns': self._punctuation_patterns(text),
            'text_statistics': self._text_statistics(text, sentences, words_only)
        }

        return metrics

    def _lexical_diversity(self, words: List[str]) -> Dict:
        """Calculate lexical diversity metrics.

        Type-Token Ratio (TTR): unique words / total words
        Higher TTR = more diverse vocabulary (human-like)
        """
        if not words:
            return {'ttr': 0.0, 'unique_words': 0, 'total_words': 0}

        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0

        return {
            'ttr': round(ttr, 4),
            'unique_words': unique_words,
            'total_words': total_words,
            'interpretation': self._interpret_ttr(ttr)
        }

    def _interpret_ttr(self, ttr: float) -> str:
        """Interpret TTR score."""
        if ttr > 0.7:
            return "Very high diversity (likely human)"
        elif ttr > 0.5:
            return "High diversity (human-like)"
        elif ttr > 0.3:
            return "Moderate diversity"
        else:
            return "Low diversity (AI-like repetition)"

    def _perplexity_proxy(self, words: List[str]) -> Dict:
        """Calculate a proxy for perplexity without running a full LM.

        Uses word frequency distribution as a simple proxy.
        Lower entropy = more predictable (AI-like)
        """
        if not words:
            return {'entropy': 0.0, 'predictability': 'unknown'}

        # Calculate word frequency distribution
        word_freq = {}
        for word in words:
            word_freq[word] = word_freq.get(word, 0) + 1

        # Calculate entropy
        total = len(words)
        probabilities = [freq / total for freq in word_freq.values()]
        entropy = -sum(p * np.log2(p) if p > 0 else 0 for p in probabilities)

        # Normalize to 0-10 scale
        normalized_entropy = min(entropy / 10.0 * 10, 10)

        return {
            'entropy': round(entropy, 4),
            'normalized_score': round(normalized_entropy, 2),
            'predictability': self._interpret_entropy(entropy)
        }

    def _interpret_entropy(self, entropy: float) -> str:
        """Interpret entropy score."""
        if entropy < 3:
            return "Very predictable (AI-like)"
        elif entropy < 5:
            return "Somewhat predictable"
        elif entropy < 7:
            return "Moderately varied"
        else:
            return "Highly varied (human-like)"

    def _sentence_complexity(self, sentences: List[str]) -> Dict:
        """Analyze sentence structure complexity."""
        if not sentences:
            return {'avg_length': 0, 'complexity': 'N/A'}

        # Average sentence length in words
        sentence_lengths = [len(word_tokenize(s)) for s in sentences]
        avg_length = np.mean(sentence_lengths)
        std_length = np.std(sentence_lengths)

        # Count sentences with conjunctions (complexity indicator)
        complex_sentences = sum(
            1 for s in sentences
            if any(conj in s.lower() for conj in ['however', 'moreover', 'furthermore', 'nevertheless'])
        )

        return {
            'avg_sentence_length': round(avg_length, 2),
            'std_sentence_length': round(std_length, 2),
            'total_sentences': len(sentences),
            'complex_sentences': complex_sentences,
            'complexity_interpretation': self._interpret_complexity(avg_length)
        }

    def _interpret_complexity(self, avg_length: float) -> str:
        """Interpret sentence complexity."""
        if avg_length < 10:
            return "Simple sentences (AI-like brevity)"
        elif avg_length < 20:
            return "Moderate complexity"
        elif avg_length < 30:
            return "Complex sentences (human-like)"
        else:
            return "Very complex (possibly academic)"

    def _readability_scores(self, text: str, sentences: List[str], words: List[str]) -> Dict:
        """Calculate readability scores (Flesch-Kincaid, etc.)."""
        if not sentences or not words:
            return {'flesch_reading_ease': 0, 'grade_level': 0}

        # Count syllables (approximation)
        total_syllables = sum(self._count_syllables(word) for word in words)

        # Flesch Reading Ease
        # 206.835 - 1.015 * (words/sentences) - 84.6 * (syllables/words)
        words_per_sentence = len(words) / len(sentences)
        syllables_per_word = total_syllables / len(words) if words else 0

        flesch_reading_ease = (
            206.835
            - 1.015 * words_per_sentence
            - 84.6 * syllables_per_word
        )

        # Flesch-Kincaid Grade Level
        # 0.39 * (words/sentences) + 11.8 * (syllables/words) - 15.59
        grade_level = (
            0.39 * words_per_sentence
            + 11.8 * syllables_per_word
            - 15.59
        )

        return {
            'flesch_reading_ease': round(max(0, min(100, flesch_reading_ease)), 2),
            'grade_level': round(max(0, grade_level), 2),
            'interpretation': self._interpret_readability(flesch_reading_ease)
        }

    def _count_syllables(self, word: str) -> int:
        """Approximate syllable count for a word."""
        word = word.lower()
        syllables = 0
        vowels = 'aeiouy'
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllables += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith('e'):
            syllables -= 1

        # Every word has at least one syllable
        return max(1, syllables)

    def _interpret_readability(self, score: float) -> str:
        """Interpret Flesch Reading Ease score."""
        if score >= 90:
            return "Very easy to read (5th grade)"
        elif score >= 80:
            return "Easy to read (6th grade)"
        elif score >= 70:
            return "Fairly easy (7th grade)"
        elif score >= 60:
            return "Plain English (8-9th grade)"
        elif score >= 50:
            return "Fairly difficult (10-12th grade)"
        elif score >= 30:
            return "Difficult (college level)"
        else:
            return "Very difficult (professional)"

    def _vocabulary_richness(self, words: List[str]) -> Dict:
        """Analyze vocabulary sophistication."""
        if not words:
            return {'avg_word_length': 0, 'long_words': 0}

        # Average word length
        avg_word_length = np.mean([len(w) for w in words])

        # Count "sophisticated" words (7+ letters)
        long_words = sum(1 for w in words if len(w) >= 7)
        long_word_ratio = long_words / len(words) if words else 0

        return {
            'avg_word_length': round(avg_word_length, 2),
            'long_words_count': long_words,
            'long_word_ratio': round(long_word_ratio, 4),
            'interpretation': self._interpret_vocabulary(long_word_ratio)
        }

    def _interpret_vocabulary(self, ratio: float) -> str:
        """Interpret vocabulary sophistication."""
        if ratio > 0.3:
            return "Advanced vocabulary (academic/professional)"
        elif ratio > 0.2:
            return "Sophisticated vocabulary"
        elif ratio > 0.1:
            return "Standard vocabulary"
        else:
            return "Simple vocabulary"

    def _punctuation_patterns(self, text: str) -> Dict:
        """Analyze punctuation usage patterns."""
        # Count different punctuation types
        commas = text.count(',')
        periods = text.count('.')
        exclamations = text.count('!')
        questions = text.count('?')
        semicolons = text.count(';')
        colons = text.count(':')

        total_chars = len(text)
        total_punctuation = commas + periods + exclamations + questions + semicolons + colons

        punctuation_density = total_punctuation / total_chars if total_chars > 0 else 0

        return {
            'commas': commas,
            'periods': periods,
            'exclamations': exclamations,
            'questions': questions,
            'semicolons': semicolons,
            'colons': colons,
            'punctuation_density': round(punctuation_density, 4),
            'interpretation': self._interpret_punctuation(exclamations, questions)
        }

    def _interpret_punctuation(self, exclamations: int, questions: int) -> str:
        """Interpret punctuation patterns."""
        if exclamations > 3 or questions > 3:
            return "Informal/conversational (human-like emotion)"
        elif exclamations == 0 and questions == 0:
            return "Formal/declarative (AI-like)"
        else:
            return "Balanced punctuation"

    def _text_statistics(self, text: str, sentences: List[str], words: List[str]) -> Dict:
        """Basic text statistics."""
        return {
            'character_count': len(text),
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_words_per_sentence': round(len(words) / len(sentences), 2) if sentences else 0
        }
