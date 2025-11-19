"""Feature extraction for hybrid model (perplexity + stylometric features)."""
import math
from typing import Dict, List

import nltk
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

# Download required NLTK data (run once)
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class FeatureExtractor:
    """Extract linguistic and perplexity features from text."""

    def __init__(self, perplexity_model: str = "gpt2", device: str = "cpu"):
        """Initialize feature extractor.

        Args:
            perplexity_model: HuggingFace model for perplexity calculation
            device: Device to run model on
        """
        self.device = device
        self.perplexity_model = GPT2LMHeadModel.from_pretrained(perplexity_model).to(device)
        self.perplexity_tokenizer = GPT2TokenizerFast.from_pretrained(perplexity_model)
        self.perplexity_model.eval()

    def compute_perplexity(self, text: str) -> float:
        """Compute perplexity of text using GPT-2.

        Lower perplexity often indicates AI-generated text.

        Args:
            text: Input text

        Returns:
            Perplexity score
        """
        encodings = self.perplexity_tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=1024
        )

        max_length = encodings.input_ids.size(1)
        stride = 512
        nlls = []

        for i in range(0, max_length, stride):
            begin_loc = max(i + stride - 1024, 0)
            end_loc = min(i + stride, max_length)
            trg_len = end_loc - i

            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(self.device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = self.perplexity_model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs.loss * trg_len

            nlls.append(neg_log_likelihood)

        ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
        return ppl.item()

    def compute_stylometric_features(self, text: str) -> Dict[str, float]:
        """Compute stylometric features.

        Args:
            text: Input text

        Returns:
            Dictionary of features
        """
        # Tokenize into sentences and words
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)

        # Filter out punctuation for some metrics
        words_only = [w for w in words if w.isalnum()]

        # Compute features
        features = {}

        # Type-Token Ratio (vocabulary richness)
        if len(words_only) > 0:
            features['type_token_ratio'] = len(set(words_only)) / len(words_only)
        else:
            features['type_token_ratio'] = 0.0

        # Average sentence length
        if len(sentences) > 0:
            features['avg_sentence_length'] = len(words) / len(sentences)
        else:
            features['avg_sentence_length'] = 0.0

        # Lexical diversity (unique words / total words)
        if len(words) > 0:
            features['lexical_diversity'] = len(set(words)) / len(words)
        else:
            features['lexical_diversity'] = 0.0

        # Punctuation ratio
        punctuation_count = sum(1 for c in text if c in '.,!?;:')
        features['punctuation_ratio'] = punctuation_count / len(text) if len(text) > 0 else 0.0

        # Average word length
        if len(words_only) > 0:
            features['avg_word_length'] = sum(len(w) for w in words_only) / len(words_only)
        else:
            features['avg_word_length'] = 0.0

        return features

    def extract_all_features(self, text: str) -> Dict[str, float]:
        """Extract all features (perplexity + stylometric).

        Args:
            text: Input text

        Returns:
            Dictionary of all features
        """
        features = {}

        # Perplexity
        features['perplexity'] = self.compute_perplexity(text)

        # Stylometric features
        stylometric = self.compute_stylometric_features(text)
        features.update(stylometric)

        return features

    def extract_features_batch(self, texts: List[str]) -> List[Dict[str, float]]:
        """Extract features for a batch of texts.

        Args:
            texts: List of input texts

        Returns:
            List of feature dictionaries
        """
        return [self.extract_all_features(text) for text in texts]
