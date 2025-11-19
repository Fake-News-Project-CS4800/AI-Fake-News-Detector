"""Model explainability using SHAP and saliency methods."""
import re
from typing import Dict, List, Tuple

import torch
import numpy as np
from captum.attr import LayerIntegratedGradients
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)


class ModelExplainer:
    """Explainer for fake news classifier."""

    def __init__(self, model, tokenizer, device='cpu'):
        """Initialize explainer.

        Args:
            model: Trained model (HuggingFace AutoModelForSequenceClassification)
            tokenizer: HuggingFace tokenizer
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model.eval()

        # Get the base model (roberta, bert, distilbert, etc.)
        if hasattr(model, 'roberta'):
            base_model = model.roberta
        elif hasattr(model, 'bert'):
            base_model = model.bert
        elif hasattr(model, 'distilbert'):
            base_model = model.distilbert
        else:
            base_model = None

        # Initialize Integrated Gradients if base model available
        if base_model is not None:
            self.ig = LayerIntegratedGradients(
                self.forward_func,
                base_model.embeddings.word_embeddings
            )
            self.base_model = base_model
        else:
            self.ig = None
            self.base_model = None

    def forward_func(self, inputs):
        """Forward function for attribution.

        Args:
            inputs: Input embeddings

        Returns:
            Model logits
        """
        # Create attention mask (all ones for embedded inputs)
        attention_mask = torch.ones(inputs.shape[:2], dtype=torch.long).to(self.device)

        # Forward pass with embeddings using base model
        outputs = self.base_model(
            inputs_embeds=inputs,
            attention_mask=attention_mask
        )

        # Get pooled output (usually CLS token)
        if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            pooled = outputs.last_hidden_state[:, 0, :]

        # Pass through classifier head
        logits = self.model.classifier(pooled)

        return logits

    def get_token_attributions(
        self,
        text: str,
        target_class: int
    ) -> List[Tuple[str, float]]:
        """Get token-level attributions using Integrated Gradients.

        Args:
            text: Input text
            target_class: Target class index

        Returns:
            List of (token, attribution_score) tuples
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get embeddings
        if self.base_model is not None:
            embeddings = self.base_model.embeddings.word_embeddings(input_ids)

            # Compute attributions
            attributions = self.ig.attribute(
                embeddings,
                target=target_class,
                n_steps=50
            )
        else:
            # Fallback if no base model available
            return []

        # Sum attribution across embedding dimension
        attributions = attributions.sum(dim=-1).squeeze(0)
        attributions = attributions / torch.norm(attributions)
        attributions = attributions.cpu().detach().numpy()

        # Get tokens
        tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])

        # Pair tokens with attributions
        token_attributions = []
        for token, attr in zip(tokens, attributions):
            if token not in ['[CLS]', '[SEP]', '[PAD]']:
                token_attributions.append((token, float(attr)))

        return token_attributions

    def get_heuristic_reasons(self, text: str, predicted_class: int) -> List[str]:
        """Generate human-readable reasons based on heuristics.

        Args:
            text: Input text
            predicted_class: Predicted class (0=Human, 1=AI, 2=Inconclusive)

        Returns:
            List of reason strings
        """
        reasons = []

        # Tokenize into sentences and words
        sentences = nltk.sent_tokenize(text)
        words = nltk.word_tokenize(text)
        words_only = [w.lower() for w in words if w.isalnum()]

        # Check for repetitive structure
        bigrams = [' '.join(words_only[i:i+2]) for i in range(len(words_only)-1)]
        trigrams = [' '.join(words_only[i:i+3]) for i in range(len(words_only)-2)]

        # Count duplicates
        bigram_repetition = len(bigrams) - len(set(bigrams))
        trigram_repetition = len(trigrams) - len(set(trigrams))

        if predicted_class == 1:  # AI-generated
            # Check for low diversity
            if len(words_only) > 0:
                lexical_diversity = len(set(words_only)) / len(words_only)
                if lexical_diversity < 0.5:
                    reasons.append(f"Low lexical diversity ({lexical_diversity:.2f})")

            # Check for repetitive patterns
            if bigram_repetition > 10:
                reasons.append(f"Repetitive word pairs detected ({bigram_repetition} duplicates)")

            if trigram_repetition > 5:
                reasons.append(f"Repetitive phrase patterns ({trigram_repetition} duplicates)")

            # Check for uniform sentence length
            if len(sentences) > 3:
                sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
                avg_len = np.mean(sent_lengths)
                std_len = np.std(sent_lengths)
                if std_len < 3:
                    reasons.append(f"Unnaturally uniform sentence lengths (avg: {avg_len:.1f}±{std_len:.1f})")

            # Check for common AI patterns
            ai_patterns = [
                r'\bin conclusion\b',
                r'\bin summary\b',
                r'\boverall\b',
                r'\bfurthermore\b',
                r'\bmoreover\b',
                r'\badditionally\b'
            ]
            pattern_count = sum(1 for pattern in ai_patterns if re.search(pattern, text.lower()))
            if pattern_count >= 3:
                reasons.append("High frequency of AI-typical transitional phrases")

        elif predicted_class == 0:  # Human-written
            # Check for natural variation
            if len(sentences) > 3:
                sent_lengths = [len(nltk.word_tokenize(s)) for s in sentences]
                std_len = np.std(sent_lengths)
                if std_len > 5:
                    reasons.append(f"Natural variation in sentence structure")

            # Check for high diversity
            if len(words_only) > 0:
                lexical_diversity = len(set(words_only)) / len(words_only)
                if lexical_diversity > 0.7:
                    reasons.append(f"High lexical diversity ({lexical_diversity:.2f})")

        elif predicted_class == 2:  # Inconclusive
            reasons.append("Mixed signals in text characteristics")
            reasons.append("Low confidence in classification")

        # If no specific reasons found, add generic one
        if len(reasons) == 0:
            class_names = ['human-written', 'AI-generated', 'inconclusive']
            reasons.append(f"Text patterns suggest {class_names[predicted_class]} content")

        return reasons

    def explain_prediction(
        self,
        text: str,
        num_top_tokens: int = 10
    ) -> Dict:
        """Generate comprehensive explanation for prediction.

        Args:
            text: Input text
            num_top_tokens: Number of top attributing tokens to return

        Returns:
            Dictionary with explanation components
        """
        # Get prediction
        encoding = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            max_length=512
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            logits = outputs.logits

            # Model outputs 2 classes: [Human, AI]
            probs = torch.softmax(logits, dim=-1)
            predicted_class_2way = torch.argmax(probs, dim=-1).item()
            confidence_2way = probs[0, predicted_class_2way].item()

        # Convert 2-class to 3-class system using confidence threshold
        confidence_threshold = 0.7  # Should match API config

        if confidence_2way < confidence_threshold:
            # Low confidence -> Inconclusive
            predicted_class = 2  # Inconclusive
            confidence = 1.0 - confidence_2way
        else:
            # High confidence -> Human or AI
            predicted_class = predicted_class_2way
            confidence = confidence_2way

        # Get probabilities for all 3 classes
        human_prob = probs[0, 0].item()
        ai_prob = probs[0, 1].item()

        # Calculate inconclusive probability
        max_prob = max(human_prob, ai_prob)
        inconclusive_prob = 1.0 - max_prob if max_prob < confidence_threshold else 0.0

        # Normalize
        total = human_prob + ai_prob + inconclusive_prob
        probabilities_3way = {
            'Human': human_prob / total,
            'AI': ai_prob / total,
            'Inconclusive': inconclusive_prob / total
        }

        # Get token attributions (use 2-way class for attribution)
        # Note: Disabled for pre-trained models due to compatibility issues
        try:
            token_attributions = self.get_token_attributions(text, predicted_class_2way if predicted_class != 2 else 0)
            # Sort by absolute attribution
            token_attributions.sort(key=lambda x: abs(x[1]), reverse=True)
            top_tokens = token_attributions[:num_top_tokens]
        except Exception as e:
            # Fallback: no token attributions
            print(f"Token attribution failed: {e}")
            top_tokens = []

        # Get heuristic reasons
        reasons = self.get_heuristic_reasons(text, predicted_class)

        # Create explanation
        class_names = ['Human', 'AI', 'Inconclusive']
        explanation = {
            'predicted_class': predicted_class,
            'class_name': class_names[predicted_class],
            'confidence': confidence,
            'probabilities': probabilities_3way,
            'top_tokens': top_tokens,
            'reasons': reasons
        }

        return explanation

    def format_explanation(self, explanation: Dict) -> str:
        """Format explanation as human-readable text.

        Args:
            explanation: Explanation dictionary

        Returns:
            Formatted explanation string
        """
        output = []
        output.append("=" * 60)
        output.append("PREDICTION EXPLANATION")
        output.append("=" * 60)
        output.append(f"\nPrediction: {explanation['class_name']}")
        output.append(f"Confidence: {explanation['confidence']:.2%}")

        output.append("\nClass Probabilities:")
        for class_name, prob in explanation['probabilities'].items():
            output.append(f"  {class_name:15s}: {prob:.2%}")

        output.append("\nReasons:")
        for i, reason in enumerate(explanation['reasons'], 1):
            output.append(f"  {i}. {reason}")

        output.append("\nTop Influential Tokens:")
        for token, score in explanation['top_tokens']:
            direction = "→" if score > 0 else "←"
            output.append(f"  {direction} {token:20s} ({score:+.4f})")

        output.append("=" * 60)

        return "\n".join(output)
