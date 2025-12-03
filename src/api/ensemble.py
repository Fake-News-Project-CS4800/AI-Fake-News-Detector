"""Ensemble logic for combining multiple AI detection models."""
from typing import Dict, List, Tuple


class EnsembleDetector:
    """Combines predictions from multiple models for robust detection."""

    def __init__(self, agreement_threshold: float = 0.8):
        """Initialize ensemble detector.

        Args:
            agreement_threshold: Minimum confidence for high-confidence agreement
        """
        self.agreement_threshold = agreement_threshold

    def combine_predictions(
        self,
        roberta_prediction: Dict,
        gemini_prediction: Dict
    ) -> Dict:
        """Combine predictions from RoBERTa and Gemini models.

        Args:
            roberta_prediction: Dict with 'label', 'confidence', 'probabilities'
            gemini_prediction: Dict with 'label', 'confidence', 'reasoning'

        Returns:
            Dict with ensemble prediction and metadata
        """
        roberta_label = roberta_prediction['label']
        roberta_conf = roberta_prediction['confidence']

        gemini_label = gemini_prediction['label']
        gemini_conf = gemini_prediction['confidence']

        # Calculate agreement
        models_agree = roberta_label == gemini_label
        both_confident = (
            roberta_conf >= self.agreement_threshold and
            gemini_conf >= self.agreement_threshold
        )

        # Decision logic
        if models_agree:
            if both_confident:
                # Strong agreement - use agreed label with high confidence
                final_label = roberta_label
                final_confidence = (roberta_conf + gemini_conf) / 2
                agreement_level = 'Strong Agreement'
            else:
                # Weak agreement - use agreed label with moderate confidence
                final_label = roberta_label
                final_confidence = min(roberta_conf, gemini_conf)
                agreement_level = 'Weak Agreement'
        else:
            # Disagreement - check which model is more confident
            if abs(roberta_conf - gemini_conf) > 0.3:
                # Significant confidence difference - trust more confident model
                if roberta_conf > gemini_conf:
                    final_label = roberta_label
                    final_confidence = roberta_conf * 0.8  # Reduce confidence due to disagreement
                else:
                    final_label = gemini_label
                    final_confidence = gemini_conf * 0.8
                agreement_level = 'Disagreement - Confidence-Based'
            else:
                # Similar confidence but different labels - mark as inconclusive
                final_label = 'Inconclusive'
                final_confidence = 1.0 - max(roberta_conf, gemini_conf)
                agreement_level = 'Disagreement - Inconclusive'

        # Build ensemble result
        return {
            'label': final_label,
            'confidence': final_confidence,
            'agreement_level': agreement_level,
            'models_agree': models_agree,
            'roberta': {
                'label': roberta_label,
                'confidence': roberta_conf,
                'probabilities': roberta_prediction.get('probabilities', {})
            },
            'gemini': {
                'label': gemini_label,
                'confidence': gemini_conf,
                'reasoning': gemini_prediction.get('reasoning', '')
            }
        }

    def get_ensemble_reasons(
        self,
        ensemble_result: Dict,
        roberta_reasons: List[str]
    ) -> List[str]:
        """Generate explanation reasons for ensemble prediction.

        Args:
            ensemble_result: Result from combine_predictions
            roberta_reasons: Original reasons from RoBERTa model

        Returns:
            List of explanation strings
        """
        reasons = []

        # Add agreement status
        if ensemble_result['models_agree']:
            reasons.append(
                f"✓ Both models agree: {ensemble_result['label']} "
                f"({ensemble_result['agreement_level']})"
            )
        else:
            reasons.append(
                f"⚠ Models disagree: RoBERTa says {ensemble_result['roberta']['label']}, "
                f"Gemini says {ensemble_result['gemini']['label']}"
            )

        # Add model-specific insights
        reasons.append(
            f"RoBERTa confidence: {ensemble_result['roberta']['confidence']:.2%}"
        )
        reasons.append(
            f"Gemini confidence: {ensemble_result['gemini']['confidence']:.2%}"
        )

        # Add Gemini reasoning if available
        if ensemble_result['gemini']['reasoning']:
            reasons.append(f"Gemini analysis: {ensemble_result['gemini']['reasoning']}")

        # Add original RoBERTa reasons
        reasons.extend(roberta_reasons)

        return reasons
