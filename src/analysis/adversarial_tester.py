"""Adversarial testing module for evaluating model robustness."""
import random
import re
from typing import Dict, List, Callable
import numpy as np


class AdversarialTester:
    """Tests model robustness against adversarial text perturbations."""

    def __init__(self, predict_fn: Callable):
        """
        Initialize adversarial tester.

        Args:
            predict_fn: Function that takes text and returns prediction dict
                       with keys: label, confidence, probabilities
        """
        self.predict_fn = predict_fn

        # Common realistic typo patterns
        self.typo_patterns = [
            ('the', 'teh'),
            ('and', 'adn'),
            ('you', 'yuo'),
            ('that', 'taht'),
            ('with', 'wiht'),
            ('from', 'form'),
            ('about', 'abotu'),
            ('would', 'woudl'),
            ('people', 'peopel'),
            ('really', 'realy'),
        ]

        # Filler words people naturally add
        self.filler_words = [
            'actually', 'basically', 'honestly', 'literally',
            'like', 'you know', 'I mean', 'sort of', 'kind of',
            'um', 'uh', 'well'
        ]

        # Contractions and their expansions
        self.contractions = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "shouldn't": "should not",
            "wouldn't": "would not",
            "I'm": "I am",
            "you're": "you are",
            "it's": "it is",
            "that's": "that is",
            "isn't": "is not",
        }

    def run_adversarial_tests(self, text: str, max_tests: int = 20) -> Dict:
        """
        Run comprehensive adversarial testing on text.

        Args:
            text: Original text to test
            max_tests: Maximum number of adversarial examples to generate

        Returns:
            Dictionary with test results and robustness metrics
        """
        # Get baseline prediction
        baseline = self.predict_fn(text)

        # Generate different types of adversarial examples (REALISTIC ONLY)
        adversarial_examples = []

        # 1. Natural typos
        typo_examples = self._generate_typo_perturbations(text)
        adversarial_examples.extend(typo_examples[:5])

        # 2. Punctuation variations (realistic)
        punct_examples = self._generate_punctuation_perturbations(text)
        adversarial_examples.extend(punct_examples[:4])

        # 3. Extra whitespace
        whitespace_examples = self._generate_whitespace_perturbations(text)
        adversarial_examples.extend(whitespace_examples[:2])

        # 4. Filler word insertions
        filler_examples = self._generate_filler_insertions(text)
        adversarial_examples.extend(filler_examples[:3])

        # 5. Contractions/expansions
        contraction_examples = self._generate_contraction_changes(text)
        adversarial_examples.extend(contraction_examples[:3])

        # 6. Capitalization (realistic only)
        case_examples = self._generate_realistic_case_changes(text)
        adversarial_examples.extend(case_examples[:2])

        # 7. Minor word changes (realistic)
        word_examples = self._generate_realistic_word_changes(text)
        adversarial_examples.extend(word_examples[:1])

        # Limit total examples
        adversarial_examples = adversarial_examples[:max_tests]

        # Test each adversarial example
        results = []
        label_flips = 0
        confidence_changes = []

        for adv_text, attack_type in adversarial_examples:
            try:
                prediction = self.predict_fn(adv_text)

                # Check if label flipped
                label_flipped = prediction['label'] != baseline['label']
                if label_flipped:
                    label_flips += 1

                # Calculate confidence change
                confidence_change = abs(prediction['confidence'] - baseline['confidence'])
                confidence_changes.append(confidence_change)

                results.append({
                    'text': adv_text,
                    'attack_type': attack_type,
                    'label': prediction['label'],
                    'confidence': prediction['confidence'],
                    'label_flipped': label_flipped,
                    'confidence_change': confidence_change,
                    'probabilities': prediction['probabilities']
                })
            except Exception as e:
                print(f"Warning: Adversarial test failed for {attack_type}: {e}")
                continue

        # Calculate robustness metrics
        total_tests = len(results)
        robustness_score = 1.0 - (label_flips / total_tests) if total_tests > 0 else 0.0
        avg_confidence_change = np.mean(confidence_changes) if confidence_changes else 0.0
        max_confidence_change = np.max(confidence_changes) if confidence_changes else 0.0

        # Stability assessment
        if robustness_score >= 0.9:
            stability = 'Very Stable'
        elif robustness_score >= 0.7:
            stability = 'Stable'
        elif robustness_score >= 0.5:
            stability = 'Moderately Stable'
        else:
            stability = 'Unstable'

        return {
            'baseline': baseline,
            'adversarial_examples': results,
            'summary': {
                'total_tests': total_tests,
                'label_flips': label_flips,
                'flip_rate': label_flips / total_tests if total_tests > 0 else 0.0,
                'robustness_score': robustness_score,
                'avg_confidence_change': avg_confidence_change,
                'max_confidence_change': max_confidence_change,
                'stability': stability
            }
        }

    def _generate_realistic_case_changes(self, text: str) -> List[tuple]:
        """Generate realistic case changes."""
        examples = []

        # All lowercase (people write like this casually)
        examples.append((text.lower(), 'Capitalization: All lowercase'))

        # First letter lowercase (missing capitalization)
        if text and text[0].isupper():
            lowercase_start = text[0].lower() + text[1:]
            examples.append((lowercase_start, 'Capitalization: Missing first capital'))

        return examples

    def _generate_punctuation_perturbations(self, text: str) -> List[tuple]:
        """Generate realistic punctuation perturbations."""
        examples = []

        # Add extra commas (people do this for emphasis)
        if ',' in text:
            extra_commas = text.replace(',', ',,', 1)
            examples.append((extra_commas, 'Punctuation: Extra comma'))

        # Add extra periods (people do this for effect)
        if '.' in text:
            extra_periods = text.replace('.', '...', 1)
            examples.append((extra_periods, 'Punctuation: Ellipsis'))

        # Add exclamation mark at end
        if text.endswith('.'):
            with_exclamation = text[:-1] + '!'
            examples.append((with_exclamation, 'Punctuation: Exclamation added'))

        # Missing space after comma (common typo)
        space_removed = re.sub(r',\s+', ',', text, count=1)
        if space_removed != text:
            examples.append((space_removed, 'Punctuation: Missing space after comma'))

        # Missing period at end
        if text.endswith('.'):
            no_period = text[:-1]
            examples.append((no_period, 'Punctuation: Missing final period'))

        return examples

    def _generate_typo_perturbations(self, text: str) -> List[tuple]:
        """Generate realistic typo perturbations."""
        examples = []

        # Apply common typo patterns
        for original, typo in self.typo_patterns:
            # Try both cases
            if original in text.lower():
                # Find the word with proper case
                words = text.split()
                for i, word in enumerate(words):
                    if word.lower() == original:
                        words_copy = words.copy()
                        # Preserve case
                        if word[0].isupper():
                            words_copy[i] = typo.capitalize()
                        else:
                            words_copy[i] = typo
                        typo_text = ' '.join(words_copy)
                        examples.append((typo_text, f'Typo: {original}→{typo}'))
                        break

        # Doubled letter (common typo)
        words = text.split()
        if len(words) > 3:
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            if len(word) > 3:
                char_idx = random.randint(1, len(word) - 1)
                typo_word = word[:char_idx] + word[char_idx] + word[char_idx:]
                words[word_idx] = typo_word
                examples.append((' '.join(words), f'Typo: Doubled letter in "{word}"'))

        # Missing letter (common typo)
        words = text.split()
        if len(words) > 3:
            word_idx = random.randint(0, len(words) - 1)
            word = words[word_idx]
            if len(word) > 4:
                char_idx = random.randint(1, len(word) - 2)
                typo_word = word[:char_idx] + word[char_idx+1:]
                words[word_idx] = typo_word
                examples.append((' '.join(words), f'Typo: Missing letter in "{word}"'))

        return examples

    def _generate_filler_insertions(self, text: str) -> List[tuple]:
        """Generate filler word insertions (realistic)."""
        examples = []

        sentences = text.split('. ')
        if len(sentences) > 0:
            # Add filler at the beginning
            filler = random.choice(self.filler_words)
            modified = filler.capitalize() + ', ' + text
            examples.append((modified, f'Filler: Added "{filler}" at start'))

            # Add filler in the middle
            if len(sentences) > 1:
                filler = random.choice(self.filler_words)
                sentences_copy = sentences.copy()
                sentences_copy[0] = sentences_copy[0] + ', ' + filler + ','
                modified = '. '.join(sentences_copy)
                examples.append((modified, f'Filler: Inserted "{filler}"'))

        return examples

    def _generate_contraction_changes(self, text: str) -> List[tuple]:
        """Generate contraction/expansion changes."""
        examples = []

        # Contract expanded forms
        for contraction, expanded in self.contractions.items():
            if expanded.lower() in text.lower():
                modified = re.sub(expanded, contraction, text, count=1, flags=re.IGNORECASE)
                if modified != text:
                    examples.append((modified, f'Contraction: {expanded}→{contraction}'))

        # Expand contractions
        for contraction, expanded in self.contractions.items():
            if contraction.lower() in text.lower():
                modified = re.sub(contraction, expanded, text, count=1, flags=re.IGNORECASE)
                if modified != text:
                    examples.append((modified, f'Expansion: {contraction}→{expanded}'))

        return examples

    def _generate_realistic_word_changes(self, text: str) -> List[tuple]:
        """Generate realistic word-level changes."""
        examples = []

        # Add/remove articles
        words = text.split()
        if len(words) > 5:
            # Try adding "the"
            for i in range(1, len(words) - 1):
                if words[i][0].islower() and words[i-1].lower() not in ['a', 'an', 'the']:
                    words_copy = words.copy()
                    words_copy.insert(i, 'the')
                    modified = ' '.join(words_copy)
                    examples.append((modified, 'Article: Added "the"'))
                    break

        return examples

    def _generate_whitespace_perturbations(self, text: str) -> List[tuple]:
        """Generate realistic whitespace perturbations."""
        examples = []

        # Extra spaces between words (typo)
        words = text.split()
        if len(words) > 2:
            idx = random.randint(1, len(words) - 1)
            words_copy = words.copy()
            words_copy[idx] = '  ' + words_copy[idx]
            modified = ' '.join(words_copy)
            examples.append((modified, 'Whitespace: Extra space'))

        # Missing space after punctuation (common typo)
        if '.' in text or ',' in text:
            no_space = re.sub(r'([.,])(\s+)(\w)', r'\1\3', text, count=1)
            if no_space != text:
                examples.append((no_space, 'Whitespace: Missing space after punctuation'))

        return examples
