"""Text preprocessing and normalization utilities."""
import hashlib
import re
import unicodedata
from typing import Dict, List, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split


def compute_hash(text: str) -> str:
    """Compute SHA-256 hash of normalized text.

    Args:
        text: Input text to hash

    Returns:
        Hexadecimal hash string
    """
    # Normalize before hashing
    normalized = normalize_text(text)
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()


def normalize_text(text: str) -> str:
    """Normalize text for consistent processing.

    Args:
        text: Input text

    Returns:
        Normalized text
    """
    # Unicode normalization
    text = unicodedata.normalize('NFKC', text)

    # Convert to lowercase
    text = text.lower()

    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


class TextPreprocessor:
    """Preprocessor for cleaning and preparing text data."""

    def __init__(self, max_length: int = 512):
        """Initialize preprocessor.

        Args:
            max_length: Maximum sequence length
        """
        self.max_length = max_length
        self.seen_hashes = set()

    def clean_text(self, text: str) -> str:
        """Clean and normalize text.

        Args:
            text: Input text

        Returns:
            Cleaned text
        """
        # Normalize
        text = normalize_text(text)

        # Remove URLs
        text = re.sub(r'http\S+|www.\S+', '', text)

        # Remove email addresses
        text = re.sub(r'\S+@\S+', '', text)

        # Remove excessive punctuation
        text = re.sub(r'([!?.]){2,}', r'\1', text)

        # Final cleanup
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def is_duplicate(self, text: str) -> bool:
        """Check if text is a duplicate based on hash.

        Args:
            text: Input text

        Returns:
            True if duplicate, False otherwise
        """
        text_hash = compute_hash(text)
        if text_hash in self.seen_hashes:
            return True
        self.seen_hashes.add(text_hash)
        return False

    def preprocess_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = 'text',
        label_column: str = 'label'
    ) -> pd.DataFrame:
        """Preprocess entire dataset.

        Args:
            df: Input dataframe
            text_column: Name of text column
            label_column: Name of label column

        Returns:
            Preprocessed dataframe
        """
        # Reset seen hashes
        self.seen_hashes = set()

        # Clean text
        df['cleaned_text'] = df[text_column].apply(self.clean_text)

        # Compute hashes
        df['text_hash'] = df['cleaned_text'].apply(compute_hash)

        # Remove duplicates
        df = df[~df['cleaned_text'].apply(self.is_duplicate)]

        # Filter out empty texts
        df = df[df['cleaned_text'].str.len() > 0]

        # Reset index
        df = df.reset_index(drop=True)

        return df

    def create_splits(
        self,
        df: pd.DataFrame,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        stratify_column: str = 'label',
        random_state: int = 42
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Split data into train/validation/test sets.

        Args:
            df: Input dataframe
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            stratify_column: Column to stratify on
            random_state: Random seed

        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"

        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            test_size=(1 - train_ratio),
            stratify=df[stratify_column] if stratify_column else None,
            random_state=random_state
        )

        # Second split: val vs test
        val_ratio_adjusted = val_ratio / (val_ratio + test_ratio)
        val_df, test_df = train_test_split(
            temp_df,
            test_size=(1 - val_ratio_adjusted),
            stratify=temp_df[stratify_column] if stratify_column else None,
            random_state=random_state
        )

        return train_df, val_df, test_df

    def save_splits(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        output_dir: str = './data/splits'
    ):
        """Save data splits to CSV files.

        Args:
            train_df: Training dataframe
            val_df: Validation dataframe
            test_df: Test dataframe
            output_dir: Output directory
        """
        train_df.to_csv(f'{output_dir}/train.csv', index=False)
        val_df.to_csv(f'{output_dir}/val.csv', index=False)
        test_df.to_csv(f'{output_dir}/test.csv', index=False)

        print(f"Saved splits to {output_dir}")
        print(f"Train: {len(train_df)} samples")
        print(f"Val: {len(val_df)} samples")
        print(f"Test: {len(test_df)} samples")
