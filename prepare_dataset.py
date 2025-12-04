"""Download and prepare the AI-human-text dataset for training."""
import os
import pandas as pd
from datasets import load_dataset
from src.data.preprocessor import TextPreprocessor

def prepare_ai_human_dataset():
    """Download and prepare the AI-human-text dataset."""
    print("Downloading AI-human-text dataset...")
    
    # Download dataset from HuggingFace
    dataset = load_dataset("andythetechnerd03/AI-human-text")
    
    # Convert to pandas DataFrame
    df = dataset['train'].to_pandas()
    
    print(f"Dataset shape: {df.shape}")

    df = df.sample(n=10000, random_state=104)
    print(f"Reduced dataset shape: {df.shape}")

    print(f"Columns: {df.columns.tolist()}")
    print("\nFirst few rows:")
    print(df.head())
    
    # The dataset has "generated" column with 0=Human, 1=AI
    if 'generated' in df.columns:
        # Rename 'generated' to 'label' and keep the values as is (0=Human, 1=AI)
        df['label'] = df['generated'].astype(int)
    else:
        # Print available columns to help debug
        print(f"Available columns: {df.columns.tolist()}")
        raise ValueError("Could not find 'generated' column. Please check the dataset structure.")
    
    
    # Remove rows with unmapped labels
    df = df.dropna(subset=['label'])
    df['label'] = df['label'].astype(int)
    
    print(f"Label distribution:\n{df['label'].value_counts()}")
    
    # Create data directory if it doesn't exist
    os.makedirs('./data/splits', exist_ok=True)
    
    # Preprocess the dataset
    preprocessor = TextPreprocessor()
    df_clean = preprocessor.preprocess_dataset(df, text_column='text', label_column='label')
    
    # Create train/val/test splits
    train_df, val_df, test_df = preprocessor.create_splits(
        df_clean, 
        train_ratio=0.8, 
        val_ratio=0.1, 
        test_ratio=0.1
    )
    
    # Save splits
    preprocessor.save_splits(train_df, val_df, test_df, './data/splits')
    
    print("\nDataset preparation complete!")
    print(f"Train: {len(train_df)} samples")
    print(f"Val: {len(val_df)} samples") 
    print(f"Test: {len(test_df)} samples")
    
    return train_df, val_df, test_df

if __name__ == "__main__":
    prepare_ai_human_dataset()