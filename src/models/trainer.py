"""Model training utilities."""
import os
from typing import Dict, Optional

import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
import pandas as pd

from .classifier import FakeNewsClassifier
from ..evaluation.metrics import compute_metrics


class TextDataset(Dataset):
    """Dataset for text classification."""

    def __init__(
        self,
        texts: list,
        labels: list,
        tokenizer,
        max_length: int = 512,
        hybrid_features: Optional[list] = None
    ):
        """Initialize dataset.

        Args:
            texts: List of text samples
            labels: List of labels
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
            hybrid_features: Optional list of feature dictionaries
        """
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.hybrid_features = hybrid_features

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        item = {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

        # Add hybrid features if available
        if self.hybrid_features is not None:
            features = self.hybrid_features[idx]
            # Convert dict to tensor
            feature_values = list(features.values())
            item['hybrid_features'] = torch.tensor(feature_values, dtype=torch.float)

        return item


class ModelTrainer:
    """Trainer for fake news classifier."""

    def __init__(self, config_path: str = "./configs/model_config.yaml"):
        """Initialize trainer with config.

        Args:
            config_path: Path to configuration file
        """
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.model_config = self.config['model']
        self.train_config = self.config['training']
        self.data_config = self.config['data']

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])

        # Initialize model
        self.model = FakeNewsClassifier(
            model_name=self.model_config['name'],
            num_labels=self.model_config['num_labels'],
            use_hybrid_features=self.model_config.get('use_hybrid_features', False)
        ).to(self.device)

    def load_data(self) -> tuple:
        """Load training and validation data.

        Returns:
            Tuple of (train_loader, val_loader)
        """
        # Load CSVs
        train_df = pd.read_csv(self.data_config['train_path'])
        val_df = pd.read_csv(self.data_config['val_path'])

        text_col = self.data_config['text_column']
        label_col = self.data_config['label_column']

        # Create datasets
        train_dataset = TextDataset(
            texts=train_df[text_col].tolist(),
            labels=train_df[label_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        val_dataset = TextDataset(
            texts=val_df[text_col].tolist(),
            labels=val_df[label_col].tolist(),
            tokenizer=self.tokenizer,
            max_length=self.model_config['max_length']
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=2
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=2
        )

        return train_loader, val_loader

    def train(self):
        """Train the model."""
        # Load data
        train_loader, val_loader = self.load_data()

        # Setup optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.train_config['learning_rate'],
            weight_decay=self.train_config['weight_decay']
        )

        total_steps = len(train_loader) * self.train_config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.train_config['warmup_steps'],
            num_training_steps=total_steps
        )

        # Loss function
        criterion = torch.nn.CrossEntropyLoss()

        # Training loop
        best_val_f1 = 0.0
        global_step = 0

        for epoch in range(self.train_config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.train_config['num_epochs']}")

            # Training phase
            self.model.train()
            train_loss = 0.0

            for batch in tqdm(train_loader, desc="Training"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                loss = criterion(logits, labels)

                # Backward pass
                loss.backward()

                # Gradient accumulation
                if (global_step + 1) % self.train_config['gradient_accumulation_steps'] == 0:
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                train_loss += loss.item()
                global_step += 1

                # Logging
                if global_step % self.train_config['logging_steps'] == 0:
                    avg_loss = train_loss / self.train_config['logging_steps']
                    print(f"Step {global_step}, Loss: {avg_loss:.4f}")
                    train_loss = 0.0

                # Evaluation
                if global_step % self.train_config['eval_steps'] == 0:
                    val_metrics = self.evaluate(val_loader)
                    print(f"Validation - F1: {val_metrics['f1_macro']:.4f}, "
                          f"Accuracy: {val_metrics['accuracy']:.4f}")

                    # Save best model
                    if val_metrics['f1_macro'] > best_val_f1:
                        best_val_f1 = val_metrics['f1_macro']
                        self.save_model(self.train_config['output_dir'] + '/best_model')
                        print(f"Saved new best model (F1: {best_val_f1:.4f})")

                    self.model.train()

        print(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")

    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate model on dataloader.

        Args:
            dataloader: DataLoader to evaluate on

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        metrics = compute_metrics(all_labels, all_preds, all_probs)
        return metrics

    def save_model(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)
        torch.save(self.model.state_dict(), f"{path}/model.pt")
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")
