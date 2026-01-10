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

        # Ensure numeric values are properly converted from YAML
        self.train_config['learning_rate'] = float(self.train_config['learning_rate'])
        self.train_config['weight_decay'] = float(self.train_config['weight_decay'])
        self.train_config['batch_size'] = int(self.train_config['batch_size'])
        self.train_config['num_epochs'] = int(self.train_config['num_epochs'])

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_config['name'])

		# Initialize model - check if loading pre-trained or training from scratch
        model_name = self.model_config['name']
        
        if "chatgpt-detector-roberta" in model_name or "roberta" in model_name:
            # Load the pre-trained model directly for fine-tuning
            print(f"Loading pre-trained model: {model_name}")
            from transformers import AutoModelForSequenceClassification
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                num_labels=self.model_config['num_labels']
            ).to(self.device)
        else:
            # Use custom classifier for other models
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
            num_workers=0  # Changed from 2 to 0 to avoid fork issues
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=0  # Changed from 2 to 0 to avoid fork issues
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

		# Estimate training time
        estimated_time_per_batch = 20  # seconds (conservative estimate for CPU)
        estimated_total_time = len(train_loader) * self.train_config['num_epochs'] * estimated_time_per_batch / 60
        print(f"Estimated training time: {estimated_total_time:.1f} minutes")

        for epoch in range(self.train_config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.train_config['num_epochs']}")

            # Training phase
            self.model.train()
            train_loss = 0.0

            # Add time tracking
            import time
            epoch_start_time = time.time()

            for batch_idx, batch in enumerate(tqdm(train_loader, desc="Training")):
                batch_start_time = time.time()

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                # Forward pass
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

				# Extract logits from output (handle both custom and pre-trained models)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

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

        # Save final model at the end of training (regardless of validation)
        final_model_path = self.train_config['output_dir'] + '/final_model'
        self.save_model(final_model_path)
        print(f"✅ Final model saved to: {final_model_path}")
        
        print(f"\nTraining complete! Best validation F1: {best_val_f1:.4f}")
        print(f"Models saved:")
        print(f"  - Best model: {self.train_config['output_dir']}/best_model")
        print(f"  - Final model: {final_model_path}")

    def evaluate(self, dataloader: DataLoader, max_batches: int = 50) -> Dict[str, float]:
        """Evaluate model on dataloader.

        Args:
            dataloader: DataLoader to evaluate on
            max_batches: Maximum number of batches to evaluate (for speed)

        Returns:
            Dictionary of metrics
        """
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        batch_count = 0
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                # Limit validation for speed
                if batch_count >= max_batches:
                    break
                    
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )

                # Extract logits from output (handle both custom and pre-trained models)
                if hasattr(outputs, 'logits'):
                    logits = outputs.logits
                else:
                    logits = outputs

                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                
                batch_count += 1
                metrics = compute_metrics(all_labels, all_preds, all_probs)
        return metrics

    def save_model(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save model
        """
        os.makedirs(path, exist_ok=True)
		# For pre-trained models, use the built-in save methods
        if hasattr(self.model, 'save_pretrained'):
            self.model.save_pretrained(path)
            self.tokenizer.save_pretrained(path)
            print(f"Pre-trained model saved to {path}")
        else:
            # For custom models, use state dict
            torch.save(self.model.state_dict(), f"{path}/model.pt")
            self.tokenizer.save_pretrained(path)
            print(f"Custom model saved to {path}")

        # torch.save(self.model.state_dict(), f"{path}/model.pt")
        # self.tokenizer.save_pretrained(path)
        # print(f"Model saved to {path}")

    def load_model(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to load model from
        """
        self.model.load_state_dict(torch.load(f"{path}/model.pt", map_location=self.device))
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        print(f"Model loaded from {path}")
