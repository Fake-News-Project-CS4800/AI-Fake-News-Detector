"""Fake news classifier model architecture."""
import torch
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class FakeNewsClassifier(nn.Module):
    """Transformer-based classifier for fake news detection.

    Supports both standard BERT-style classification and hybrid mode
    with additional features (perplexity, stylometric).
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        num_labels: int = 3,
        use_hybrid_features: bool = False,
        num_hybrid_features: int = 6
    ):
        """Initialize classifier.

        Args:
            model_name: Pretrained model name from HuggingFace
            num_labels: Number of classification labels (3: Human, AI, Inconclusive)
            use_hybrid_features: Whether to use additional features
            num_hybrid_features: Number of additional features to concatenate
        """
        super().__init__()

        self.model_name = model_name
        self.num_labels = num_labels
        self.use_hybrid_features = use_hybrid_features
        self.num_hybrid_features = num_hybrid_features

        # Load pretrained transformer
        self.config = AutoConfig.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name, config=self.config)

        # Get hidden size
        self.hidden_size = self.config.hidden_size

        # Classification head
        if use_hybrid_features:
            # Concatenate transformer output with additional features
            classifier_input_size = self.hidden_size + num_hybrid_features
        else:
            classifier_input_size = self.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(0.1),
            nn.Linear(classifier_input_size, self.hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_size // 2, num_labels)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        hybrid_features: torch.Tensor = None
    ):
        """Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            hybrid_features: Additional features [batch_size, num_features]

        Returns:
            Logits [batch_size, num_labels]
        """
        # Get transformer outputs
        outputs = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # Use [CLS] token representation (first token)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]

        # Concatenate with hybrid features if enabled
        if self.use_hybrid_features:
            if hybrid_features is None:
                raise ValueError("hybrid_features required when use_hybrid_features=True")
            combined = torch.cat([cls_output, hybrid_features], dim=1)
        else:
            combined = cls_output

        # Classification
        logits = self.classifier(combined)

        return logits

    def freeze_transformer(self):
        """Freeze transformer weights (only train classification head)."""
        for param in self.transformer.parameters():
            param.requires_grad = False

    def unfreeze_transformer(self):
        """Unfreeze transformer weights."""
        for param in self.transformer.parameters():
            param.requires_grad = True
