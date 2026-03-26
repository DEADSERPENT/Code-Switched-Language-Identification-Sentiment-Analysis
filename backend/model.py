"""
CoSwitchModel: Multi-task model for code-switched language identification
and sentiment analysis. Uses XLM-RoBERTa-base as shared encoder with two
classification heads.
"""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaConfig

# Label mappings — keep in sync with dataset.py and app.py
LID_LABELS = ["lang1", "lang2", "mixed", "ne", "other", "univ"]
SENTIMENT_LABELS = ["negative", "neutral", "positive"]

LID_LABEL2ID = {l: i for i, l in enumerate(LID_LABELS)}
LID_ID2LABEL = {i: l for i, l in enumerate(LID_LABELS)}

SENTIMENT_LABEL2ID = {l: i for i, l in enumerate(SENTIMENT_LABELS)}
SENTIMENT_ID2LABEL = {i: l for i, l in enumerate(SENTIMENT_LABELS)}


class CoSwitchModel(nn.Module):
    """
    Dual-head model:
      - Task A: Token-level Language Identification (sequence labeling)
      - Task B: Sentence-level Sentiment Classification (CLS token)

    Both tasks share the XLM-RoBERTa encoder, enabling cross-task transfer.
    """

    def __init__(
        self,
        model_name: str = "FacebookAI/xlm-roberta-base",
        num_lid_labels: int = len(LID_LABELS),
        num_sentiment_labels: int = len(SENTIMENT_LABELS),
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = XLMRobertaModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size  # 768

        # Task A: token-level LID head
        self.lid_dropout = nn.Dropout(dropout)
        self.lid_classifier = nn.Linear(hidden, num_lid_labels)

        # Task B: sentence-level sentiment head (slightly deeper for richer features)
        self.sentiment_dropout = nn.Dropout(dropout)
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(hidden, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_sentiment_labels),
        )

        self.num_lid_labels = num_lid_labels
        self.num_sentiment_labels = num_sentiment_labels

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            lid_logits:       (batch, seq_len, num_lid_labels)
            sentiment_logits: (batch, num_sentiment_labels)
        """
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        sequence_output = outputs.last_hidden_state  # (B, L, 768)
        cls_output = sequence_output[:, 0, :]        # (B, 768)

        lid_logits = self.lid_classifier(self.lid_dropout(sequence_output))
        sentiment_logits = self.sentiment_classifier(self.sentiment_dropout(cls_output))

        return lid_logits, sentiment_logits

    def save(self, directory: str) -> None:
        import os, json
        os.makedirs(directory, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(directory, "model.pt"))
        config = {
            "base_model": self.encoder.config.name_or_path,
            "num_lid_labels": self.num_lid_labels,
            "num_sentiment_labels": self.num_sentiment_labels,
            "lid_labels": LID_LABELS,
            "sentiment_labels": SENTIMENT_LABELS,
        }
        with open(os.path.join(directory, "config.json"), "w") as f:
            json.dump(config, f, indent=2)
        print(f"Model saved to {directory}")

    @classmethod
    def load(cls, directory: str, device: str = "cpu") -> "CoSwitchModel":
        import os, json
        with open(os.path.join(directory, "config.json")) as f:
            config = json.load(f)
        model = cls(
            model_name=config["base_model"],
            num_lid_labels=config["num_lid_labels"],
            num_sentiment_labels=config["num_sentiment_labels"],
        )
        state = torch.load(
            os.path.join(directory, "model.pt"),
            map_location=device,
            weights_only=True,
        )
        model.load_state_dict(state)
        model.to(device)
        model.eval()
        return model
