"""
Temporal Transformer for anomaly detection (Fixed version with middle-frame masking).

Key fix:
  - The middle frame is replaced by a learnable mask token
  - Transformer must predict the middle latent from surrounding context only
  - Prevents the model from simply copying the middle frame
"""

import torch
import torch.nn as nn
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, latent_dim: int, max_seq_len: int = 64, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_seq_len, latent_dim)
        position = torch.arange(0, max_seq_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, latent_dim, 2).float() * (-np.log(10000.0) / latent_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_seq_len, latent_dim)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerBlock(nn.Module):
    def __init__(self, latent_dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            latent_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(latent_dim)
        self.norm2 = nn.LayerNorm(latent_dim)
        self.ffn = nn.Sequential(
            nn.Linear(latent_dim, latent_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(latent_dim * 4, latent_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)
        return x


class TemporalTransformerModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 256,
        sequence_length: int = 9,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.latent_dim = latent_dim
        self.sequence_length = sequence_length
        self.middle_idx = sequence_length // 2

        # Learnable mask token (replaces the middle frame)
        self.mask_token = nn.Parameter(torch.randn(1, 1, latent_dim) * 0.02)

        self.input_proj = nn.Linear(latent_dim, latent_dim)
        self.pos_encoding = PositionalEncoding(latent_dim, max_seq_len=sequence_length, dropout=dropout)

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(latent_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.prediction_head = nn.Sequential(
            nn.Linear(latent_dim, latent_dim),
            nn.GELU(),
            nn.Linear(latent_dim, latent_dim),
        )

    def _mask_middle(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Replace the middle frame with the learnable mask token.
        This forces the model to predict from context only.
        """
        B, T, D = latent_sequence.shape
        masked = latent_sequence.clone()

        # Expand mask token to batch size and insert at middle position
        mask = self.mask_token.expand(B, -1, -1)          # (B, 1, D)
        masked[:, self.middle_idx:self.middle_idx+1, :] = mask
        return masked

    def forward(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Args:
            latent_sequence: (B, seq_len, latent_dim)

        Returns:
            predicted_middle: (B, latent_dim)
        """
        # 1. Mask the middle frame
        masked_seq = self._mask_middle(latent_sequence)

        # 2. Project + positional encoding
        x = self.input_proj(masked_seq)
        x = self.pos_encoding(x)

        # 3. Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # 4. Take the middle position (which now only contains contextual info)
        middle_features = x[:, self.middle_idx, :]

        # 5. Predict the original middle latent
        predicted_middle = self.prediction_head(middle_features)
        return predicted_middle

    def anomaly_score(self, latent_sequence: torch.Tensor) -> torch.Tensor:
        """
        Score = MSE between predicted middle latent and actual middle latent.
        High score → middle frame does not fit the temporal context → anomaly.
        """
        predicted = self.forward(latent_sequence)                 # (B, D)
        actual = latent_sequence[:, self.middle_idx, :]           # (B, D)
        scores = torch.mean((predicted - actual) ** 2, dim=1)     # (B,)
        return scores


if __name__ == "__main__":
    print("Testing masked TemporalTransformerModel...")
    model = TemporalTransformerModel(latent_dim=256, sequence_length=9, num_layers=2)

    dummy = torch.randn(8, 9, 256)
    pred = model(dummy)
    scores = model.anomaly_score(dummy)

    print(f"Input:     {dummy.shape}")
    print(f"Predicted: {pred.shape}")
    print(f"Scores:    {scores.shape}")
    print(f"Params:    {sum(p.numel() for p in model.parameters()):,}")
    print("OK")