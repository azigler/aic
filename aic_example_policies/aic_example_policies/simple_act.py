"""SimpleACT model definition.

Shared between training (scripts/train_act.py) and inference (ros/OurACT.py).
"""

import torch
import torch.nn as nn


class SimpleACT(nn.Module):
    """Simplified ACT model for training when LeRobot is not installed.

    Architecture:
      - Per-camera CNN encoder (ResNet-18 backbone, pretrained)
      - State MLP encoder
      - Transformer decoder that predicts action chunks
    """

    def __init__(
        self,
        state_dim: int = 26,
        action_dim: int = 7,
        chunk_size: int = 50,
        hidden_dim: int = 512,
        n_heads: int = 8,
        n_layers: int = 4,
        n_cameras: int = 3,
        img_size: int = 256,
        pretrained: bool = True,
    ):
        super().__init__()
        self.chunk_size = chunk_size
        self.action_dim = action_dim

        # Vision encoder: ResNet-18 per camera (shared weights)
        import torchvision.models as models

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        try:
            resnet = models.resnet18(weights=weights)
        except Exception:
            # Fallback: no pretrained weights (inference loads full checkpoint anyway)
            resnet = models.resnet18(weights=None)
        # Remove final FC, keep up to avgpool -> 512D feature
        self.vision_backbone = nn.Sequential(*list(resnet.children())[:-1])
        vision_feat_dim = 512

        # Freeze early layers, fine-tune later layers
        for param in list(self.vision_backbone.parameters())[:-20]:
            param.requires_grad = False

        # Project concatenated camera features
        self.vision_proj = nn.Linear(vision_feat_dim * n_cameras, hidden_dim)

        # State encoder
        self.state_proj = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # Combine vision + state
        self.combine = nn.Linear(hidden_dim * 2, hidden_dim)

        # Transformer decoder for action chunking
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            batch_first=True,
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=n_layers
        )

        # Learnable action queries (one per chunk step)
        self.action_queries = nn.Parameter(
            torch.randn(1, chunk_size, hidden_dim) * 0.02
        )

        # Action head
        self.action_head = nn.Linear(hidden_dim, action_dim)

    def forward(
        self,
        img_left: torch.Tensor,
        img_center: torch.Tensor,
        img_right: torch.Tensor,
        state: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            img_left, img_center, img_right: (B, 3, H, W) camera images
            state: (B, state_dim) robot state

        Returns:
            actions: (B, chunk_size, action_dim) predicted action chunk
        """
        batch_size = img_left.shape[0]

        # Encode each camera image
        feat_left = self.vision_backbone(img_left).flatten(1)  # (B, 512)
        feat_center = self.vision_backbone(img_center).flatten(1)
        feat_right = self.vision_backbone(img_right).flatten(1)

        # Concatenate and project
        vision_feat = self.vision_proj(
            torch.cat([feat_left, feat_center, feat_right], dim=1)
        )  # (B, hidden)

        # Encode state
        state_feat = self.state_proj(state)  # (B, hidden)

        # Combine
        context = self.combine(
            torch.cat([vision_feat, state_feat], dim=1)
        )  # (B, hidden)

        # Expand context as memory for transformer decoder: (B, 1, hidden)
        memory = context.unsqueeze(1)

        # Action queries: (B, chunk_size, hidden)
        queries = self.action_queries.expand(batch_size, -1, -1)

        # Decode
        decoded = self.transformer_decoder(
            queries, memory
        )  # (B, chunk, hidden)

        # Predict actions
        actions = self.action_head(decoded)  # (B, chunk_size, action_dim)
        return actions
