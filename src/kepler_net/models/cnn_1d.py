# src/kepler_net/models/cnn_1d.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ConvBlock(nn.Module):
    """A stabilized convolutional block with Dilation for expanded receptive fields."""

    def __init__(self, in_channels, out_channels, kernel_size, pool_size, dilation=1):
        super().__init__()

        # Calculate dynamic padding to keep sequence dimensions clean during dilation
        padding = dilation * (kernel_size - 1) // 2

        self.block = nn.Sequential(
            nn.Conv1d(
                in_channels,
                out_channels,
                kernel_size,
                padding=padding,
                dilation=dilation,
            ),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(pool_size),
        )

    def forward(self, x):
        return self.block(x)


class KeplerCNN(nn.Module):
    """
    Dilated 1D CNN for time-series.
    Exponentially expands the receptive field to capture long-range transit geometry.
    """

    def __init__(self, global_length: int = 2001, local_length: int = 201):
        super().__init__()

        # Global Branch: Dilation scales exponentially (1, 2, 4) to capture the wide U-shape
        self.global_branch = nn.Sequential(
            ConvBlock(1, 16, kernel_size=5, pool_size=2, dilation=1),
            ConvBlock(16, 32, kernel_size=5, pool_size=2, dilation=2),
            ConvBlock(32, 64, kernel_size=5, pool_size=2, dilation=4),
            nn.AdaptiveMaxPool1d(
                4
            ),  # Squeeze to 4 time-steps to prevent noise memorization
        )

        # Local Branch: Keeps tight dilation to focus on the micro-structure of the dip
        self.local_branch = nn.Sequential(
            ConvBlock(1, 16, kernel_size=3, pool_size=2, dilation=1),
            ConvBlock(16, 32, kernel_size=3, pool_size=2, dilation=2),
            nn.AdaptiveMaxPool1d(4),
        )

        self.fc_input_dim = (64 * 4) + (32 * 4)

        # Classifier
        self.fc_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.fc_input_dim, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, a=0.2, nonlinearity="leaky_relu")
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(
        self, global_view: torch.Tensor, local_view: torch.Tensor
    ) -> torch.Tensor:
        g = self.global_branch(global_view)
        local_features = self.local_branch(local_view)

        x_fused = torch.cat((g, local_features), dim=1)
        logits = self.fc_head(x_fused)

        return logits


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    model = KeplerCNN()

    dummy_global = torch.randn(64, 1, 2001)
    dummy_local = torch.randn(64, 1, 201)
    out = model(dummy_global, dummy_local)

    logger.info(
        f"Model initialized successfully. FC Input Dimension: {model.fc_input_dim}"
    )
    logger.info(f"Output Shape: {out.shape} -> (batch_size, 1 prediction logit)")
