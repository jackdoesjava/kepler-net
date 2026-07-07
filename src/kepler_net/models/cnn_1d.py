# src/kepler_net/models/cnn_1d.py

import torch
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)

class KeplerCNN(nn.Module):
    """
    Dual-branch 1D Convolutional Neural Network for exoplanet transit detection.
    Processes a 'global' macro-view and a 'local' micro-view simultaneously.
    """
    def __init__(self, global_length: int = 2001, local_length: int = 201):
        super().__init__()
        self.global_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            nn.Flatten()
        )
        
        self.local_branch = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=5, stride=2),
            
            nn.Flatten()
        )
        
        self._calculate_fc_input_dims(global_length, local_length)
        
        self.fc_head = nn.Sequential(
            nn.Linear(self.fc_input_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3), 
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 1) 
        )

    def _calculate_fc_input_dims(self, global_length: int, local_length: int):
        """Passes dummy tensors through the branches to dynamically calculate the dense layer input size."""
        dummy_global = torch.zeros(1, 1, global_length)
        dummy_local = torch.zeros(1, 1, local_length)
        
        global_out = self.global_branch(dummy_global).shape[1]
        local_out = self.local_branch(dummy_local).shape[1]
        
        self.fc_input_dim = global_out + local_out

    def forward(self, global_view: torch.Tensor, local_view: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        Expects inputs of shape: (batch_size, channels=1, sequence_length)
        """
        x_global = self.global_branch(global_view)
        x_local = self.local_branch(local_view)
        
        x_fused = torch.cat((x_global, x_local), dim=1)
        
        logits = self.fc_head(x_fused)
        
        return logits

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    model = KeplerCNN(global_length=2001, local_length=201)
    
    dummy_global_batch = torch.randn(64, 1, 2001)
    dummy_local_batch = torch.randn(64, 1, 201)
    
    output = model(dummy_global_batch, dummy_local_batch)
    
    logger.info(f"Model initialized successfully. FC Input Dimension: {model.fc_input_dim}")
    logger.info(f"Output Shape: {output.shape} -> (batch_size, 1 prediction logit)")