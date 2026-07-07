# src/kepler_net/evaluation/evaluate.py

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import precision_recall_curve, average_precision_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Handles out-of-sample evaluation, threshold tuning, and metric extraction."""
    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> dict:
        """
        Runs inference over a validation/test dataset and calculates precise 
        continuous classification metrics.
        """
        all_probs = []
        all_labels = []

        for global_v, local_v, labels in dataloader:
            global_v = global_v.to(self.device)
            local_v = local_v.to(self.device)
            
            # Forward pass through the dual-branch CNN
            logits = self.model(global_v, local_v)
            probs = torch.sigmoid(logits).cpu().numpy()
            
            all_probs.extend(probs)
            all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs).flatten()
        all_labels = np.array(all_labels).flatten()

        # Calculate continuous metrics
        pr_auc = average_precision_score(all_labels, all_probs)
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

        # Optimize for maximum Recall while preserving minimum baseline Precision
        # Find the highest threshold where recall stays above 95%
        high_recall_idx = np.where(recalls >= 0.95)[0]
        if len(high_recall_idx) > 0:
            optimal_idx = high_recall_idx[-1]
            # Handle edge case where threshold array length is len(precisions) - 1
            opt_threshold = thresholds[min(optimal_idx, len(thresholds)-1)]
            corresponding_precision = precisions[optimal_idx]
        else:
            opt_threshold = 0.5
            corresponding_precision = 0.0

        return {
            "pr_auc": pr_auc,
            "optimal_threshold": opt_threshold,
            "precision_at_95_recall": corresponding_precision
        }

if __name__ == "__main__":
    # Diagnostic sanity check with a mock validation pass
    from src.kepler_net.models.cnn_1d import KeplerCNN
    from src.kepler_net.training.trainer import KeplerDataset
    
    processed_dir = Path("data/processed")
    if processed_dir.exists() and list(processed_dir.glob("*.npz")):
        dataset = KeplerDataset(processed_dir)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        
        model = KeplerCNN()
        evaluator = ModelEvaluator(model)
        
        metrics = evaluator.evaluate_dataset(val_loader)
        logger.info(f"Evaluation verified. Baseline PR-AUC: {metrics['pr_auc']:.4f}")
        logger.info(f"Target Classification Threshold for 95% Recall: {metrics['optimal_threshold']:.4f}")