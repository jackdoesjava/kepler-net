# src/kepler_net/evaluation/evaluate.py

import logging
import os
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class Visualizer:
    """Generates publication-quality diagnostic plots."""

    @staticmethod
    def plot_pr_curve(y_true, y_probs, pr_auc):
        precisions, recalls, _ = precision_recall_curve(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, linewidth=2, label=f"PR-AUC = {pr_auc:.3f}")
        plt.fill_between(recalls, precisions, alpha=0.2)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.7)
        plt.savefig("outputs/pr_curve.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_confusion_matrix(y_true, y_pred_binary):
        cm = confusion_matrix(y_true, y_pred_binary)
        plt.figure(figsize=(6, 5))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Binary", "Planet"],
            yticklabels=["Binary", "Planet"],
        )
        plt.ylabel("Actual")
        plt.xlabel("Predicted")
        plt.title("Confusion Matrix")
        plt.savefig("outputs/confusion_matrix.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_score_distribution(y_true, y_probs):
        plt.figure(figsize=(8, 5))
        sns.histplot(x=y_probs, hue=y_true, kde=True, bins=20)
        plt.title("Prediction Confidence Distribution")
        plt.xlabel("Model Probability Score")
        plt.savefig("outputs/confidence_dist.png", dpi=300)
        plt.close()

    @staticmethod
    def plot_roc_curve(y_true, y_probs):
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
        plt.plot([0, 1], [0, 1], linestyle="--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.savefig("outputs/roc_curve.png", dpi=300)
        plt.close()


class ModelEvaluator:
    """Handles inference and diagnostic visualization."""

    def __init__(self, model: nn.Module):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()
        os.makedirs("outputs", exist_ok=True)

    @torch.no_grad()
    def evaluate_dataset(self, dataloader: DataLoader) -> dict:
        all_probs = []
        all_labels = []

        for global_v, local_v, labels in dataloader:
            global_v, local_v, labels = (
                global_v.to(self.device),
                local_v.to(self.device),
                labels.to(self.device),
            )
            logits = self.model(global_v, local_v)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs).flatten()
        all_labels = np.array(all_labels).flatten()

        pr_auc = average_precision_score(all_labels, all_probs)
        precisions, recalls, thresholds = precision_recall_curve(all_labels, all_probs)

        f1_scores = np.divide(
            2 * (precisions * recalls),
            (precisions + recalls),
            out=np.zeros_like(precisions),
            where=(precisions + recalls) != 0,
        )

        optimal_idx = np.argmax(f1_scores)
        opt_threshold = thresholds[min(optimal_idx, len(thresholds) - 1)]
        y_pred_binary = (all_probs >= opt_threshold).astype(int)
        Visualizer.plot_pr_curve(all_labels, all_probs, pr_auc)
        Visualizer.plot_confusion_matrix(all_labels, y_pred_binary)
        Visualizer.plot_score_distribution(all_labels, all_probs)
        Visualizer.plot_roc_curve(all_labels, all_probs)

        logger.info(f"Optimal F1-Score Threshold calculated at: {opt_threshold:.4f}")
        logger.info("Diagnostics saved to outputs/ directory.")

        return {"pr_auc": pr_auc, "threshold": opt_threshold}


if __name__ == "__main__":
    from src.kepler_net.models.cnn_1d import KeplerCNN
    from src.kepler_net.training.trainer import KeplerDataset

    processed_dir = Path("data/processed")
    if processed_dir.exists():
        dataset = KeplerDataset(processed_dir)
        val_loader = DataLoader(dataset, batch_size=2, shuffle=False)
        model = KeplerCNN()

        evaluator = ModelEvaluator(model)
        metrics = evaluator.evaluate_dataset(val_loader)
        logger.info(f"Evaluation complete. PR-AUC: {metrics['pr_auc']:.4f}")
