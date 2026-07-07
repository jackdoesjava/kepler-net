# src/kepler_net/training/trainer.py

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import average_precision_score

from src.kepler_net.models.cnn_1d import KeplerCNN
from src.kepler_net.utils.logger import ExperimentLogger

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KeplerDataset(Dataset):
    """
    Dynamically maps and loads all preprocessed exoplanet tensors.
    Parses binary labels directly from filenames (e.g., 'kepler-10_label_1.npz').
    """
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        all_files = list(self.data_dir.glob("*.npz"))
        
        self.file_paths = []
        self.labels = []
        
        for path in all_files:
            if "_label_" in path.stem:
                # path.stem is "kepler-10_label_1_tensors"
                # This splits it to "1_tensors" and then removes "_tensors" leaving just "1"
                label_string = path.stem.split("_label_")[-1].replace("_tensors", "")
                self.labels.append(float(label_string))
                self.file_paths.append(path)
            else:
                logger.warning(f"Skipping {path.name}: No '_label_' format found.")
                
        if not self.file_paths:
            raise FileNotFoundError(f"No labeled .npz files found in {data_dir}.")

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        data = np.load(path)
        global_view = torch.tensor(data['global_view'], dtype=torch.float32).unsqueeze(0)
        local_view = torch.tensor(data['local_view'], dtype=torch.float32).unsqueeze(0)
        target_label = torch.tensor([label], dtype=torch.float32)
        
        return global_view, local_view, target_label

class ModelTrainer:
    def __init__(self, model: nn.Module, learning_rate: float, logger: ExperimentLogger = None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.exp_logger = logger
        
        # Class imbalance handling
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([200.0]).to(self.device))
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_one_epoch(self, dataloader: DataLoader, epoch: int) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        for global_v, local_v, labels in dataloader:
            global_v, local_v, labels = global_v.to(self.device), local_v.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(global_v, local_v)
            loss = self.criterion(logits, labels)
            
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            probs = torch.sigmoid(logits).detach().cpu().numpy()
            all_preds.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            
        avg_loss = total_loss / len(dataloader)
        
        # Calculate true PR-AUC across the entire batch matrix
        try:
            pr_auc = average_precision_score(all_labels, all_preds)
        except ValueError:
            pr_auc = 0.0 
            
        if self.exp_logger:
            self.exp_logger.log_metrics({"train_loss": avg_loss, "train_pr_auc": pr_auc}, epoch=epoch)
            
        return avg_loss, pr_auc

if __name__ == "__main__":
    processed_dir = Path("data/processed")
    
    # Initialize offline experiment tracker
    tracker = ExperimentLogger(project_name="kepler-net", config={"learning_rate": 0.001, "epochs": 10})
    
    # Load the comprehensive dataset
    dataset = KeplerDataset(processed_dir)
    # Batch size set to 2 for a balanced distribution among our 4 curated test targets
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    logger.info(f"Loaded dataset successfully. Total target stars to scan: {len(dataset)}")
    
    cnn = KeplerCNN()
    trainer = ModelTrainer(model=cnn, learning_rate=0.001, logger=tracker)
    
    logger.info("Starting production multi-class training loop...")
    for epoch in range(1, 11):
        loss, pr_auc = trainer.train_one_epoch(dataloader, epoch)
        logger.info(f"Epoch {epoch}/10 | Avg Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f}")
        
    tracker.finish()