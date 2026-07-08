# src/kepler_net/training/trainer.py

import logging
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import average_precision_score
import hydra
from omegaconf import DictConfig

from src.kepler_net.models.cnn_1d import KeplerCNN
from src.kepler_net.utils.logger import ExperimentLogger

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class KeplerDataset(Dataset):
    """
    Dynamically maps and loads all preprocessed exoplanet tensors.
    Parses binary labels directly from filenames and gracefully skips bad files.
    """

    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        all_files = list(self.data_dir.glob("*.npz"))

        self.file_paths = []
        self.labels = []

        for path in all_files:
            if "_label_" in path.stem:
                # Extracts the label integer and ignores the _tensors suffix
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

        # Lazy loading of tensors
        data = np.load(path)
        global_view = torch.tensor(data["global_view"], dtype=torch.float32).unsqueeze(
            0
        )
        local_view = torch.tensor(data["local_view"], dtype=torch.float32).unsqueeze(0)
        target_label = torch.tensor([label], dtype=torch.float32)

        return global_view, local_view, target_label


class ModelTrainer:
    """Handles the training loop, class imbalance, and metric calculation."""

    def __init__(
        self, model: nn.Module, learning_rate: float, logger: ExperimentLogger = None
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.exp_logger = logger

        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([1.0]).to(self.device)
        )
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

    def train_one_epoch(
        self, dataloader: DataLoader, epoch: int
    ) -> tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        all_preds = []
        all_labels = []

        for global_v, local_v, labels in dataloader:
            global_v, local_v, labels = (
                global_v.to(self.device),
                local_v.to(self.device),
                labels.to(self.device),
            )

            # --- AGGRESSIVE AUGMENTATION ---
            # 1. Randomly roll the global sequence left or right by up to 100 indices
            shift = torch.randint(-100, 100, (1,)).item()
            global_v = torch.roll(global_v, shifts=shift, dims=-1)

            # 2. Add Gaussian noise
            global_v = global_v + torch.randn_like(global_v) * 0.05
            local_v = local_v + torch.randn_like(local_v) * 0.05
            # -------------------------------

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

        try:
            pr_auc = average_precision_score(all_labels, all_preds)
        except ValueError:
            pr_auc = 0.0

        if self.exp_logger:
            self.exp_logger.log_metrics(
                {"train_loss": avg_loss, "train_pr_auc": pr_auc}, epoch=epoch
            )

        return avg_loss, pr_auc


@hydra.main(version_base=None, config_path="../../../configs", config_name="config")
def main(cfg: DictConfig):
    """Hydra automatically loads the YAML files into the 'cfg' object."""

    # 1. Set the Random Seed for reproducibility
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    processed_dir = Path(cfg.dataset.processed_data_dir)

    # 2. Initialize W&B using Hydra config values
    tracker = ExperimentLogger(
        project_name=cfg.project_name,
        config={
            "learning_rate": cfg.model.learning_rate,
            "batch_size": cfg.dataset.batch_size,
            "epochs": cfg.max_epochs,
        },
    )

    # 3. Load Data dynamically via YAML parameters
    try:
        dataset = KeplerDataset(processed_dir)
        dataloader = DataLoader(
            dataset, batch_size=cfg.dataset.batch_size, shuffle=True
        )
        logger.info(f"Loaded dataset successfully. Total target stars: {len(dataset)}")
    except FileNotFoundError as e:
        logger.error(e)
        return

    # 4. Initialize Model & Trainer dynamically
    cnn = KeplerCNN(
        global_length=cfg.dataset.num_global_bins,
        local_length=cfg.dataset.num_local_bins,
    )

    trainer = ModelTrainer(
        model=cnn, learning_rate=cfg.model.learning_rate, logger=tracker
    )

    # 5. Execute Training Loop
    logger.info(f"Starting training loop for {cfg.max_epochs} epochs...")
    for epoch in range(1, cfg.max_epochs + 1):
        loss, pr_auc = trainer.train_one_epoch(dataloader, epoch)
        logger.info(
            f"Epoch {epoch}/{cfg.max_epochs} | Avg Loss: {loss:.4f} | PR-AUC: {pr_auc:.4f}"
        )

    tracker.finish()


if __name__ == "__main__":
    main()
