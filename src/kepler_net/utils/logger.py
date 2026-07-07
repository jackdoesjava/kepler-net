# src/kepler_net/utils/logger.py

import logging
import wandb

logger = logging.getLogger(__name__)

class ExperimentLogger:
    """Handles experiment tracking via Weights & Biases."""
    
    def __init__(self, project_name: str, config: dict):
        wandb.init(project=project_name, config=config)
        logger.info(f"Initialized W&B run: {wandb.run.name}")
        
    def log_metrics(self, metrics: dict, epoch: int):
        """Pushes a dictionary of metrics to the live dashboard."""
        wandb.log(metrics, step=epoch)
        
    def finish(self):
        wandb.finish()