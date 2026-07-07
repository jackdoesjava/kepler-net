# src/kepler_net/data/preprocess.py

import logging
from pathlib import Path
import numpy as np
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class LightCurveProcessor:
    """Handles phase folding, detrending, and binning of Kepler light curves."""
    
    def __init__(
        self, 
        num_global_bins: int = 2001, 
        num_local_bins: int = 201,
        processed_dir: str | Path = "data/processed"
    ):
        self.num_global_bins = num_global_bins
        self.num_local_bins = num_local_bins
        
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_and_fold(self, df: pl.DataFrame, period: float, t0: float) -> pl.DataFrame:
        """Median-normalizes flux and folds the time series around the period."""
        median_flux = df["flux"].median()
        
        return df.with_columns(
            norm_flux=(pl.col("flux") / median_flux) - 1.0,
            phase=(((pl.col("time") - t0) / period) - 
                   ((pl.col("time") - t0) / period + 0.5).floor())
        ).sort("phase")

    def _bin_data(self, phase: np.ndarray, flux: np.ndarray, bins: int, phase_range: tuple) -> np.ndarray:
        """Averages continuous phase/flux points into discrete 1D arrays."""
        bin_edges = np.linspace(phase_range[0], phase_range[1], bins + 1)
        binned_flux = np.zeros(bins)
        
        bin_indices = np.digitize(phase, bin_edges)
        
        for i in range(1, bins + 1):
            mask = (bin_indices == i)
            if np.any(mask):
                binned_flux[i-1] = np.median(flux[mask])
            else:
                binned_flux[i-1] = 0.0  
                
        return binned_flux

    def process_and_save(self, raw_file_path: Path, period: float, t0: float, target_id: str) -> Path:
        """Executes the pipeline and saves the final tensors as an .npz file."""
        logger.info(f"Processing {raw_file_path.name}...")
        
        df = pl.read_parquet(raw_file_path).drop_nulls()
        df = self._normalize_and_fold(df, period, t0)
        
        phase = df["phase"].to_numpy()
        flux = df["norm_flux"].to_numpy()
        
        global_view = self._bin_data(phase, flux, self.num_global_bins, (-0.5, 0.5))
        local_view = self._bin_data(phase, flux, self.num_local_bins, (-0.1, 0.1))
        
        output_path = self.processed_dir / f"{target_id}_tensors.npz"
        
        np.savez_compressed(
            output_path, 
            global_view=global_view.astype(np.float32), 
            local_view=local_view.astype(np.float32)
        )
        
        logger.info(f"Saved processed tensors to {output_path}")
        return output_path

if __name__ == "__main__":
    raw_file = Path("data/raw/kepler-10.parquet")
    
    if raw_file.exists():
        processor = LightCurveProcessor()
        processor.process_and_save(
            raw_file_path=raw_file, 
            period=0.837491, 
            t0=131.02641,
            target_id="kepler-10b"
        )
    else:
        logger.error("Raw data not found. Run ingest.py first.")