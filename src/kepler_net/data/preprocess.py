# src/kepler_net/data/preprocess.py

import logging
from pathlib import Path
import numpy as np
import polars as pl

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class LightCurveProcessor:
    """Handles professional-grade light curve detrending, sigma-clipping, and binning."""

    def __init__(
        self,
        num_global_bins: int = 2001,
        num_local_bins: int = 201,
        processed_dir: str | Path = "data/processed",
    ):
        self.num_global_bins = num_global_bins
        self.num_local_bins = num_local_bins
        self.processed_dir = Path(processed_dir)
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def _normalize_and_fold(
        self, df: pl.DataFrame, period: float, t0: float
    ) -> pl.DataFrame:
        """Applies sigma-clipping and Z-score normalization for neural network readiness."""

        # 1. Sigma Clipping (Remove outliers/cosmic rays)
        mean, std = df["flux"].mean(), df["flux"].std()
        df = df.filter(
            (pl.col("flux") > mean - 3 * std) & (pl.col("flux") < mean + 3 * std)
        )

        # 2. Z-Score Normalization (Centers at 0, unit variance)
        # This is CRITICAL for Neural Network convergence
        df = df.with_columns(
            norm_flux=(pl.col("flux") - pl.col("flux").mean()) / pl.col("flux").std()
        )

        # 3. Phase Folding (Modulo arithmetic)
        df = df.with_columns(phase=((pl.col("time") - t0) / period) % 1)

        # Shift phase so transit is centered at 0.0
        return df.with_columns(
            phase=pl.when(pl.col("phase") > 0.5)
            .then(pl.col("phase") - 1.0)
            .otherwise(pl.col("phase"))
        ).sort("phase")

    def _bin_data(
        self, phase: np.ndarray, flux: np.ndarray, bins: int, phase_range: tuple
    ) -> np.ndarray:
        """Fast vectorized binning of light curve data."""
        # Create bins
        hist, bin_edges = np.histogram(phase, bins=bins, range=phase_range)

        # Compute median flux in each bin using digitize
        indices = np.digitize(phase, bin_edges)
        binned_flux = np.array(
            [
                np.median(flux[indices == i]) if np.any(indices == i) else 0.0
                for i in range(1, len(bin_edges))
            ]
        )

        return binned_flux

    def process_and_save(
        self, raw_file_path: Path, period: float, t0: float, target_id: str
    ) -> Path:
        """Executes the pipeline and saves tensors as Z-score normalized .npz."""
        logger.info(f"Processing {raw_file_path.name} (Z-Score Normalized)...")

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
            local_view=local_view.astype(np.float32),
        )

        return output_path
