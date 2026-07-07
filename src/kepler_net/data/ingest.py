# src/kepler_net/data/ingest.py

import logging
from pathlib import Path
import lightkurve as lk
import polars as pl

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class KeplerIngestor:
    """Handles retrieval of Kepler photometric data from MAST."""
    
    def __init__(self, raw_data_dir: str | Path = "data/raw"):
        self.raw_data_dir = Path(raw_data_dir)
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        
    def fetch_target(self, target_name: str, quarter: str | int | list = "all") -> Path | None:
        """Downloads starlight data and saves it to a Polars Parquet file."""
        logger.info(f"Querying MAST for {target_name}...")
        
        search_result = lk.search_lightcurve(target_name, author="Kepler", quarter=quarter)
        if not search_result:
            logger.error(f"No data found for {target_name}.")
            return None
            
        logger.info(f"Downloading {len(search_result)} datasets for {target_name}...")

        lc_collection = search_result.download_all()
        lc_stitched = lc_collection.stitch().remove_nans()

        df = pl.DataFrame({
            "time": lc_stitched.time.value,
            "flux": lc_stitched.flux.value,
            "flux_err": lc_stitched.flux_err.value
        })

        filename = target_name.replace(" ", "_").lower()
        output_path = self.raw_data_dir / f"{filename}.parquet"
        
        df.write_parquet(output_path)
        logger.info(f"Saved {df.height} rows to {output_path}")
        
        return output_path

if __name__ == "__main__":
    ingestor = KeplerIngestor()
    ingestor.fetch_target("Kepler-10", quarter=3) 