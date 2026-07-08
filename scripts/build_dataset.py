# scripts/build_dataset.py

import logging
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import polars as pl

from src.kepler_net.data.ingest import KeplerIngestor
from src.kepler_net.data.preprocess import LightCurveProcessor

# Suppress lightkurve's noisy download logs to keep our multi-threading terminal clean
logging.getLogger("lightkurve").setLevel(logging.WARNING)
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Live NASA API endpoint for the Kepler Object of Interest (KOI) cumulative table
NASA_KOI_URL = (
    "https://exoplanetarchive.ipac.caltech.edu/cgi-bin/nstedAPI/"
    "nph-nstedAPI?table=cumulative&select=kepid,kepoi_name,"
    "koi_disposition,koi_period,koi_time0bk&format=csv"
)


def fetch_nasa_catalog(sample_size_per_class: int = 20) -> pl.DataFrame:
    """Hits the live NASA Exoplanet Archive API and returns a balanced Polars DataFrame."""
    logger.info("Downloading live KOI catalog from NASA Exoplanet Archive...")

    # Save the raw catalog to the interim folder to cache the query
    interim_dir = Path("data/interim")
    interim_dir.mkdir(parents=True, exist_ok=True)
    temp_csv = interim_dir / "koi_catalog.csv"

    urllib.request.urlretrieve(NASA_KOI_URL, temp_csv)

    # Blast through the CSV with Polars
    df = pl.read_csv(temp_csv, null_values=["", "null"])
    df = df.drop_nulls(subset=["koi_period", "koi_time0bk", "kepoi_name"])

    # Filter into Confirmed Planets (1) and False Positives/Eclipsing Binaries (0)
    planets = df.filter(pl.col("koi_disposition") == "CONFIRMED").head(
        sample_size_per_class
    )
    fps = df.filter(pl.col("koi_disposition") == "FALSE POSITIVE").head(
        sample_size_per_class
    )

    combined = pl.concat(
        [planets.with_columns(label=pl.lit(1)), fps.with_columns(label=pl.lit(0))]
    )

    return combined


def process_single_star(row: dict) -> bool:
    """Worker function to ingest and preprocess a single star."""
    ingestor = KeplerIngestor()
    processor = LightCurveProcessor()

    # FIX: MAST needs the KIC ID (the star) to find the data,
    # but we will save the file using the KOI name (the planet candidate)
    search_target = f"KIC {row['kepid']}"
    file_name = row["kepoi_name"].replace(".", "_").lower()

    try:
        # Pull Quarter 3 data
        raw_file = ingestor.fetch_target(search_target, quarter=3)
        if raw_file:
            processor.process_and_save(
                raw_file_path=raw_file,
                period=row["koi_period"],
                t0=row["koi_time0bk"],
                target_id=f"{file_name}_label_{row['label']}",
            )
            return True
    except Exception:
        # Silently fail targets with missing/corrupted Q3 data on the NASA servers
        pass

    return False


def build():
    # 1. Fetch the data table
    df = fetch_nasa_catalog(sample_size_per_class=200)
    targets = df.to_dicts()

    logger.info(
        f"Catalog built. Launching parallel download engine for {len(targets)} stars..."
    )

    successful = 0
    # 2. Parallelize the I/O bound network requests
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {executor.submit(process_single_star, t): t for t in targets}

        for i, future in enumerate(as_completed(futures), 1):
            if future.result():
                successful += 1
            if i % 5 == 0:
                logger.info(f"Progress: {i}/{len(targets)} targets processed...")

    logger.info(
        f"Dataset Build Complete! Successfully processed {successful} valid lightcurves."
    )


if __name__ == "__main__":
    build()
