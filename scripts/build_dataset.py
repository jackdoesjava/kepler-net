# scripts/build_dataset.py

import logging
from src.kepler_net.data.ingest import KeplerIngestor
from src.kepler_net.data.preprocess import LightCurveProcessor

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

TARGETS = [
    # Confirmed Planets (Label 1)
    {"name": "Kepler-10", "period": 0.837491, "t0": 131.026, "label": 1},
    {"name": "Kepler-22", "period": 289.862, "t0": 169.535, "label": 1},
    
    # Eclipsing Binaries (Label 0) - These look like planets but have sharp V-shaped dips
    {"name": "KIC 11446443", "period": 14.618, "t0": 132.88, "label": 0},
    {"name": "KIC 3239945", "period": 4.148, "t0": 133.56, "label": 0},
]

def build():
    ingestor = KeplerIngestor()
    processor = LightCurveProcessor()
    
    for target in TARGETS:
        logger.info(f"--- Processing {target['name']} ---")
        
        raw_file = ingestor.fetch_target(target['name'], quarter=3)
        
        if raw_file:
            target_id = target['name'].replace(" ", "_").lower()
            processor.process_and_save(
                raw_file_path=raw_file,
                period=target['period'],
                t0=target['t0'],
                target_id=f"{target_id}_label_{target['label']}"
            )

if __name__ == "__main__":
    build()