import numpy as np
import pandas as pd

# Pre-computed reference table (Elo band → percentiles)
# Example structure: rows per Elo bucket every 200 Elo
REFERENCE = {
    800:  {"acpl": [2000, 3500, 5000], "entropy": [1.0, 3.0, 6.0]},  # 25th/50th/75th
    1000: {"acpl": [1500, 3000, 4500], "entropy": [1.2, 3.5, 6.5]},
    1200: {"acpl": [1200, 2500, 4000], "entropy": [1.4, 3.8, 7.0]},
    1400: {"acpl": [1000, 2000, 3500], "entropy": [1.6, 4.0, 7.5]},
    1600: {"acpl": [800, 1800, 3200], "entropy": [1.8, 4.2, 8.0]},
    1800: {"acpl": [600, 1500, 2800], "entropy": [2.0, 4.5, 8.5]},
    2000: {"acpl": [400, 1200, 2400], "entropy": [2.2, 4.8, 9.0]},
    2200: {"acpl": [300, 1000, 2000], "entropy": [2.4, 5.0, 9.5]},
    2400: {"acpl": [200, 800, 1600], "entropy": [2.6, 5.2, 10.0]},
    2600: {"acpl": [150, 600, 1200], "entropy": [2.8, 5.4, 10.5]},
    2800: {"acpl": [100, 400, 800], "entropy": [3.0, 5.6, 11.0]},
}

def _pct(value: float, quartiles: list[float]) -> int:
    """Return approximate percentile (0,25,50,75,100) given quartiles array."""
    if value is None or np.isnan(value):
        return None
    
    if value == 0.0:
        return 5  # Very low percentile for 0 values
    
    if value <= quartiles[0]:
        return 10
    if value <= quartiles[1]:
        return 35
    if value <= quartiles[2]:
        return 65
    return 90

def compute_benchmark(avg_acpl: float,
                      mean_entropy: float,
                      player_elo: int | None) -> dict:
    """Map player metrics to percentile buckets vs nearest Elo band."""
    import logging
    import numpy as np
    logger = logging.getLogger(__name__)
    
    logger.info(f"DEBUG BENCHMARK: compute_benchmark called with avg_acpl={avg_acpl}, mean_entropy={mean_entropy}, player_elo={player_elo}")
    
    if player_elo is None:
        logger.warning("DEBUG BENCHMARK: player_elo is None, using default ELO 1600")
        player_elo = 1600  # Use reasonable default instead of returning empty dict

    # round to nearest 200 Elo bucket
    bucket = int(round(player_elo / 200) * 200)
    logger.info(f"DEBUG BENCHMARK: ELO {player_elo} rounded to bucket {bucket}")
    
    ref = REFERENCE.get(bucket)
    if not ref:
        available_buckets = sorted(REFERENCE.keys())
        closest_bucket = min(available_buckets, key=lambda x: abs(x - bucket))
        ref = REFERENCE.get(closest_bucket)
        logger.info(f"DEBUG BENCHMARK: No exact match for bucket {bucket}, using closest bucket {closest_bucket}")
        
        if not ref:
            logger.warning(f"DEBUG BENCHMARK: No reference data available, returning empty dict")
            return {}

    logger.info(f"DEBUG BENCHMARK: Found reference data for bucket {bucket}: {ref}")
    
    pct_acpl = _pct(avg_acpl, ref["acpl"])
    pct_entropy = _pct(mean_entropy, ref["entropy"])

    logger.info(f"DEBUG BENCHMARK: Calculated percentiles - acpl: {pct_acpl}, entropy: {pct_entropy}")

    return {
        "percentile_acpl": pct_acpl,
        "percentile_entropy": pct_entropy,
    }
