#!/usr/bin/env python3
"""
Compare APOKASC-3 (Teff/logg) to your Kepler GMM-gated rotator-led sample.
This will usually show APOKASC clustered at low logg (giants), while your sample sits at high logg (dwarfs).

Outputs:
- /mnt/g/STAR_HPV/results/ripeness/apokasc3_vs_gmmgate_selection_bias.yaml
"""

from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

APO_CLEAN = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet"
GMMGATE = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
OUT = Path("/mnt/g/STAR_HPV/results/ripeness/apokasc3_vs_gmmgate_selection_bias.yaml")


def qstats(x: pd.Series) -> dict:
    x = pd.to_numeric(x, errors="coerce")
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return {"n": 0}
    return {
        "n": int(len(x)),
        "min": float(np.min(x)),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(np.max(x)),
    }


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    apo = pd.read_parquet(APO_CLEAN)
    gmm = pd.read_parquet(GMMGATE)

    # APO columns are Teff16, logg etc vary; use what exists
    # In your cleaned file, you already have: Teff16, _RA, _DE, and many others.
    # Find best-effort Teff/logg columns:
    teff_apo = None
    for c in ["Teff16", "Teff", "TEFF", "Teff_gspspec", "Teff_gspphot"]:
        if c in apo.columns:
            teff_apo = c
            break

    logg_apo = None
    for c in ["logg", "log(g)", "logg_gspspec_calibrated", "logg_gspphot", "logg"]:
        if c in apo.columns:
            logg_apo = c
            break

    # Kepler gmmgate uses Gaia values; you have teff_bin_s; may not have logg.
    teff_gmm = None
    for c in ["Teff", "teff_gspphot", "teff_gspspec", "Teff_gspphot"]:
        if c in gmm.columns:
            teff_gmm = c
            break

    res = {
        "apokasc3": {
            "rows": int(len(apo)),
            "teff_col": teff_apo,
            "logg_col": logg_apo,
            "teff_stats": qstats(apo[teff_apo]) if teff_apo else {"n": 0},
            "logg_stats": qstats(apo[logg_apo]) if logg_apo else {"n": 0},
        },
        "kepler_gmmgate": {
            "rows": int(len(gmm)),
            "teff_col": teff_gmm,
            "teff_stats": qstats(gmm[teff_gmm]) if teff_gmm else {"n": 0},
        },
        "interpretation": [
            "If APOKASC logg is mostly ~2–3 while your sample is dwarfs, then KIC overlap=0 is expected selection bias.",
            "If APOKASC contains many dwarfs (logg~4+) and overlap is still 0, then investigate KIC cleaning or catalog mismatch.",
        ],
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(res, f, sort_keys=False)

    print(f"saved: {OUT}")


if __name__ == "__main__":
    main()
