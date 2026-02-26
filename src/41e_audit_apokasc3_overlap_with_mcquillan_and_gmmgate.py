#!/usr/bin/env python3
"""
Audit overlap between APOKASC-3 age catalog (by KIC) and:
1) McQuillan 2014 rotators (full 34k)
2) Kepler rotator-led Gaia+Kepler merged sample (pre-gate, if you have it)
3) Kepler GMM-gated sample (7366)

This tells us if "0 overlap" is a real selection mismatch or a KIC mismatch.

Outputs:
- /mnt/g/STAR_HPV/results/ripeness/apokasc3_overlap_audit.yaml
"""

from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd

APO_AGE = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet"
MCQ = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"
GMMGATE = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"

# Optional: if you have the pre-gate rotator-led state vector
PRE_GATE_CANDIDATES = [
    "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector.parquet",
    "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet",  # harmless duplicate
]

OUT = Path("/mnt/g/STAR_HPV/results/ripeness/apokasc3_overlap_audit.yaml")


def to_int_kic(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def load_kic_set(path: str, kic_col_candidates=("kic", "KIC")) -> set[int]:
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)
    kcol = None
    for c in kic_col_candidates:
        if c in df.columns:
            kcol = c
            break
    if kcol is None:
        return set()
    k = to_int_kic(df[kcol]).dropna().astype(int).unique().tolist()
    return set(k)


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)

    apo = pd.read_parquet(APO_AGE)
    apo_k = set(to_int_kic(apo["kic"]).dropna().astype(int).unique().tolist())

    mcq = pd.read_parquet(MCQ)
    mcq_k = set(to_int_kic(mcq["kic"]).dropna().astype(int).unique().tolist())

    gmm = pd.read_parquet(GMMGATE)
    gmm_k = set(to_int_kic(gmm["kic"]).dropna().astype(int).unique().tolist())

    # pre-gate if exists
    pre = None
    for p in PRE_GATE_CANDIDATES:
        try:
            pre = p
            break
        except Exception:
            pass

    res = {
        "apokasc3": {"rows": int(len(apo)), "unique_kic": int(len(apo_k))},
        "mcquillan": {"rows": int(len(mcq)), "unique_kic": int(len(mcq_k))},
        "kepler_gmmgate": {"rows": int(len(gmm)), "unique_kic": int(len(gmm_k))},
        "overlap": {
            "apokasc_vs_mcquillan": int(len(apo_k & mcq_k)),
            "apokasc_vs_gmmgate": int(len(apo_k & gmm_k)),
            "mcquillan_vs_gmmgate": int(len(mcq_k & gmm_k)),
        },
        "notes": [
            "If apokasc_vs_mcquillan > 0 but apokasc_vs_gmmgate = 0, then your gating/windowing selected a dwarf regime disjoint from APOKASC (expected).",
            "If apokasc_vs_mcquillan = 0, investigate KIC namespace/cleaning or verify APOKASC table is truly KIC-based (should be).",
        ],
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(res, f, sort_keys=False)

    print(f"saved: {OUT}")
    print(res)


if __name__ == "__main__":
    main()
