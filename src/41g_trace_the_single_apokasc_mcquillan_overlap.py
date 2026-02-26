#!/usr/bin/env python3
"""
Find the single KIC overlap between APOKASC-3 ages and McQuillan rotators,
then trace whether it appears in:
- Kepler rotator-led Gaia merge (if available)
- Kepler GMM-gated sample
- Window / tails (if present)

Outputs:
- /mnt/g/STAR_HPV/results/ripeness/apokasc_mcquillan_bridge_star.yaml
- /mnt/g/STAR_HPV/results/ripeness/apokasc_mcquillan_bridge_star.csv
"""

from __future__ import annotations
from pathlib import Path
import yaml
import pandas as pd
import numpy as np

APO_AGE_KIC = "/mnt/g/STAR_HPV/raw/ages/apokasc3_kic_ages_true.parquet"
MCQ = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"
IDMAP = "/mnt/g/STAR_HPV/interim/kepler/kepler_gaia_ids.csv"
GMMGATE = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"

# Optional catalogs
ANCHOR = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet"
TAILS = [
    "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q90.parquet",
    "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_global_postgate_q95.parquet",
    "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q90.parquet",
    "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_q95.parquet",
    "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q90.parquet",
    "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_q95.parquet",
]

OUTDIR = Path("/mnt/g/STAR_HPV/results/ripeness/apokasc_mcquillan_bridge_star")
OUTCSV = OUTDIR / "bridge_star.csv"
OUTYAML = OUTDIR / "bridge_star.yaml"


def to_int(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").astype("Int64")


def safe_read_parquet(path: str) -> pd.DataFrame | None:
    try:
        return pd.read_parquet(path)
    except Exception:
        return None


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    apo = pd.read_parquet(APO_AGE_KIC)
    mcq = pd.read_parquet(MCQ)

    apo_k = set(to_int(apo["kic"]).dropna().astype(int).tolist())
    mcq_k = set(to_int(mcq["kic"]).dropna().astype(int).tolist())
    inter = sorted(list(apo_k & mcq_k))

    if len(inter) == 0:
        raise RuntimeError("No APOKASC∩McQuillan overlap found (expected 1 per your audit).")

    if len(inter) > 10:
        print(f"warning: overlap count={len(inter)}; tracing first 10.")

    kic = inter[0]

    row = {}
    row["kic"] = int(kic)

    # APO age
    arow = apo.loc[to_int(apo["kic"]).astype("Int64") == kic].head(1)
    if len(arow):
        row["age_gyr"] = float(arow["age_gyr"].iloc[0])
        row["age_gyr_err"] = float(arow["age_gyr_err"].iloc[0]) if "age_gyr_err" in arow.columns and pd.notna(arow["age_gyr_err"].iloc[0]) else np.nan
        row["age_source"] = str(arow["age_source"].iloc[0]) if "age_source" in arow.columns else "APOKASC"
    else:
        row["age_gyr"] = np.nan
        row["age_gyr_err"] = np.nan
        row["age_source"] = "APOKASC"

    # McQuillan Prot/Teff
    mrow = mcq.loc[to_int(mcq["kic"]).astype("Int64") == kic].head(1)
    if len(mrow):
        row["prot_days_mcq"] = float(mrow["prot_days"].iloc[0])
        row["teff_mcq"] = float(mrow["Teff"].iloc[0]) if "Teff" in mrow.columns else np.nan
    else:
        row["prot_days_mcq"] = np.nan
        row["teff_mcq"] = np.nan

    # Gaia mapping (source_id)
    idmap = pd.read_csv(IDMAP, dtype={"source_id": "string"})
    idmap["kic"] = to_int(idmap["kic"])
    irow = idmap.loc[idmap["kic"].astype("Int64") == kic].head(1)
    if len(irow):
        row["source_id"] = str(irow["source_id"].iloc[0])
    else:
        row["source_id"] = None

    # Presence in GMM gate
    gmm = pd.read_parquet(GMMGATE)
    gmm["kic"] = to_int(gmm["kic"])
    grow = gmm.loc[gmm["kic"].astype("Int64") == kic].head(1)
    row["in_gmmgate"] = bool(len(grow))
    if len(grow):
        # carry key scores if present
        for c in ["teff_bin_s", "prot_days", "Phi_gmm", "in_window_gmm", "ripeness_gmm"]:
            if c in grow.columns:
                row[c] = grow[c].iloc[0]

        # If ripeness not stored, compute it
        if "ripeness_gmm" not in row and ("Phi_gmm" in grow.columns and "prot_days" in grow.columns):
            row["ripeness_gmm"] = float(grow["Phi_gmm"].iloc[0]) * float(grow["prot_days"].iloc[0])

    # Presence in anchor and tails
    row["in_anchor"] = False
    try:
        anc = pd.read_parquet(ANCHOR)
        anc["kic"] = to_int(anc["kic"])
        row["in_anchor"] = bool(len(anc.loc[anc["kic"].astype("Int64") == kic]))
    except Exception:
        pass

    tails_presence = {}
    for p in TAILS:
        df = safe_read_parquet(p)
        if df is None:
            continue
        if "kic" in df.columns:
            df["kic"] = to_int(df["kic"])
            tails_presence[Path(p).name] = bool(len(df.loc[df["kic"].astype("Int64") == kic]))
        else:
            tails_presence[Path(p).name] = False
    row["tails_presence"] = tails_presence

    # Save
    outdf = pd.DataFrame([row])
    outdf.to_csv(OUTCSV, index=False)

    summary = {
        "overlap_count": int(len(inter)),
        "kic_traced": int(kic),
        "row": {k: (None if (isinstance(v, float) and np.isnan(v)) else v) for k, v in row.items()},
        "notes": [
            "This star is the only bridge between APOKASC ages and McQuillan rotators in your current build.",
            "If it is not in the GMM-gated sample, that is consistent with your slow-peak gating removing it.",
        ],
    }
    with open(OUTYAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUTCSV}")
    print(f"saved: {OUTYAML}")
    print(summary)


if __name__ == "__main__":
    main()
