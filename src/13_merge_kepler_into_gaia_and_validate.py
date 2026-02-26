import os
import yaml
import numpy as np
import pandas as pd

GAIA_IN = "/mnt/g/STAR_HPV/processed/state_vectors/gaia_shv_state_vector_gateaware_v2_absH.parquet"
MCQ_IN  = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"
X_IN    = "/mnt/g/STAR_HPV/raw/kepler/godoyrivera25_tableA1.csv"

OUT_PARQ = "/mnt/g/STAR_HPV/processed/state_vectors/gaia_kepler_overlap.parquet"
OUT_YAML = "/mnt/g/STAR_HPV/results/kepler_validation/summary.yaml"


def parse_int64_exact(s: pd.Series, name: str) -> pd.Series:
    """
    Parse large integer IDs exactly by going through string -> Python int.
    Avoid float64 precision loss.
    """
    st = s.astype("string").str.strip()
    # drop obvious empties
    st = st.replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "<NA>": pd.NA})
    # Some CSVs may contain ".0" if exported through Excel
    st = st.str.replace(r"\.0$", "", regex=True)

    def to_int(x):
        if x is None or x is pd.NA:
            return pd.NA
        try:
            return int(x)
        except Exception:
            return pd.NA

    out = st.map(to_int)
    return out.astype("Int64")


def main():
    os.makedirs(os.path.dirname(OUT_PARQ), exist_ok=True)
    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)

    gaia = pd.read_parquet(GAIA_IN)
    mcq  = pd.read_parquet(MCQ_IN)

    # Read only needed columns, as strings to preserve huge integers
    x = pd.read_csv(
        X_IN,
        usecols=["KIC", "GaiaDR3"],
        dtype={"KIC": "string", "GaiaDR3": "string"},
        low_memory=False
    ).rename(columns={"KIC": "kic", "GaiaDR3": "source_id"})

    # Exact parsing
    x["kic"] = parse_int64_exact(x["kic"], "kic")
    x["source_id"] = parse_int64_exact(x["source_id"], "source_id")
    x = x.dropna(subset=["kic", "source_id"]).copy()

    # Normalize McQuillan
    mcq["kic"] = parse_int64_exact(mcq["kic"], "kic")
    mcq["prot_days"] = pd.to_numeric(mcq["prot_days"], errors="coerce")
    mcq = mcq.dropna(subset=["kic", "prot_days"]).copy()
    mcq = mcq[mcq["prot_days"] > 0].copy()

    # Ensure Gaia source_id is Int64 for exact join
    gaia["source_id"] = parse_int64_exact(gaia["source_id"], "gaia_source_id")
    gaia = gaia.dropna(subset=["source_id"]).copy()

    # Merge: McQ -> crossmatch -> Gaia
    km = mcq.merge(x, on="kic", how="inner")
    merged = gaia.merge(km[["source_id", "kic", "prot_days"]], on="source_id", how="inner")

    # Compare Gaia Prot vs Kepler Prot if Gaia prot present
    if "prot" in merged.columns and len(merged) > 0:
        merged["log_prot_gaia"] = np.log10(merged["prot"].astype(float))
        merged["log_prot_kepler"] = np.log10(merged["prot_days"].astype(float))
        merged["dlog_prot"] = merged["log_prot_gaia"] - merged["log_prot_kepler"]

    merged.to_parquet(OUT_PARQ, index=False)

    summary = {
        "gaia_rows_in": int(gaia.shape[0]),
        "mcquillan_rows": int(mcq.shape[0]),
        "crossmatch_rows": int(x.shape[0]),
        "overlap_rows": int(merged.shape[0]),
        "overlap_by_teff_bin": merged.groupby("teff_bin").size().to_dict() if ("teff_bin" in merged.columns and len(merged) > 0) else {},
    }

    if "dlog_prot" in merged.columns and len(merged) > 0:
        med = float(np.nanmedian(merged["dlog_prot"]))
        mad = float(np.nanmedian(np.abs(merged["dlog_prot"] - med)))
        summary["prot_log_diff_median"] = med
        summary["prot_log_diff_mad"] = mad

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved overlap parquet: {OUT_PARQ}")
    print(f"saved summary: {OUT_YAML}")
    print(summary)


if __name__ == "__main__":
    main()
