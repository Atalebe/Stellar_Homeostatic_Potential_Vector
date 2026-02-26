import os
import yaml
import numpy as np
import pandas as pd


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def mad(x: pd.Series) -> float:
    x = x.dropna().to_numpy()
    if x.size == 0:
        return np.nan
    med = np.median(x)
    return np.median(np.abs(x - med))


def robust_z(x: pd.Series) -> pd.Series:
    m = x.median()
    s = mad(x)
    if not np.isfinite(s) or s == 0:
        # fallback to IQR scale
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        s = (q3 - q1) / 1.349 if (q3 > q1) else np.nan
    if not np.isfinite(s) or s == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - m) / s


def teff_bin(teff: pd.Series, edges: list[float]) -> pd.Categorical:
    labels = [f"{int(edges[i])}-{int(edges[i+1])}" for i in range(len(edges)-1)]
    return pd.cut(teff, bins=edges, labels=labels, include_lowest=True, right=False)


def main():
    cfg = load_cfg("configs/state_vector.yaml")
    p_in = cfg["paths"]["in_parquet"]
    p_out = cfg["paths"]["out_parquet"]
    p_sum = cfg["paths"]["out_summary"]

    os.makedirs(os.path.dirname(p_out), exist_ok=True)
    os.makedirs(os.path.dirname(p_sum), exist_ok=True)

    df = pd.read_parquet(p_in)

    f = cfg["filters"]
    # Basic cleaning
    df = df.copy()
    df = df[(df["ruwe"] < f["ruwe_max"])].copy()
    df = df[df["parallax_error"] > 0].copy()
    df["poe"] = df["parallax"] / df["parallax_error"]
    df = df[df["poe"] > f["poe_min"]].copy()

    df = df[df["teff_gspphot"].between(f["teff_min"], f["teff_max"])].copy()
    df = df[df["prot"].between(f["prot_min_days"], f["prot_max_days"])].copy()

    # Class gating: dwarfs vs giants if logg present
    cg = cfg["class_gating"]
    df["class_stage"] = "unknown"
    if "logg_gspphot" in df.columns:
        df.loc[df["logg_gspphot"] >= cg["dwarf_logg_min"], "class_stage"] = "dwarf"
        df.loc[df["logg_gspphot"] <= cg["giant_logg_max"], "class_stage"] = "giant"

    # Teff bins for dwarfs (others get 'other')
    edges = cfg["bins"]["teff_edges"]
    df["teff_bin"] = teff_bin(df["teff_gspphot"], edges).astype("object")
    df.loc[df["class_stage"] != "dwarf", "teff_bin"] = "other"

    # -----------------------
    # Tier 1 proxy definitions
    # -----------------------
    # H: forcing proxy (phot_g_mean_mag is inverted luminosity proxy)
    # Using -G makes brighter -> higher H.
    df["H_raw"] = -df["phot_g_mean_mag"]

    # S: hazard proxy from activity index (arot) + RUWE penalty
    # Larger activity index means more activity; hazard is negative
    # so define S_raw as negative of activity + contamination.
    df["S_raw"] = -(df["arot"].astype(float)) - 0.5 * (df["ruwe"].astype(float))

    # M: memory proxy, use metallicity proxy if present (mh_gspphot)
    # Keep it simple: higher metallicity -> higher reservoir.
    if "mh_gspphot" in df.columns:
        df["M_raw"] = df["mh_gspphot"].astype(float)
    else:
        df["M_raw"] = np.nan

    # R: regulation proxy from rotation: closeness to class median rotation
    # within teff_bin among dwarfs.
    # Define R_raw = -|log10(prot) - log10(median_prot_in_bin)|
    df["log_prot"] = np.log10(df["prot"].astype(float))
    med_by_bin = df.groupby("teff_bin")["log_prot"].median()
    df["log_prot_med"] = df["teff_bin"].map(med_by_bin)
    df["R_raw"] = -(df["log_prot"] - df["log_prot_med"]).abs()

    # Robust normalize within class group
    group_key = ["class_stage", "teff_bin"]

    for col in ["R_raw", "H_raw", "M_raw", "S_raw"]:
        df[f"{col[:-4]}"] = df.groupby(group_key)[col].transform(robust_z)

    # Scalar potential
    w = cfg["weights"]
    df["Phi"] = (
        w["wR"] * df["R"] +
        w["wH"] * df["H"] +
        w["wM"] * df["M"] +
        w["wS"] * df["S"]
    )

    # Window membership per group (bounded)
    qlo = float(cfg["window"]["q_lo"])
    qhi = float(cfg["window"]["q_hi"])
    qlo_by = df.groupby(group_key)["Phi"].transform(lambda x: x.quantile(qlo))
    qhi_by = df.groupby(group_key)["Phi"].transform(lambda x: x.quantile(qhi))
    df["Phi_lo"] = qlo_by
    df["Phi_hi"] = qhi_by
    df["in_window"] = (df["Phi"] >= df["Phi_lo"]) & (df["Phi"] <= df["Phi_hi"])

    # Keep a clean output schema
    keep = [
        "source_id", "ra", "dec",
        "parallax", "parallax_error", "poe",
        "phot_g_mean_mag", "bp_rp", "ruwe",
        "teff_gspphot", "logg_gspphot", "mh_gspphot",
        "prot", "prot_err", "arot", "arot_err",
        "class_stage", "teff_bin",
        "R_raw", "H_raw", "M_raw", "S_raw",
        "R", "H", "M", "S",
        "Phi", "Phi_lo", "Phi_hi", "in_window"
    ]
    keep = [c for c in keep if c in df.columns]
    out = df[keep].copy()

    out.to_parquet(p_out, index=False)

    # Summary
    summary = {
        "n_in": int(pd.read_parquet(p_in, columns=["source_id"]).shape[0]),
        "n_after_filters": int(out.shape[0]),
        "window_fraction_overall": float(out["in_window"].mean()),
        "by_stage": out.groupby("class_stage")["in_window"].mean().to_dict(),
        "by_teff_bin_dwarfs": out[out["class_stage"]=="dwarf"].groupby("teff_bin")["in_window"].mean().to_dict(),
    }

    with open(p_sum, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved state vector: {p_out}")
    print(f"saved summary: {p_sum}")
    print("window fraction overall:", summary["window_fraction_overall"])


if __name__ == "__main__":
    main()
