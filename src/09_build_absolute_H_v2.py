import os
import yaml
import numpy as np
import pandas as pd


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


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
        q1 = x.quantile(0.25)
        q3 = x.quantile(0.75)
        s = (q3 - q1) / 1.349 if (q3 > q1) else np.nan
    if not np.isfinite(s) or s == 0:
        return pd.Series(np.nan, index=x.index)
    return (x - m) / s


def main():
    cfg_all = load_cfg("configs/state_vector.yaml")
    cfg = cfg_all["absH_v2"]

    p_in = cfg["in_parquet"]
    p_aux = cfg["aux_parquet_v1"]
    p_out = cfg["out_parquet"]
    p_sum = cfg["out_summary"]
    ensure_dir(os.path.dirname(p_out))
    ensure_dir(os.path.dirname(p_sum))

    df = pd.read_parquet(p_in).copy()

    # Bring in parallax and Gmag if missing (v2 parquet is compact)
    need_aux = []
    for col in ["parallax", "phot_g_mean_mag"]:
        if col not in df.columns:
            need_aux.append(col)

    if need_aux:
        aux_cols = ["source_id"] + need_aux
        aux = pd.read_parquet(p_aux, columns=aux_cols).copy()
        before = len(df)
        df = df.merge(aux, on="source_id", how="left", validate="m:1")
        missing_after = df[need_aux].isna().mean().to_dict()
        print(f"merged aux cols {need_aux} from: {p_aux}")
        print(f"rows: {before} -> {len(df)} ; missing fractions after merge:", missing_after)

    # Require fields now
    for c in ["source_id", "parallax", "phot_g_mean_mag", "class_stage", "teff_bin", "R_v2", "M_v2", "S_v2", "R_raw_v2"]:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} after merge. Check aux_parquet_v1 path and schema.")

    # Parallax sanity (mas)
    plx_min = float(cfg.get("parallax_mas_min", 0.2))
    df["parallax"] = df["parallax"].astype(float)
    df = df[df["parallax"] > plx_min].copy()

    # Absolute magnitude:
    # M_G = G + 5*log10(parallax_mas) - 10
    plx = df["parallax"].to_numpy(dtype=float)
    G = df["phot_g_mean_mag"].to_numpy(dtype=float)
    df["M_G"] = G + 5.0 * np.log10(plx) - 10.0

    # Forcing proxy: H_abs = -M_G
    df["H_abs_raw"] = -df["M_G"]

    # Normalize within grouping
    group_key = ["class_stage", "teff_bin"]
    df["H_abs_v2"] = df.groupby(group_key)["H_abs_raw"].transform(robust_z)

    # Phi with absolute-H
    w = cfg["weights"]
    wR = float(w["wR"])
    wH = float(w["wH"])
    wM = float(w["wM"])
    wS = float(w["wS"])

    df["Phi_absH"] = wR * df["R_v2"] + wH * df["H_abs_v2"] + wM * df["M_v2"] + wS * df["S_v2"]

    # Window per group
    qlo = float(cfg["window"]["q_lo"])
    qhi = float(cfg["window"]["q_hi"])
    df["Phi_absH_lo"] = df.groupby(group_key)["Phi_absH"].transform(lambda x: x.quantile(qlo))
    df["Phi_absH_hi"] = df.groupby(group_key)["Phi_absH"].transform(lambda x: x.quantile(qhi))
    df["in_window_absH"] = (df["Phi_absH"] >= df["Phi_absH_lo"]) & (df["Phi_absH"] <= df["Phi_absH_hi"])

    # Save
    keep = [
        "source_id", "class_stage", "teff_bin",
        "parallax", "phot_g_mean_mag", "M_G",
        "R_raw_v2", "R_v2",
        "H_abs_raw", "H_abs_v2",
        "M_v2", "S_v2",
        "Phi_absH", "Phi_absH_lo", "Phi_absH_hi", "in_window_absH",
        # keep original Phi_v2 if present
        "Phi_v2", "in_window_v2"
    ]
    keep = [c for c in keep if c in df.columns]
    df[keep].to_parquet(p_out, index=False)

    summary = {
        "input_rows_v2": int(pd.read_parquet(p_in, columns=["source_id"]).shape[0]),
        "rows_after_merge": int(df.shape[0]),
        "parallax_mas_min": plx_min,
        "weights": {"wR": wR, "wH": wH, "wM": wM, "wS": wS},
        "window": {"q_lo": qlo, "q_hi": qhi},
        "window_fraction_overall": float(df["in_window_absH"].mean()),
        "by_teff_bin_dwarfs": (
            df[df["class_stage"] == "dwarf"].groupby("teff_bin")["in_window_absH"].mean().to_dict()
        ),
    }

    with open(p_sum, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {p_out}")
    print(f"summary: {p_sum}")
    print("window frac:", summary["window_fraction_overall"])


if __name__ == "__main__":
    main()
