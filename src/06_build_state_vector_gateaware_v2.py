import os
import yaml
import numpy as np
import pandas as pd


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(obj: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        yaml.safe_dump(obj, f, sort_keys=False)


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
    cfg_all = load_yaml("configs/state_vector.yaml")
    cfg = cfg_all["state_vector_v2"]

    p_in = cfg["in_parquet"]
    gate_yaml = cfg["gate_yaml"]
    p_out = cfg["out_parquet"]
    p_sum = cfg["out_summary"]

    os.makedirs(os.path.dirname(p_out), exist_ok=True)

    df = pd.read_parquet(p_in)

    nm = load_yaml(gate_yaml)
    gate_map = {}
    for tb, b in nm.get("bins", {}).items():
        gate = b.get("observed", {}).get("gate_logprot", None)
        if gate is not None:
            gate_map[str(tb)] = float(gate)

    if not gate_map:
        raise RuntimeError(f"No gate_logprot entries found in {gate_yaml}")

    # Ensure log_prot exists
    if "log_prot" not in df.columns:
        df["log_prot"] = np.log10(df["prot"].astype(float))

    # Map gate by teff_bin
    df["gate_logprot"] = df["teff_bin"].astype(str).map(gate_map)

    # Drop rows without gate (bins not in nulls config)
    before = len(df)
    df = df[df["gate_logprot"].notna()].copy()
    after_gate = len(df)

    # Gate-aware regulation axis: distance from gate (post-gate positive)
    df["R_raw_v2"] = df["log_prot"] - df["gate_logprot"]

    # Decide whether to enforce post-gate membership before windowing
    require_post_gate = bool(cfg.get("require_post_gate", False))
    if require_post_gate:
        df = df[df["R_raw_v2"] >= 0].copy()

    # Normalize within the same class grouping as v1
    group_key = ["class_stage", "teff_bin"]

    # Keep the original raw proxies for H, M, S but re-normalize for consistency
    # v1 output already has H_raw/M_raw/S_raw, but recompute robust z to match the new population if post-gate filter is used.
    needed = ["H_raw", "M_raw", "S_raw"]
    for c in needed:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} in input parquet: {p_in}")

    df["R_v2"] = df.groupby(group_key)["R_raw_v2"].transform(robust_z)
    df["H_v2"] = df.groupby(group_key)["H_raw"].transform(robust_z)
    df["M_v2"] = df.groupby(group_key)["M_raw"].transform(robust_z)
    df["S_v2"] = df.groupby(group_key)["S_raw"].transform(robust_z)

    # Use the same weights as v1 if present, else all ones
    w2 = cfg.get("weights_v2", {})
    wR = float(w2.get("wR", 1.0))
    wH = float(w2.get("wH", 1.0))
    wM = float(w2.get("wM", 1.0))
    wS = float(w2.get("wS", 1.0))

    df["Phi_v2"] = wR * df["R_v2"] + wH * df["H_v2"] + wM * df["M_v2"] + wS * df["S_v2"]

    # Window per group, bounded quantiles on Phi_v2
    qlo = float(cfg["window"]["q_lo"])
    qhi = float(cfg["window"]["q_hi"])

    df["Phi_v2_lo"] = df.groupby(group_key)["Phi_v2"].transform(lambda x: x.quantile(qlo))
    df["Phi_v2_hi"] = df.groupby(group_key)["Phi_v2"].transform(lambda x: x.quantile(qhi))
    df["in_window_v2"] = (df["Phi_v2"] >= df["Phi_v2_lo"]) & (df["Phi_v2"] <= df["Phi_v2_hi"])

    # Write output (keep v1 columns too, so you can compare)
    out_cols = [
        "source_id", "teff_gspphot", "teff_bin", "class_stage",
        "prot", "log_prot", "gate_logprot",
        "H_raw", "M_raw", "S_raw",
        "R_raw_v2", "R_v2", "H_v2", "M_v2", "S_v2",
        "Phi_v2", "Phi_v2_lo", "Phi_v2_hi", "in_window_v2",
        # bring along v1 if present
        "R_raw", "R", "H", "M", "S", "Phi", "in_window"
    ]
    out_cols = [c for c in out_cols if c in df.columns]
    df[out_cols].to_parquet(p_out, index=False)

    # Summary
    summary = {
        "input_rows": int(before),
        "rows_with_gate": int(after_gate),
        "rows_final": int(len(df)),
        "require_post_gate": require_post_gate,
        "window_fraction_overall_v2": float(df["in_window_v2"].mean()),
        "by_teff_bin_dwarfs_v2": (
            df[df["class_stage"] == "dwarf"]
              .groupby("teff_bin")["in_window_v2"].mean().to_dict()
        ),
        "gate_bins_used": sorted(gate_map.keys()),
    }
    save_yaml(summary, p_sum)

    print(f"saved v2 state vector: {p_out}")
    print(f"saved v2 summary: {p_sum}")
    print("window fraction overall v2:", summary["window_fraction_overall_v2"])
    if require_post_gate:
        print("note: post-gate filter ON, Phi window computed within post-gate subset")


if __name__ == "__main__":
    main()
