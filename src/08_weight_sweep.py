import os
import yaml
import numpy as np
import pandas as pd


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def sweep():
    cfg = load_cfg("configs/state_vector.yaml")["weight_sweep"]

    p_in = cfg["in_parquet"]
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    only_dwarfs = bool(cfg.get("only_dwarfs", True))
    teff_bins = cfg["teff_bins"]
    qlo = float(cfg["q_lo"])
    qhi = float(cfg["q_hi"])

    min_rows_bin = int(cfg["min_rows_bin"])
    min_each = int(cfg["min_rows_each_group"])

    wR_grid = [float(x) for x in cfg["wR_grid"]]
    wH_grid = [float(x) for x in cfg["wH_grid"]]
    wM_grid = [float(x) for x in cfg["wM_grid"]]
    wS_grid = [float(x) for x in cfg["wS_grid"]]

    df = pd.read_parquet(p_in)

    # This parquet is already post-gate filtered if you built it that way.
    if only_dwarfs:
        df = df[df["class_stage"] == "dwarf"].copy()

    df = df[df["teff_bin"].astype(str).isin(teff_bins)].copy()

    # Required columns from v2 builder
    need = ["teff_bin", "class_stage", "R_raw_v2", "R_v2", "H_v2", "M_v2", "S_v2"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing required column: {c}")

    # Grouping key must match v2 normalization
    group_key = ["class_stage", "teff_bin"]

    rows = []

    for wR in wR_grid:
        for wH in wH_grid:
            for wM in wM_grid:
                for wS in wS_grid:
                    # Compute Phi for these weights
                    phi = wR * df["R_v2"] + wH * df["H_v2"] + wM * df["M_v2"] + wS * df["S_v2"]

                    # Window thresholds per group
                    lo = phi.groupby([df[k] for k in group_key]).transform(lambda x: x.quantile(qlo))
                    hi = phi.groupby([df[k] for k in group_key]).transform(lambda x: x.quantile(qhi))
                    win = (phi >= lo) & (phi <= hi)

                    # Evaluate depth deltas per Teff bin
                    for tb in teff_bins:
                        sub = df[df["teff_bin"].astype(str) == tb]
                        if len(sub) < min_rows_bin:
                            continue

                        wmask = win.loc[sub.index]
                        nW = int(wmask.sum())
                        nN = int((~wmask).sum())
                        if nW < min_each or nN < min_each:
                            continue

                        a = sub.loc[wmask, "R_raw_v2"].astype(float).to_numpy()
                        b = sub.loc[~wmask, "R_raw_v2"].astype(float).to_numpy()

                        rows.append({
                            "teff_bin": tb,
                            "wR": wR, "wH": wH, "wM": wM, "wS": wS,
                            "rows": int(len(sub)),
                            "rows_window": nW,
                            "rows_nonwindow": nN,
                            "Rraw_median_window": float(np.nanmedian(a)),
                            "Rraw_median_nonwindow": float(np.nanmedian(b)),
                            "Rraw_delta_median": float(np.nanmedian(a) - np.nanmedian(b)),
                        })

    out = pd.DataFrame(rows)
    out_csv = os.path.join(out_dir, "weight_sweep.csv")
    out.to_csv(out_csv, index=False)

    # For each teff bin, keep top 10 weight sets by delta
    best = {}
    for tb in teff_bins:
        sub = out[out["teff_bin"] == tb].sort_values("Rraw_delta_median", ascending=False).head(10)
        best[tb] = sub.to_dict(orient="records")

    out_yaml = os.path.join(out_dir, "weight_sweep_best.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"best_by_teff_bin": best}, f, sort_keys=False)

    print("saved:", out_csv)
    print("saved:", out_yaml)


if __name__ == "__main__":
    sweep()
