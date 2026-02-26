import os
import yaml
import numpy as np
import pandas as pd


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def run_one(df: pd.DataFrame, teff_bins, qlo, qhi, grids, mins):
    only_dwarfs = True
    if only_dwarfs:
        df = df[df["class_stage"] == "dwarf"].copy()

    df = df[df["teff_bin"].astype(str).isin(teff_bins)].copy()

    # Choose H field: use H_abs_v2 if present, else H_v2
    Hcol = "H_abs_v2" if "H_abs_v2" in df.columns else "H_v2"

    need = ["R_raw_v2", "R_v2", Hcol, "M_v2", "S_v2", "class_stage", "teff_bin"]
    for c in need:
        if c not in df.columns:
            raise RuntimeError(f"Missing {c} for sweep")

    group_key = ["class_stage", "teff_bin"]

    rows = []
    for wR in grids["wR"]:
        for wH in grids["wH"]:
            for wM in grids["wM"]:
                for wS in grids["wS"]:
                    phi = wR*df["R_v2"] + wH*df[Hcol] + wM*df["M_v2"] + wS*df["S_v2"]
                    lo = phi.groupby([df[k] for k in group_key]).transform(lambda x: x.quantile(qlo))
                    hi = phi.groupby([df[k] for k in group_key]).transform(lambda x: x.quantile(qhi))
                    win = (phi >= lo) & (phi <= hi)

                    for tb in teff_bins:
                        sub = df[df["teff_bin"].astype(str) == tb]
                        if len(sub) < mins["min_rows_bin"]:
                            continue
                        wmask = win.loc[sub.index]
                        nW = int(wmask.sum())
                        nN = int((~wmask).sum())
                        if nW < mins["min_each"] or nN < mins["min_each"]:
                            continue

                        a = sub.loc[wmask, "R_raw_v2"].astype(float).to_numpy()
                        b = sub.loc[~wmask, "R_raw_v2"].astype(float).to_numpy()
                        delta = float(np.nanmedian(a) - np.nanmedian(b))

                        rows.append({
                            "teff_bin": tb,
                            "q_lo": qlo, "q_hi": qhi,
                            "wR": wR, "wH": wH, "wM": wM, "wS": wS,
                            "rows": int(len(sub)),
                            "rows_window": nW,
                            "Rraw_delta_median": delta,
                        })

    out = pd.DataFrame(rows)
    return out


def main():
    cfg = load_cfg("configs/state_vector.yaml")["quantile_grid_sweep"]
    p_in = cfg["in_parquet"]
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    df = pd.read_parquet(p_in)

    teff_bins = cfg["teff_bins"]
    mins = {
        "min_rows_bin": int(cfg["min_rows_bin"]),
        "min_each": int(cfg["min_rows_each_group"]),
    }

    grids = {
        "wR": [float(x) for x in cfg["wR_grid"]],
        "wH": [float(x) for x in cfg["wH_grid"]],
        "wM": [float(x) for x in cfg["wM_grid"]],
        "wS": [float(x) for x in cfg["wS_grid"]],
    }

    windows = cfg["windows"]

    all_out = []
    best = {}

    for w in windows:
        qlo = float(w["q_lo"])
        qhi = float(w["q_hi"])
        res = run_one(df, teff_bins, qlo, qhi, grids, mins)
        all_out.append(res)

        # best per teff bin for this window
        key = f"q{qlo:.2f}_{qhi:.2f}"
        best[key] = {}
        for tb in teff_bins:
            sub = res[res["teff_bin"] == tb].sort_values("Rraw_delta_median", ascending=False).head(10)
            best[key][tb] = sub.to_dict(orient="records")

    out = pd.concat(all_out, ignore_index=True)
    out_csv = os.path.join(out_dir, "quantile_grid_sweep.csv")
    out.to_csv(out_csv, index=False)

    out_yaml = os.path.join(out_dir, "quantile_grid_best.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump({"best": best}, f, sort_keys=False)

    print("saved:", out_csv)
    print("saved:", out_yaml)


if __name__ == "__main__":
    main()
