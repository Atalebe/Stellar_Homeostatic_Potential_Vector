import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def quantile_clip(arr: np.ndarray, qlo: float, qhi: float) -> np.ndarray:
    lo = np.nanquantile(arr, qlo)
    hi = np.nanquantile(arr, qhi)
    return np.clip(arr, lo, hi)


def binned_median(x: np.ndarray, y: np.ndarray, nbins: int, min_count: int):
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nbins, np.nan)
    cnt = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        cnt[i] = int(m.sum())
        if cnt[i] >= min_count:
            med[i] = np.nanmedian(y[m])

    return centers, med, cnt


def gate_stats(log_prot: np.ndarray, arot: np.ndarray, nbins: int, min_count: int, qlo: float, qhi: float):
    y = quantile_clip(arot, qlo, qhi)
    x = log_prot

    centers, med, cnt = binned_median(x, y, nbins=nbins, min_count=min_count)
    m = np.isfinite(med)

    if m.sum() < 6:
        return {"gate_logprot": None, "min_slope": None, "n_valid_bins": int(m.sum())}

    # slope of median curve
    dmed = np.gradient(med[m], centers[m])
    j = int(np.nanargmin(dmed))  # most negative slope
    return {
        "gate_logprot": float(centers[m][j]),
        "min_slope": float(dmed[j]),
        "n_valid_bins": int(m.sum()),
    }


def main():
    cfg = load_cfg("configs/state_vector.yaml")["nulls"]
    p_in = cfg["in_parquet"]
    out_dir = cfg["out_dir"]
    ensure_dir(out_dir)

    only_dwarfs = bool(cfg.get("only_dwarfs", True))
    teff_bins = cfg["teff_bins"]
    nbins = int(cfg["nbins_logprot"])
    min_count = int(cfg["min_bin_count"])
    qlo = float(cfg["clip_qlo"])
    qhi = float(cfg["clip_qhi"])
    n_iter = int(cfg["n_iter"])
    seed = int(cfg["seed"])

    rng = np.random.default_rng(seed)

    df = pd.read_parquet(p_in)
    if only_dwarfs:
        df = df[df["class_stage"] == "dwarf"].copy()

    df = df[df["teff_bin"].astype(str).isin(teff_bins)].copy()
    df = df[np.isfinite(df["prot"].astype(float))].copy()
    df = df[np.isfinite(df["arot"].astype(float))].copy()

    df["log_prot"] = np.log10(df["prot"].astype(float))

    results = {"config": cfg, "bins": {}}

    for tb in teff_bins:
        sub = df[df["teff_bin"].astype(str) == tb].copy()
        x = sub["log_prot"].to_numpy()
        y = sub["arot"].to_numpy()

        obs = gate_stats(x, y, nbins=nbins, min_count=min_count, qlo=qlo, qhi=qhi)

        # Null A: shuffle y within bin
        nullA_gate = []
        nullA_slope = []
        for _ in range(n_iter):
            ysh = rng.permutation(y)
            st = gate_stats(x, ysh, nbins=nbins, min_count=min_count, qlo=qlo, qhi=qhi)
            nullA_gate.append(st["gate_logprot"])
            nullA_slope.append(st["min_slope"])

        # Null B: shuffle x within bin
        nullB_gate = []
        nullB_slope = []
        for _ in range(n_iter):
            xsh = rng.permutation(x)
            st = gate_stats(xsh, y, nbins=nbins, min_count=min_count, qlo=qlo, qhi=qhi)
            nullB_gate.append(st["gate_logprot"])
            nullB_slope.append(st["min_slope"])

        # p-values based on slope strength (more negative = stronger gate)
        def pval(null_slopes, obs_slope):
            ns = np.array([v for v in null_slopes if v is not None and np.isfinite(v)], dtype=float)
            if obs_slope is None or not np.isfinite(obs_slope) or ns.size == 0:
                return None
            # fraction as or more extreme (<= because more negative is stronger)
            return float((np.sum(ns <= obs_slope) + 1) / (ns.size + 1))

        pA = pval(nullA_slope, obs["min_slope"])
        pB = pval(nullB_slope, obs["min_slope"])

        results["bins"][tb] = {
            "rows": int(sub.shape[0]),
            "observed": obs,
            "nullA_shuffle_activity": {
                "p_value_slope": pA,
                "gate_logprot_mean": float(np.nanmean([v for v in nullA_gate if v is not None])),
                "slope_mean": float(np.nanmean([v for v in nullA_slope if v is not None])),
            },
            "nullB_shuffle_rotation": {
                "p_value_slope": pB,
                "gate_logprot_mean": float(np.nanmean([v for v in nullB_gate if v is not None])),
                "slope_mean": float(np.nanmean([v for v in nullB_slope if v is not None])),
            },
        }

        # Plots: histogram of null slopes + observed line
        for tag, slopes in [("nullA", nullA_slope), ("nullB", nullB_slope)]:
            ns = np.array([v for v in slopes if v is not None and np.isfinite(v)], dtype=float)
            fig = plt.figure()
            plt.hist(ns, bins=40)
            if obs["min_slope"] is not None and np.isfinite(obs["min_slope"]):
                plt.axvline(obs["min_slope"])
            plt.xlabel("min slope of median activity vs logProt (more negative = stronger gate)")
            plt.ylabel("count")
            plt.title(f"{tb} {tag}: null slope distribution")
            plt.tight_layout()
            fig.savefig(os.path.join(out_dir, f"{tb}_{tag}_slope_hist.png"), dpi=200)
            plt.close(fig)

    out_yaml = os.path.join(out_dir, "null_metrics.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(results, f, sort_keys=False)

    print(f"saved: {out_yaml}")
    print(f"plots in: {out_dir}")


if __name__ == "__main__":
    main()
