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


def quantile_clip(s: pd.Series, qlo: float, qhi: float) -> pd.Series:
    lo = s.quantile(qlo)
    hi = s.quantile(qhi)
    return s.clip(lower=lo, upper=hi)


def binned_median(x: np.ndarray, y: np.ndarray, nbins: int):
    """Return bin centers, median(y), count per bin for x-binned y."""
    edges = np.linspace(np.nanmin(x), np.nanmax(x), nbins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    med = np.full(nbins, np.nan)
    cnt = np.zeros(nbins, dtype=int)

    for i in range(nbins):
        m = (x >= edges[i]) & (x < edges[i + 1])
        cnt[i] = int(m.sum())
        if cnt[i] > 0:
            med[i] = np.nanmedian(y[m])

    return centers, med, cnt


def main():
    cfg = load_cfg("configs/state_vector.yaml")
    g = cfg["gate"]

    p_in = g["in_parquet"]
    out_dir = g["out_dir"]
    ensure_dir(out_dir)

    df = pd.read_parquet(p_in)

    # Gate diagnostics use: prot (days) and activity proxy arot
    # Apply dwarf-only filter if requested
    if g.get("only_dwarfs", True):
        df = df[df["class_stage"] == "dwarf"].copy()

    # Must have teff_bin
    df = df[df["teff_bin"].notna()].copy()

    # Basic sanity: finite values
    df = df[np.isfinite(df["prot"].astype(float))].copy()
    df = df[np.isfinite(df["arot"].astype(float))].copy()

    # log10 rotation
    df["log_prot"] = np.log10(df["prot"].astype(float))

    # limit rotation range
    lpmin = float(g["prot_log10_min"])
    lpmax = float(g["prot_log10_max"])
    df = df[df["log_prot"].between(lpmin, lpmax)].copy()

    # clip activity tails for plotting stability
    qlo = float(g["arot_qclip_lo"])
    qhi = float(g["arot_qclip_hi"])
    df["arot_clip"] = df.groupby("teff_bin")["arot"].transform(lambda s: quantile_clip(s, qlo, qhi))

    nbins = int(g["nbins_logprot"])
    min_rows = int(g["min_rows_per_bin"])

    # overall plot
    fig = plt.figure()
    plt.hist2d(df["log_prot"], df["arot_clip"], bins=200)
    plt.xlabel("log10(P_rot / days)")
    plt.ylabel("activity proxy (clipped)  max_activity_index_g")
    plt.title("Stellar stability gate diagnostic (all selected stars)")
    plt.tight_layout()
    p = os.path.join(out_dir, "gate_all_hist2d.png")
    fig.savefig(p, dpi=200)
    plt.close(fig)

    # per Teff bin plots + metrics
    metrics = {"input_rows": int(df.shape[0]), "bins": {}}

    teff_bins = sorted(df["teff_bin"].astype(str).unique().tolist())
    for tb in teff_bins:
        sub = df[df["teff_bin"].astype(str) == tb].copy()
        n = int(sub.shape[0])
        if n < min_rows:
            continue

        x = sub["log_prot"].to_numpy()
        y = sub["arot_clip"].to_numpy()

        centers, med, cnt = binned_median(x, y, nbins=nbins)

        # Estimate a crude "gate" location: where median activity drops fastest
        # Compute gradient where both bins are populated
        m = np.isfinite(med) & (cnt > 0)
        gate = None
        if m.sum() >= 6:
            dm = np.gradient(med[m], centers[m])
            # most negative slope as "gate"
            j = int(np.nanargmin(dm))
            gate = float(centers[m][j])

        metrics["bins"][tb] = {
            "rows": n,
            "gate_logprot_est": gate,
            "logprot_range": [float(np.nanmin(x)), float(np.nanmax(x))],
            "arot_clip_range": [float(np.nanmin(y)), float(np.nanmax(y))],
        }

        # Plot per bin
        fig = plt.figure()
        plt.hist2d(x, y, bins=180)
        plt.xlabel("log10(P_rot / days)")
        plt.ylabel("activity proxy (clipped)  max_activity_index_g")
        plt.title(f"Gate diagnostic: Teff bin {tb} (N={n})")

        # Overlay median trend where available
        plt.plot(centers[m], med[m])

        # Overlay gate estimate
        if gate is not None:
            plt.axvline(gate)

        plt.tight_layout()
        out_png = os.path.join(out_dir, f"gate_teffbin_{tb.replace('/','_')}.png")
        fig.savefig(out_png, dpi=200)
        plt.close(fig)

    # Save metrics yaml
    out_yaml = os.path.join(out_dir, "gate_metrics.yaml")
    with open(out_yaml, "w") as f:
        yaml.safe_dump(metrics, f, sort_keys=False)

    print(f"saved: {out_dir}")
    print(f"plots: gate_all_hist2d.png and per-bin gate_teffbin_*.png")
    print(f"metrics: {out_yaml}")


if __name__ == "__main__":
    main()
