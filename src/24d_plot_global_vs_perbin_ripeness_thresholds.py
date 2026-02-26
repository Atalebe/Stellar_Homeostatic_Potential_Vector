import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
PERBIN_YAML = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_window_perbin_summary.yaml"

OUT_DIR = "/mnt/g/STAR_HPV/results/figures/ripeness_thresholds"
OUT_TEX = f"{OUT_DIR}/figures_ripeness_thresholds.tex"

BINS = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]

def ensure_dir(p): os.makedirs(p, exist_ok=True)

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def main():
    ensure_dir(OUT_DIR)

    df = pd.read_parquet(INP)
    df["source_id"] = norm_sid(df["source_id"])

    w = df[df["in_window_gmm"].astype(bool)].copy()
    w["prot_days"] = pd.to_numeric(w["prot_days"], errors="coerce")
    w["Phi_gmm"] = pd.to_numeric(w["Phi_gmm"], errors="coerce")
    w = w.dropna(subset=["prot_days", "Phi_gmm"]).copy()
    w["ripeness_gmm"] = w["prot_days"] * w["Phi_gmm"]

    # global q90 within window
    global_q90 = float(w["ripeness_gmm"].quantile(0.90))
    global_q95 = float(w["ripeness_gmm"].quantile(0.95))

    with open(PERBIN_YAML, "r") as f:
        per = yaml.safe_load(f)

    figs = []
    for b in BINS:
        wb = w[w["teff_bin_s"].astype(str) == b].copy()
        if len(wb) == 0:
            continue

        perbin_q90 = per["by_bin"][b]["tails"]["q90"]["ripeness_threshold_bin"]
        perbin_q95 = per["by_bin"][b]["tails"]["q95"]["ripeness_threshold_bin"]

        x = wb["ripeness_gmm"].to_numpy()
        x = x[np.isfinite(x)]

        plt.figure()
        plt.hist(x, bins=30)
        plt.axvline(perbin_q90)
        plt.axvline(perbin_q95)
        plt.axvline(global_q90, linestyle="--")
        plt.axvline(global_q95, linestyle="--")
        plt.title(f"Ripeness in-window: {b} (N={len(wb)})")
        plt.xlabel("ripeness_gmm = Phi_gmm * Prot_days")
        plt.ylabel("count")

        out_png = f"{OUT_DIR}/ripeness_thresholds_{b.replace('-','_')}.png"
        plt.savefig(out_png, dpi=200, bbox_inches="tight")
        plt.close()
        figs.append(out_png)

    # LaTeX snippet
    lines = []
    lines.append(r"\FindingBlock{Global vs per-bin ripeness thresholds within the window}{")
    lines.append(r"The dashed lines show global $q_{90}$/$q_{95}$ thresholds computed across all in-window stars; solid lines show per-bin $q_{90}$/$q_{95}$ thresholds. The separation demonstrates why global tails can exclude hotter bins despite clear per-bin tails.}")
    for p in figs:
        base = os.path.basename(p)
        lines.append(r"\begin{figure}[!ht]")
        lines.append(r"\centering")
        lines.append(rf"\includegraphics[width=0.85\linewidth]{{{OUT_DIR}/{base}}}")
        lines.append(rf"\caption{{Ripeness distribution within the in-window cohort for Teff bin {base.replace('ripeness_thresholds_','').replace('.png','').replace('_','-')}. Solid: per-bin thresholds. Dashed: global thresholds.}}")
        lines.append(r"\end{figure}")

    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("saved figures to:", OUT_DIR)
    print("saved LaTeX snippet:", OUT_TEX)

if __name__ == "__main__":
    main()
