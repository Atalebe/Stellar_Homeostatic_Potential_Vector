import os, yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector.parquet"
GMM = "/mnt/g/STAR_HPV/results/gate/kepler_gmm_gate.yaml"

OUT = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
SUM = "/mnt/g/STAR_HPV/results/state_vectors/summary_kepler_led_gmmgate.yaml"

# Locked settings (matches your reviewer-backed baseline)
QLO, QHI = 0.35, 0.65
WEIGHTS = dict(wR=1.0, wH=1.0)   # M,S omitted by design here

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUM), exist_ok=True)

    df = pd.read_parquet(INP).copy()
    with open(GMM, "r") as f:
        g = yaml.safe_load(f)

    gate_days = {k: v["gate_days_mid"] for k, v in g.get("bins", {}).items()}
    if not gate_days:
        raise RuntimeError("No gate_days_mid found in GMM yaml")

    # Ensure teff_bin string keys
    df["teff_bin_s"] = df["teff_bin"].astype(str)
    df["gate_days_gmm"] = df["teff_bin_s"].map(gate_days)
    df["gate_logP_gmm"] = np.log10(df["gate_days_gmm"].astype(float))

    # Clean prot
    df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
    df = df[df["prot_days"] > 0].copy()
    df["log_prot"] = np.log10(df["prot_days"].astype(float))

    # New post-gate definition
    df["R_raw_gmm"] = df["log_prot"] - df["gate_logP_gmm"]
    df = df[df["R_raw_gmm"] >= 0].copy()

    # Robust z within bin
    def rz(x):
        m = x.median()
        mad = (x - m).abs().median()
        if mad == 0 or not np.isfinite(mad):
            return x*0 + np.nan
        return (x - m) / mad

    df["R_gmm"] = df.groupby("teff_bin_s")["R_raw_gmm"].transform(rz)
    df["H_z"] = df.groupby("teff_bin_s")["H_raw"].transform(rz)

    # Phi and window
    df["Phi_gmm"] = WEIGHTS["wR"] * df["R_gmm"] + WEIGHTS["wH"] * df["H_z"]
    df["Phi_lo"] = df.groupby("teff_bin_s")["Phi_gmm"].transform(lambda x: x.quantile(QLO))
    df["Phi_hi"] = df.groupby("teff_bin_s")["Phi_gmm"].transform(lambda x: x.quantile(QHI))
    df["in_window_gmm"] = (df["Phi_gmm"] >= df["Phi_lo"]) & (df["Phi_gmm"] <= df["Phi_hi"])

    df.to_parquet(OUT, index=False)

    summary = {
        "rows_final": int(len(df)),
        "rows_by_bin": df.groupby("teff_bin_s").size().to_dict(),
        "window_frac_by_bin": df.groupby("teff_bin_s")["in_window_gmm"].mean().to_dict(),
        "gate_days_mid": gate_days,
        "window": {"q_lo": QLO, "q_hi": QHI},
        "weights": WEIGHTS,
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", OUT)
    print("summary:", SUM)
    print(summary)

if __name__ == "__main__":
    main()
