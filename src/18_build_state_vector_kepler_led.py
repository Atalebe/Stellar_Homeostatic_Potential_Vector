import os, glob, yaml
import numpy as np
import pandas as pd

MCQ   = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"
IDMAP = "/mnt/g/STAR_HPV/interim/kepler/kepler_gaia_ids.csv"
PARTS = "/mnt/g/STAR_HPV/raw/gaia_dr3/kepler_ids_parts/*.parquet"

OUT = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector.parquet"
SUM = "/mnt/g/STAR_HPV/results/state_vectors/summary_kepler_led.yaml"

BINS = [3200, 4000, 5200, 6000, 7500]
LABELS = ["3200-4000","4000-5200","5200-6000","6000-7500"]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUM), exist_ok=True)

    mcq = pd.read_parquet(MCQ, columns=["kic","prot_days","Teff"]).copy()
    mcq["kic"] = pd.to_numeric(mcq["kic"], errors="coerce").astype("Int64")
    mcq["prot_days"] = pd.to_numeric(mcq["prot_days"], errors="coerce")
    mcq["Teff"] = pd.to_numeric(mcq["Teff"], errors="coerce")
    mcq = mcq.dropna(subset=["kic","prot_days","Teff"]).copy()
    mcq = mcq[mcq["prot_days"] > 0].copy()

    idmap = pd.read_csv(IDMAP, dtype={"source_id":"string"}).copy()
    idmap = idmap[["kic","source_id"]].copy()
    idmap["kic"] = pd.to_numeric(idmap["kic"], errors="coerce").astype("Int64")
    idmap["source_id"] = idmap["source_id"].astype("string").str.strip().str.replace(r"\.0$","",regex=True)
    idmap["source_id"] = pd.to_numeric(idmap["source_id"], errors="coerce").astype("Int64")
    idmap = idmap.dropna(subset=["kic","source_id"]).drop_duplicates(subset=["kic"]).copy()

    km = mcq.merge(idmap, on="kic", how="inner")
    km = km.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).copy()

    files = sorted(glob.glob(PARTS))
    if not files:
        raise RuntimeError(f"No Gaia parts found at {PARTS}")
    g = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)

    g["source_id"] = pd.to_numeric(g["source_id"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).copy()

    df = g.merge(km[["source_id","kic","prot_days","Teff"]], on="source_id", how="inner", validate="1:1")

    # Teff bins (categorical)
    df["teff_bin"] = pd.cut(df["Teff"], bins=BINS, labels=LABELS, right=False)

    # Absolute magnitude
    plx = pd.to_numeric(df["parallax"], errors="coerce").astype(float)
    Gmag = pd.to_numeric(df["phot_g_mean_mag"], errors="coerce").astype(float)
    df = df[np.isfinite(plx) & np.isfinite(Gmag) & (plx > 0)].copy()
    df["M_G"] = Gmag + 5.0*np.log10(plx) - 10.0

    # Guard again before log10
    df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
    df = df[df["prot_days"] > 0].copy()
    df["log_prot"] = np.log10(df["prot_days"].astype(float))

    # Gate center proxy: per-bin median logP
    gate = df.groupby("teff_bin")["log_prot"].median()
    df["gate_logprot"] = df["teff_bin"].map(gate)

    # Critical: ensure numeric dtype
    df["gate_logprot"] = pd.to_numeric(df["gate_logprot"], errors="coerce")

    # Post-gate
    df["R_raw"] = df["log_prot"] - df["gate_logprot"]
    df = df[df["R_raw"] >= 0].copy()

    # H = -M_G
    df["H_raw"] = -df["M_G"]

    def rz(x):
        m = x.median()
        mad = (x - m).abs().median()
        if mad == 0 or not np.isfinite(mad):
            return x*0 + np.nan
        return (x - m) / mad

    df["R"] = df.groupby("teff_bin")["R_raw"].transform(rz)
    df["H"] = df.groupby("teff_bin")["H_raw"].transform(rz)

    # Locked weights (1,1,0,0)
    df["Phi"] = df["R"] + df["H"]

    # Baseline window q0.35-0.65
    qlo, qhi = 0.35, 0.65
    df["Phi_lo"] = df.groupby("teff_bin")["Phi"].transform(lambda x: x.quantile(qlo))
    df["Phi_hi"] = df.groupby("teff_bin")["Phi"].transform(lambda x: x.quantile(qhi))
    df["in_window"] = (df["Phi"] >= df["Phi_lo"]) & (df["Phi"] <= df["Phi_hi"])

    df.to_parquet(OUT, index=False)

    summary = {
        "rows_final": int(len(df)),
        "rows_by_bin": df.groupby("teff_bin").size().to_dict(),
        "window_frac_by_bin": df.groupby("teff_bin")["in_window"].mean().to_dict(),
        "gate_logprot_by_bin": {str(k): float(v) for k, v in gate.dropna().to_dict().items()},
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", OUT)
    print("summary:", SUM)
    print(summary)

if __name__=="__main__":
    main()
