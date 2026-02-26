import os, yaml
import numpy as np
import pandas as pd

GAIA_IN = "/mnt/g/STAR_HPV/raw/gaia_dr3/gaia_legacy_ids.parquet"
AGES_IN = "/mnt/g/STAR_HPV/raw/ages/kepler_ages_sourceid_agegyr.csv"
LEGACY_COORDS = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
KIC2GAIA = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"
KEP_GMM = "/mnt/g/STAR_HPV/results/gate/kepler_gmm_gate.yaml"
MCQ = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"

OUT = "/mnt/g/STAR_HPV/processed/state_vectors/legacy_led_state_vector.parquet"
SUM = "/mnt/g/STAR_HPV/results/state_vectors/summary_legacy_led.yaml"

BINS = [3200, 4000, 5200, 6000, 7500]
LABELS = ["3200-4000","4000-5200","5200-6000","6000-7500"]
QLO, QHI = 0.35, 0.65

def pick_col(df, base):
    # try base, base_x, base_y
    for c in [base, f"{base}_x", f"{base}_y"]:
        if c in df.columns:
            return c
    return None

def rz(x):
    x = pd.to_numeric(x, errors="coerce")
    m = x.median()
    mad = (x - m).abs().median()
    if mad == 0 or not np.isfinite(mad):
        return x*0 + np.nan
    return (x - m) / mad

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    os.makedirs(os.path.dirname(SUM), exist_ok=True)

    g = pd.read_parquet(GAIA_IN).copy()
    g["source_id"] = pd.to_numeric(g["source_id"], errors="coerce").astype("Int64")
    g = g.dropna(subset=["source_id"]).drop_duplicates(subset=["source_id"]).copy()

    ages = pd.read_csv(AGES_IN, dtype={"source_id":"string"}).copy()
    ages["source_id"] = ages["source_id"].astype("string").str.strip().str.replace(r"\.0$","",regex=True)
    ages["source_id"] = pd.to_numeric(ages["source_id"], errors="coerce").astype("Int64")
    ages["age_gyr"] = pd.to_numeric(ages["age_gyr"], errors="coerce")
    ages["age_gyr_err"] = pd.to_numeric(ages.get("age_gyr_err"), errors="coerce")
    ages = ages.dropna(subset=["source_id","age_gyr"]).drop_duplicates(subset=["source_id"]).copy()

    leg = pd.read_parquet(LEGACY_COORDS).copy()
    leg["kic"] = pd.to_numeric(leg["kic"], errors="coerce").astype("Int64")
    leg["Teff"] = pd.to_numeric(leg.get("Teff"), errors="coerce")

    mcq = pd.read_parquet(MCQ, columns=["kic","prot_days"]).copy()
    mcq["kic"] = pd.to_numeric(mcq["kic"], errors="coerce").astype("Int64")
    mcq["prot_days"] = pd.to_numeric(mcq["prot_days"], errors="coerce")
    mcq = mcq.dropna(subset=["kic","prot_days"]).copy()

    idmap = pd.read_csv(KIC2GAIA).copy()
    idmap["kic"] = pd.to_numeric(idmap["kic"], errors="coerce").astype("Int64")
    idmap["source_id"] = pd.to_numeric(idmap["source_id"], errors="coerce").astype("Int64")
    idmap = idmap.dropna(subset=["kic","source_id"]).drop_duplicates(subset=["kic"]).copy()

    base = idmap.merge(leg[["kic","Teff"]], on="kic", how="left")
    base = base.merge(mcq, on="kic", how="left")
    base = base.merge(ages[["source_id","age_gyr","age_gyr_err","age_source","age_quality_flag"]], on="source_id", how="left")

    df = g.merge(base, on="source_id", how="inner")

    # Resolve Gaia columns
    c_plx = pick_col(df, "parallax")
    c_G = pick_col(df, "phot_g_mean_mag")
    c_teff_gaia = pick_col(df, "teff_gspphot")

    if c_plx is None or c_G is None:
        raise RuntimeError(f"Missing Gaia parallax/Gmag in merged df. Columns: {list(df.columns)}")

    # Prot
    df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
    df = df.dropna(subset=["prot_days"]).copy()
    df = df[df["prot_days"] > 0].copy()

    # Teff for binning
    teff = pd.to_numeric(df["Teff"], errors="coerce")
    if c_teff_gaia is not None:
        teff = teff.fillna(pd.to_numeric(df[c_teff_gaia], errors="coerce"))
    df["Teff_use"] = teff
    df["teff_bin_s"] = pd.cut(df["Teff_use"], bins=BINS, labels=LABELS, right=False).astype(str)

    # Absolute magnitude and H_raw
    plx = pd.to_numeric(df[c_plx], errors="coerce").astype(float)
    Gmag = pd.to_numeric(df[c_G], errors="coerce").astype(float)
    df = df[np.isfinite(plx) & np.isfinite(Gmag) & (plx > 0)].copy()
    df["M_G"] = Gmag + 5.0*np.log10(plx) - 10.0
    df["H_raw"] = -df["M_G"]

    # Gate days midpoints
    with open(KEP_GMM, "r") as f:
        gmm = yaml.safe_load(f)
    gate_days = {k: v["gate_days_mid"] for k, v in gmm.get("bins", {}).items()}
    df["gate_days_gmm"] = df["teff_bin_s"].map(gate_days)
    df = df.dropna(subset=["gate_days_gmm"]).copy()
    df["gate_logP_gmm"] = np.log10(df["gate_days_gmm"].astype(float))

    df["log_prot"] = np.log10(df["prot_days"].astype(float))
    df["R_raw_gmm"] = df["log_prot"] - df["gate_logP_gmm"]

    # Post-gate
    df = df[df["R_raw_gmm"] >= 0].copy()

    # Normalize and window
    df["R_gmm"] = df.groupby("teff_bin_s")["R_raw_gmm"].transform(rz)
    df["H_z"] = df.groupby("teff_bin_s")["H_raw"].transform(rz)
    df["Phi_gmm"] = df["R_gmm"] + df["H_z"]

    df["Phi_lo"] = df.groupby("teff_bin_s")["Phi_gmm"].transform(lambda x: x.quantile(QLO))
    df["Phi_hi"] = df.groupby("teff_bin_s")["Phi_gmm"].transform(lambda x: x.quantile(QHI))
    df["in_window_gmm"] = (df["Phi_gmm"] >= df["Phi_lo"]) & (df["Phi_gmm"] <= df["Phi_hi"])

    df["ripeness_rot"] = df["Phi_gmm"] * df["prot_days"]

    df.to_parquet(OUT, index=False)

    summary = {
        "gaia_rows_in": int(len(g)),
        "rows_final": int(len(df)),
        "rows_with_age": int(df["age_gyr"].notna().sum()),
        "rows_by_bin": df.groupby("teff_bin_s").size().to_dict(),
        "rows_by_bin_with_age": df[df["age_gyr"].notna()].groupby("teff_bin_s").size().to_dict(),
        "window_frac_by_bin": df.groupby("teff_bin_s")["in_window_gmm"].mean().to_dict(),
        "out": OUT,
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("saved:", OUT)
    print("summary:", SUM)
    print(summary)

if __name__ == "__main__":
    main()
