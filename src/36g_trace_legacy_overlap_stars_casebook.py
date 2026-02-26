import os, yaml
import numpy as np
import pandas as pd

AGES_SUBSET = "/mnt/g/STAR_HPV/processed/ages_merged/kepler_legacy_validation_subset.parquet"
KEP_GMMGATE = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector_gmmgate.parquet"
ANCHOR_PARQ = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_anchor_catalog.parquet"
TAIL_Q90    = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q90.parquet"
TAIL_Q95    = "/mnt/g/STAR_HPV/results/anchor_catalogs/kepler_ripe_tail_q95.parquet"
IDMAP_R5    = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"

OUT_DIR     = "/mnt/g/STAR_HPV/results/ripeness/casebook_legacy_overlap"
OUT_CSV     = f"{OUT_DIR}/legacy_overlap_casebook.csv"
OUT_YAML    = f"{OUT_DIR}/legacy_overlap_casebook.yaml"
OUT_TEX     = f"{OUT_DIR}/legacy_overlap_casebook.tex"

def norm_sid(s):
    return s.astype("string").str.strip().str.replace(r"\.0$", "", regex=True)

def add_pct_rank(df_full, df_cases, value_col, bin_col, out_col):
    tmp = df_full[["source_id", bin_col, value_col]].copy()
    tmp["source_id"] = norm_sid(tmp["source_id"])
    tmp[value_col] = pd.to_numeric(tmp[value_col], errors="coerce")
    tmp = tmp.dropna(subset=[value_col])
    tmp[out_col] = tmp.groupby(bin_col)[value_col].rank(pct=True)
    tmp = tmp[["source_id", out_col]]
    return df_cases.merge(tmp, on="source_id", how="left")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # 3-star set (has ages)
    sub = pd.read_parquet(AGES_SUBSET)
    sub["source_id"] = norm_sid(sub["source_id"])
    sids = sub["source_id"].dropna().unique().tolist()
    if len(sids) == 0:
        raise RuntimeError("No rows in legacy validation subset. Check AGES_SUBSET path.")

    # main kepler gated table
    kep = pd.read_parquet(KEP_GMMGATE)
    kep["source_id"] = norm_sid(kep["source_id"])

    # required columns present in your table
    required = ["source_id", "teff_bin_s", "prot_days", "Phi_gmm", "in_window_gmm"]
    missing = [c for c in required if c not in kep.columns]
    if missing:
        raise RuntimeError(f"Missing columns in {KEP_GMMGATE}: {missing}. Found: {list(kep.columns)[:60]}")

    # compute ripeness in-place (consistent operational definition)
    kep["ripeness_gmm"] = pd.to_numeric(kep["Phi_gmm"], errors="coerce") * pd.to_numeric(kep["prot_days"], errors="coerce")

    # load anchor and tails (membership sets)
    anc = pd.read_parquet(ANCHOR_PARQ); anc["source_id"] = norm_sid(anc["source_id"])
    q90 = pd.read_parquet(TAIL_Q90);    q90["source_id"] = norm_sid(q90["source_id"])
    q95 = pd.read_parquet(TAIL_Q95);    q95["source_id"] = norm_sid(q95["source_id"])

    anc_set = set(anc["source_id"].tolist())
    q90_set = set(q90["source_id"].tolist())
    q95_set = set(q95["source_id"].tolist())

    # mapping to KIC + match-quality
    mp = pd.read_csv(IDMAP_R5, dtype={"source_id":"string"}, low_memory=False)
    mp = mp[mp["status"].astype(str) == "ok"].copy()
    mp["source_id"] = norm_sid(mp["source_id"])
    mp["kic"] = pd.to_numeric(mp["kic"], errors="coerce").astype("Int64")

    # extract the 3 stars from kepler gated table
    k3 = kep[kep["source_id"].isin(sids)].copy()
    k3 = k3.merge(mp[["source_id","kic","dist_deg","ruwe","phot_g_mean_mag"]], on="source_id", how="left")

    # merge ages (so we always have age columns even if sub had extra fields)
    ages = sub[["source_id"] + [c for c in ["age_gyr","age_gyr_err"] if c in sub.columns]].copy()
    k3 = k3.merge(ages, on="source_id", how="left")

    # membership flags
    k3["in_anchor_catalog"] = k3["source_id"].isin(anc_set)
    k3["in_tail_q90"] = k3["source_id"].isin(q90_set)
    k3["in_tail_q95"] = k3["source_id"].isin(q95_set)

    # percentile ranks within bin (computed from full kep table)
    k3 = add_pct_rank(kep, k3, "Phi_gmm", "teff_bin_s", "phi_pct_in_bin")
    k3 = add_pct_rank(kep, k3, "prot_days", "teff_bin_s", "prot_pct_in_bin")
    k3 = add_pct_rank(kep, k3, "ripeness_gmm", "teff_bin_s", "ripeness_pct_in_bin")

    # curated output columns
    keep = [
        "source_id","kic","teff_bin_s",
        "age_gyr","age_gyr_err",
        "prot_days","prot_pct_in_bin",
        "Phi_gmm","phi_pct_in_bin",
        "ripeness_gmm","ripeness_pct_in_bin",
        "in_window_gmm",
        "in_anchor_catalog","in_tail_q90","in_tail_q95",
        "dist_deg","ruwe","phot_g_mean_mag",
        "gate_days_gmm","gate_logP_gmm",
    ]
    keep = [c for c in keep if c in k3.columns]
    casebook = k3[keep].sort_values(["teff_bin_s","ripeness_gmm"], ascending=[True, False]).copy()

    casebook.to_csv(OUT_CSV, index=False)

    summary = {
        "n_case_stars": int(len(casebook)),
        "source_ids": sids,
        "paths": {
            "kepler_gmmgate": KEP_GMMGATE,
            "ages_subset": AGES_SUBSET,
            "anchor": ANCHOR_PARQ,
            "tail_q90": TAIL_Q90,
            "tail_q95": TAIL_Q95,
        },
        "notes": [
            "ripeness_gmm computed as Phi_gmm * prot_days (operational definition).",
            "in_window_gmm taken from kepler_led_state_vector_gmmgate table.",
            "tail membership based on exported tail catalogs (q90/q95)."
        ],
        "membership_counts": {
            "in_window_gmm": int(casebook["in_window_gmm"].sum()) if "in_window_gmm" in casebook.columns else None,
            "in_anchor_catalog": int(casebook["in_anchor_catalog"].sum()),
            "in_tail_q90": int(casebook["in_tail_q90"].sum()),
            "in_tail_q95": int(casebook["in_tail_q95"].sum()),
        },
    }
    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    # LaTeX snippet (compact, no tables)
    lines = []
    lines.append(r"\FindingBlock{Case study: asteroseismic-age overlap targets traced through the pipeline}{")
    lines.append(rf"The Kepler LEGACY age list overlaps the Kepler--Gaia GMM-gated rotator sample at $N={len(casebook)}$ targets.")
    lines.append(r"Each overlapping target was traced through post-gate selection, $\Phi_\*$ windowing, and ripeness ranking within its temperature bin.")
    for _, r in casebook.iterrows():
        sid = str(r["source_id"])
        kic = str(r.get("kic","NA"))
        b = str(r.get("teff_bin_s","NA"))
        age = r.get("age_gyr", np.nan)
        prot = r.get("prot_days", np.nan)
        phi = r.get("Phi_gmm", np.nan)
        rip = r.get("ripeness_gmm", np.nan)
        inw = bool(r.get("in_window_gmm", False))
        in90 = bool(r.get("in_tail_q90", False))
        in95 = bool(r.get("in_tail_q95", False))
        lines.append(
            rf"\texttt{{source\_id={sid}}} (KIC {kic}, {b}): "
            rf"age={age:.3f} Gyr, $P_{{\rm rot}}$={prot:.3f} d, $\Phi_{{\rm gmm}}$={phi:.3f}, "
            rf"$R^{{\rm ripe}}_{{\rm gmm}}$={rip:.3f}; in-window={inw}, tail$_{{90}}$={in90}, tail$_{{95}}$={in95}."
        )
    lines.append(r"}")
    with open(OUT_TEX, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("saved:", OUT_CSV)
    print("saved:", OUT_YAML)
    print("saved:", OUT_TEX)
    print(casebook)

if __name__ == "__main__":
    main()
