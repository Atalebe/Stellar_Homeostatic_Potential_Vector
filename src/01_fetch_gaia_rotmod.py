import os
import yaml
import pyvo
import pandas as pd


def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_cfg("configs/gaia_dr3.yaml")["gaia"]
    tap_url = cfg["tap_url"]
    lang = cfg.get("language", "PostgreSQL")

    out_dir = cfg["out_dir"]
    out_merged = cfg["out_merged"]
    os.makedirs(out_dir, exist_ok=True)

    sel = cfg["selection"]
    ruwe_max = float(sel["ruwe_max"])
    teff_min = float(sel["teff_min"])
    teff_max = float(sel["teff_max"])
    poe_min = float(sel["parallax_over_error_min"])

    part_cfg = cfg["partition"]
    nparts = int(part_cfg["nparts"])
    start = int(part_cfg["start"])
    stop = int(part_cfg["stop"])

    service = pyvo.dal.TAPService(tap_url)

    for part in range(start, stop + 1):
        out_part = os.path.join(out_dir, f"gaia_rotmod_join_p{part:02d}_of{nparts:02d}.parquet")
        if os.path.exists(out_part):
            print(f"skip existing: {out_part}")
            continue

        # Partition by source_id modulo, stable and reproducible
        query = f"""
SELECT
  vrm.source_id,
  gs.ra, gs.dec,
  gs.parallax, gs.parallax_error,
  (gs.parallax / gs.parallax_error) AS parallax_over_error,
  gs.phot_g_mean_mag,
  gs.bp_rp,
  gs.ruwe,
  ap.teff_gspphot,
  ap.logg_gspphot,
  ap.mh_gspphot,
  vrm.best_rotation_period AS prot,
  vrm.best_rotation_period_error AS prot_err,
  vrm.max_activity_index_g AS arot,
  vrm.max_activity_index_g_error AS arot_err
FROM gaiadr3.vari_rotation_modulation AS vrm
JOIN gaiadr3.gaia_source AS gs
  ON gs.source_id = vrm.source_id
JOIN gaiadr3.astrophysical_parameters AS ap
  ON ap.source_id = vrm.source_id
WHERE (vrm.source_id % {nparts}) = {part}
  AND gs.ruwe < {ruwe_max}
  AND gs.parallax_error > 0
  AND (gs.parallax / gs.parallax_error) > {poe_min}
  AND ap.teff_gspphot BETWEEN {teff_min} AND {teff_max}
  AND vrm.best_rotation_period IS NOT NULL
  AND vrm.max_activity_index_g IS NOT NULL;
"""

        print(f"fetch part {part}/{nparts-1} ...")
        res = service.run_sync(query, language=lang)
        df = res.to_table().to_pandas()
        df.to_parquet(out_part, index=False)
        print(f"saved: {out_part} rows={len(df)}")

    # Merge parts into a single parquet (optional, but convenient)
    parts = sorted(
        os.path.join(out_dir, x) for x in os.listdir(out_dir)
        if x.endswith(".parquet") and x.startswith("gaia_rotmod_join_p")
    )
    if parts:
        print(f"merging {len(parts)} parts -> {out_merged}")
        df_all = pd.concat((pd.read_parquet(p) for p in parts), ignore_index=True)
        df_all.to_parquet(out_merged, index=False)
        print(f"saved merged: {out_merged} rows={len(df_all)}")


if __name__ == "__main__":
    main()
