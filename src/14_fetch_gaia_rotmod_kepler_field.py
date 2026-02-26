import os
import yaml
import pyvo
import pandas as pd

def load_cfg(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg("configs/gaia_kepler_field.yaml")["gaia"]
    tap_url = cfg["tap_url"]
    lang = cfg.get("language", "PostgreSQL")
    out_path = cfg["out_parquet"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    s = cfg["selection"]
    ruwe_max = float(s["ruwe_max"])
    teff_min = float(s["teff_min"])
    teff_max = float(s["teff_max"])
    poe_min  = float(s["parallax_over_error_min"])

    region = cfg["region"]
    ra_min = float(region["ra_min"])
    ra_max = float(region["ra_max"])
    dec_min = float(region["dec_min"])
    dec_max = float(region["dec_max"])

    service = pyvo.dal.TAPService(tap_url)

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
    WHERE gs.ra BETWEEN {ra_min} AND {ra_max}
      AND gs.dec BETWEEN {dec_min} AND {dec_max}
      AND gs.ruwe < {ruwe_max}
      AND gs.parallax_error > 0
      AND (gs.parallax / gs.parallax_error) > {poe_min}
      AND ap.teff_gspphot BETWEEN {teff_min} AND {teff_max}
      AND vrm.best_rotation_period IS NOT NULL
      AND vrm.max_activity_index_g IS NOT NULL;
    """

    res = service.run_sync(query, language=lang)
    df = res.to_table().to_pandas()
    df.to_parquet(out_path, index=False)
    print(f"saved: {out_path} rows={len(df)}")

if __name__ == "__main__":
    main()
