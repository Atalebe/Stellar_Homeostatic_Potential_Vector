import os
import yaml
import pyvo
import pandas as pd
import numpy as np

CFG = "configs/gaia_kepler_ids.yaml"

def load_cfg(p):
    with open(p,"r") as f:
        return yaml.safe_load(f)

def main():
    cfg = load_cfg(CFG)
    tap_url = cfg["tap_url"]
    lang = cfg.get("language","PostgreSQL")
    ids_csv = cfg["ids_csv"]
    out_dir = cfg["out_dir"]
    os.makedirs(out_dir, exist_ok=True)

    chunk = int(cfg.get("chunk_size", 2000))
    ids = pd.read_csv(ids_csv, dtype={"source_id":"string"})["source_id"].dropna().astype("string").unique().tolist()
    print("ids:", len(ids))

    service = pyvo.dal.TAPService(tap_url)

    for i in range(0, len(ids), chunk):
        part = i // chunk
        outp = os.path.join(out_dir, f"gaia_kepler_ids_part{part:03d}.parquet")
        if os.path.exists(outp):
            print("skip", outp)
            continue

        id_list = ",".join(ids[i:i+chunk])
        query = f"""
        SELECT
          gs.source_id,
          gs.ra, gs.dec,
          gs.parallax, gs.parallax_error,
          gs.phot_g_mean_mag,
          gs.bp_rp,
          gs.ruwe,
          ap.teff_gspphot,
          ap.logg_gspphot,
          ap.mh_gspphot
        FROM gaiadr3.gaia_source AS gs
        LEFT JOIN gaiadr3.astrophysical_parameters AS ap
          ON ap.source_id = gs.source_id
        WHERE gs.source_id IN ({id_list});
        """
        res = service.run_sync(query, language=lang)
        df = res.to_table().to_pandas()
        df.to_parquet(outp, index=False)
        print("saved", outp, "rows", len(df))

if __name__=="__main__":
    main()
