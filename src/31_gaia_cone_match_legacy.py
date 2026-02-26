import os
import numpy as np
import pandas as pd
import pyvo

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"

TAP_URL = "https://gaia.aip.de/tap"
LANG = "ADQL"
R_ARCSEC = 1.0  # match radius

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP, columns=["kic", "ra", "dec"]).copy()

    service = pyvo.dal.TAPService(TAP_URL)

    rows = []
    for _, r in df.iterrows():
        kic = int(r["kic"])
        ra = float(r["ra"])
        dec = float(r["dec"])

        radius_deg = R_ARCSEC / 3600.0

        q = f"""
        SELECT TOP 1
          gs.source_id, gs.ra, gs.dec, gs.phot_g_mean_mag, gs.ruwe
        FROM gaiadr3.gaia_source AS gs
        WHERE 1 = CONTAINS(
          POINT('ICRS', gs.ra, gs.dec),
          CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
        )
        ORDER BY gs.phot_g_mean_mag ASC
        """

        res = service.run_sync(q, language=LANG)
        t = res.to_table().to_pandas()
        if len(t) == 0:
            continue

        d = t.iloc[0].to_dict()
        d["kic"] = kic
        rows.append(d)

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
