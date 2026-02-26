import os
import numpy as np
import pandas as pd
import pyvo

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"
TMP_VOT = "/mnt/g/STAR_HPV/raw/ages/_tmp_legacy_targets.vot"

TAP_URL = "https://gaia.aip.de/tap"
LANG = "ADQL"
R_ARCSEC = 1.0

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_parquet(INP, columns=["kic","ra","dec"]).dropna().copy()
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["kic","ra","dec"]).copy()

    # Upload table must use simple names
    targets = pd.DataFrame({
        "kic": df["kic"].astype(int),
        "ra0": df["ra"].astype(float),
        "dec0": df["dec"].astype(float),
    })

    # Write VOTable for TAP_UPLOAD
    from astropy.table import Table
    tbl = Table.from_pandas(targets)
    tbl.write(TMP_VOT, format="votable", overwrite=True)

    rad_deg = R_ARCSEC / 3600.0
    q = f"""
    SELECT
      t.kic,
      gs.source_id,
      gs.ra, gs.dec,
      gs.phot_g_mean_mag,
      gs.ruwe
    FROM TAP_UPLOAD.targets AS t
    JOIN gaiadr3.gaia_source AS gs
    ON 1 = CONTAINS(
      POINT('ICRS', gs.ra, gs.dec),
      CIRCLE('ICRS', t.ra0, t.dec0, {rad_deg})
    )
    """

    service = pyvo.dal.TAPService(TAP_URL)

    # IMPORTANT: upload the VOTable under the name "targets"
    # pyvo expects uploads={"targets": <path>}
    res = service.run_sync(q, language=LANG, uploads={"targets": TMP_VOT})
    out = res.to_table().to_pandas()

    if len(out) == 0:
        raise RuntimeError("No Gaia matches returned. Increase radius or check TAP upload support.")

    # Keep brightest Gaia match per KIC (min Gmag)
    out["phot_g_mean_mag"] = pd.to_numeric(out["phot_g_mean_mag"], errors="coerce")
    out = out.sort_values(["kic","phot_g_mean_mag"], ascending=[True, True])
    out = out.drop_duplicates(subset=["kic"], keep="first").copy()

    out.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
