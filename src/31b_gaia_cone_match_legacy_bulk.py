import os
import numpy as np
import pandas as pd
import pyvo

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"

TAP_URL = "https://gaia.aip.de/tap"
LANG = "ADQL"

R_ARCSEC = 1.0
CHUNK = 15   # keep small to avoid huge query text

def make_values(rows):
    # ADQL VALUES-style inline table: (kic, ra, dec)
    # We use SELECT ... UNION ALL because VALUES is not universally supported.
    parts = []
    for kic, ra, dec in rows:
        parts.append(f"SELECT {int(kic)} AS kic, {float(ra)} AS ra0, {float(dec)} AS dec0 FROM TAP_UPLOAD.dummy")
    # We'll replace TAP_UPLOAD.dummy with a single-row subquery (SELECT 1 as dummy)
    return "\nUNION ALL\n".join(parts).replace("TAP_UPLOAD.dummy", "(SELECT 1 AS dummy)")

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP, columns=["kic","ra","dec"]).dropna().copy()
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["kic"]).copy()

    service = pyvo.dal.TAPService(TAP_URL)
    rad_deg = R_ARCSEC / 3600.0

    out_rows = []
    items = list(df[["kic","ra","dec"]].itertuples(index=False, name=None))

    for i in range(0, len(items), CHUNK):
        chunk_rows = items[i:i+CHUNK]
        inline = make_values(chunk_rows)

        q = f"""
        WITH targets AS (
          {inline}
        ),
        matches AS (
          SELECT
            t.kic,
            gs.source_id,
            gs.ra, gs.dec,
            gs.phot_g_mean_mag,
            gs.ruwe
          FROM targets t
          JOIN gaiadr3.gaia_source gs
          ON 1 = CONTAINS(
            POINT('ICRS', gs.ra, gs.dec),
            CIRCLE('ICRS', t.ra0, t.dec0, {rad_deg})
          )
        )
        SELECT m.*
        FROM matches m
        """

        try:
            res = service.run_sync(q, language=LANG)
        except Exception as e:
            print(f"chunk {i//CHUNK} failed:", e)
            continue

        part = res.to_table().to_pandas()
        if len(part) == 0:
            print(f"chunk {i//CHUNK}: no matches")
            continue

        # For each KIC, keep brightest match (min G)
        part["phot_g_mean_mag"] = pd.to_numeric(part["phot_g_mean_mag"], errors="coerce")
        part = part.sort_values(["kic","phot_g_mean_mag"], ascending=[True, True])
        part = part.drop_duplicates(subset=["kic"], keep="first")

        out_rows.append(part)
        print(f"chunk {i//CHUNK}: matched {len(part)} / {len(chunk_rows)}")

    if not out_rows:
        raise RuntimeError("No matches obtained from any chunk.")

    out = pd.concat(out_rows, ignore_index=True)
    out = out.drop_duplicates(subset=["kic"], keep="first").copy()

    out.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
