import os
import numpy as np
import pandas as pd
import pyvo

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"

TAP_URL = "https://gaia.aip.de/tap"
LANG = "ADQL"

R_ARCSEC = 1.0
CHUNK = 12  # keep query text small

def inline_union(rows):
    # ADQL inline table: SELECT literals UNION ALL SELECT literals ...
    parts = []
    for kic, ra, dec in rows:
        parts.append(f"SELECT {int(kic)} AS kic, {float(ra)} AS ra0, {float(dec)} AS dec0")
    return "\nUNION ALL\n".join(parts)

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_parquet(INP, columns=["kic","ra","dec"]).dropna().copy()
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["kic","ra","dec"]).copy()

    items = list(df[["kic","ra","dec"]].itertuples(index=False, name=None))

    service = pyvo.dal.TAPService(TAP_URL)
    rad_deg = R_ARCSEC / 3600.0

    out_parts = []

    for i in range(0, len(items), CHUNK):
        chunk_rows = items[i:i+CHUNK]
        t = inline_union(chunk_rows)

        q = f"""
        SELECT
          t.kic,
          gs.source_id,
          gs.ra, gs.dec,
          gs.phot_g_mean_mag,
          gs.ruwe
        FROM (
          {t}
        ) AS t
        JOIN gaiadr3.gaia_source AS gs
        ON 1 = CONTAINS(
          POINT('ICRS', gs.ra, gs.dec),
          CIRCLE('ICRS', t.ra0, t.dec0, {rad_deg})
        )
        """

        try:
            res = service.run_sync(q, language=LANG)
            part = res.to_table().to_pandas()
        except Exception as e:
            print(f"chunk {i//CHUNK} failed:", e)
            continue

        if len(part) == 0:
            print(f"chunk {i//CHUNK}: no matches")
            continue

        # Keep brightest Gaia match per KIC
        part["phot_g_mean_mag"] = pd.to_numeric(part["phot_g_mean_mag"], errors="coerce")
        part = part.sort_values(["kic","phot_g_mean_mag"], ascending=[True, True])
        part = part.drop_duplicates(subset=["kic"], keep="first")

        out_parts.append(part)
        print(f"chunk {i//CHUNK}: matched {len(part)} / {len(chunk_rows)}")

    if not out_parts:
        raise RuntimeError("No matches obtained from any chunk.")

    out = pd.concat(out_parts, ignore_index=True)
    out = out.drop_duplicates(subset=["kic"], keep="first").copy()
    out.to_csv(OUT, index=False)

    print("saved:", OUT, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
