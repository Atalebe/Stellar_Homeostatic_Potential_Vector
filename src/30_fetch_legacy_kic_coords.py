import os
import pandas as pd
from astroquery.vizier import Vizier

OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    Vizier.ROW_LIMIT = -1
    v = Vizier(columns=["*"], row_limit=-1)

    tables = v.get_catalogs("J/ApJ/835/173")
    if not tables:
        raise RuntimeError("No tables returned for J/ApJ/835/173")

    # Pick a table that contains KIC + RA + Dec
    pick = None
    for t in tables:
        cols = [c.lower() for c in t.colnames]
        has_kic = any(c == "kic" for c in cols)
        has_ra = any(c in ("raj2000","ra","_ra","ra_icrs") for c in cols)
        has_de = any(c in ("dej2000","dec","_de","de_icrs") for c in cols)
        if has_kic and has_ra and has_de:
            pick = t
            break

    if pick is None:
        # Print what we got to guide next step
        for i, t in enumerate(tables):
            print(i, t.meta.get("name", "table"), t.colnames[:20])
        raise RuntimeError("Could not find a LEGACY table containing KIC+RA+Dec.")

    df = pick.to_pandas()

    # Normalize column names
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc == "kic":
            colmap[c] = "kic"
        elif lc in ("raj2000","ra","_ra","ra_icrs"):
            colmap[c] = "ra"
        elif lc in ("dej2000","dec","_de","de_icrs"):
            colmap[c] = "dec"
    df = df.rename(columns=colmap)

    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["kic","ra","dec"]).drop_duplicates(subset=["kic"]).copy()

    df.to_parquet(OUT, index=False)
    print(f"saved: {OUT} rows={len(df)} cols={list(df.columns)}")
    print(df.head(10))

if __name__ == "__main__":
    main()
