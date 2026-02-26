import os
import pandas as pd
from astroquery.vizier import Vizier

OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_table4_per_pipeline.parquet"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    Vizier.ROW_LIMIT = -1
    v = Vizier(columns=["*"], row_limit=-1)

    # Explicitly request table4 from the LEGACY sample II catalog
    # Catalog: J/ApJ/835/173, file: table4.dat (per-pipeline properties; includes Age in Gyr)
    tabs = v.get_catalogs("J/ApJ/835/173/table4")
    if not tabs:
        raise RuntimeError("No tables returned for J/ApJ/835/173/table4")

    df = tabs[0].to_pandas()

    # Normalize common column names
    # VizieR sometimes uses 'KIC'/'kic' variations; keep robust.
    cols = {c.lower(): c for c in df.columns}
    kic_col = cols.get("kic", None)
    pipe_col = cols.get("pipe", None)
    age_col = cols.get("age", None)

    if kic_col is None or pipe_col is None or age_col is None:
        raise RuntimeError(f"Missing required columns. Have: {list(df.columns)}")

    df = df.rename(columns={kic_col: "kic", pipe_col: "pipe", age_col: "age_gyr"})
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df["age_gyr"] = pd.to_numeric(df["age_gyr"], errors="coerce")

    # Drop invalid ages if any
    df = df.dropna(subset=["kic", "age_gyr"]).copy()

    df.to_parquet(OUT, index=False)
    print(f"saved: {OUT} rows={len(df)} cols={len(df.columns)}")
    print(df[["pipe","kic","age_gyr"]].head(10))

if __name__ == "__main__":
    main()
