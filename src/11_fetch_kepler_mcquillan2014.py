import os
import pandas as pd
from astroquery.vizier import Vizier

OUT = "/mnt/g/STAR_HPV/raw/kepler/mcquillan2014_rotators.parquet"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    # Force "no limit" both ways
    Vizier.ROW_LIMIT = -1
    v = Vizier(columns=["*"], row_limit=-1)

    # IMPORTANT: request table1 explicitly (34030 rows)
    tables = v.get_catalogs("J/ApJS/211/24/table1")
    if not tables:
        raise RuntimeError("No tables returned for J/ApJS/211/24/table1")

    tab = tables[0]
    df = tab.to_pandas()

    # Normalize expected columns
    # table1 has KIC and Prot (case can vary)
    colmap = {}
    for c in df.columns:
        lc = c.lower()
        if lc in ("kic", "kicid", "kic_id"):
            colmap[c] = "kic"
        if lc in ("prot", "p_rot", "period"):
            colmap[c] = "prot_days"
    df = df.rename(columns=colmap)

    # Fallback detection
    if "kic" not in df.columns:
        kic_cols = [c for c in df.columns if "kic" in c.lower()]
        if not kic_cols:
            raise RuntimeError(f"Could not find KIC column. Columns: {list(df.columns)[:50]}")
        df = df.rename(columns={kic_cols[0]: "kic"})

    if "prot_days" not in df.columns:
        prot_cols = [c for c in df.columns if "prot" in c.lower() or "period" in c.lower()]
        if not prot_cols:
            raise RuntimeError(f"Could not find Prot column. Columns: {list(df.columns)[:50]}")
        df = df.rename(columns={prot_cols[0]: "prot_days"})

    # Clean
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df["prot_days"] = pd.to_numeric(df["prot_days"], errors="coerce")
    df = df.dropna(subset=["kic", "prot_days"]).copy()
    df = df[df["prot_days"] > 0].copy()

    df.to_parquet(OUT, index=False)
    print(f"saved: {OUT} rows={len(df)} cols={len(df.columns)}")
    print("head:")
    print(df.head())

if __name__ == "__main__":
    main()
