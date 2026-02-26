#!/usr/bin/env python3
"""
Clean APOKASC3 Table 5 parquet exported from VizieR.

Fixes:
- Removes unit/header separator rows accidentally parsed as data.
- Coerces KIC to Int64.
- Coerces numeric columns to float where possible.

Inputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5.parquet

Outputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean_summary.yaml
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet"
SUM = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean_summary.yaml"

def is_dash_row(s: str) -> bool:
    s = (s or "").strip()
    return len(s) > 0 and set(s) <= set("-")

def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP)

    n0 = len(df)

    # Drop obvious junk rows:
    # - Units row tends to have KIC NA and recno empty
    # - Separator row has recno '--------' or similar
    recno = df.get("recno")
    if recno is not None:
        recno_s = recno.astype("string")
        bad_sep = recno_s.map(lambda x: is_dash_row(str(x)))
    else:
        bad_sep = pd.Series(False, index=df.index)

    # Units row: KIC missing and many columns are strings like "deg", "Msun", etc.
    # Heuristic: if KIC is NA and recno blank-ish => drop.
    kic = df.get("KIC")
    if kic is not None:
        kic_s = kic.astype("string")
        bad_units = kic_s.isna() & (~bad_sep)
    else:
        bad_units = pd.Series(False, index=df.index)

    df = df.loc[~(bad_sep | bad_units)].copy()
    n1 = len(df)

    # Coerce KIC properly
    if "KIC" in df.columns:
        df["KIC"] = pd.to_numeric(df["KIC"], errors="coerce").astype("Int64")

    # Coerce everything else to numeric where it makes sense (keep non-numeric columns untouched)
    # We'll attempt conversion for object/string cols except obvious IDs/text.
    keep_text = {"Simbad"}
    for c in df.columns:
        if c in keep_text or c == "KIC":
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        # attempt numeric conversion if it looks numeric-ish
        if pd.api.types.is_string_dtype(df[c]) or pd.api.types.is_object_dtype(df[c]):
            x = pd.to_numeric(df[c], errors="coerce")
            # if conversion yields at least some finite values, keep as numeric
            if np.isfinite(x.to_numpy(dtype=float, na_value=np.nan)).sum() > 0:
                df[c] = x.astype(float)

    df.to_parquet(OUT, index=False)

    summary = {
        "input_rows": int(n0),
        "rows_after_clean": int(n1),
        "dropped_rows": int(n0 - n1),
        "kic_nonnull": int(df["KIC"].notna().sum()) if "KIC" in df.columns else None,
        "cols": list(df.columns),
        "out": OUT,
    }
    with open(SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print(f"saved: {OUT} rows={len(df)} cols={len(df.columns)}")
    print(f"summary: {SUM}")

if __name__ == "__main__":
    main()
