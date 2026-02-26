#!/usr/bin/env python3
"""
Fetch APOKASC-3 table5 from VizieR in CSV form and parse robustly.

Outputs:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5.raw.txt (raw response)
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5.csv
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5.parquet
"""

from __future__ import annotations
import os
import pandas as pd
import requests

OUT_RAW  = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5.raw.txt"
OUT_CSV  = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5.csv"
OUT_PARQ = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5.parquet"

URL = "https://tapvizier.u-strasbg.fr/viz-bin/asu-tsv"  # stable endpoint
PARAMS = {
    "-source": "J/ApJS/276/69/table5",
    "-out.all": "1",
    "-out.max": "200000",
    "-out": "KIC,ESA3,Teff17,logg17,[Fe/H]17,MSH,RSH,RGBAgeM,RCAgeM,RGBAgeNom,RCAgeNom,YRECAge,_RA,_DE",
}


def main() -> None:
    os.makedirs(os.path.dirname(OUT_PARQ), exist_ok=True)

    r = requests.get(URL, params=PARAMS, timeout=180)
    r.raise_for_status()

    txt = r.text
    with open(OUT_RAW, "w") as f:
        f.write(txt)

    # VizieR asu-tsv: header lines begin with '#'
    lines = [ln for ln in txt.splitlines() if ln.strip() and not ln.startswith("#")]

    if len(lines) < 2:
        raise RuntimeError(f"No data rows returned. See raw: {OUT_RAW}")

    from io import StringIO
    df = pd.read_csv(StringIO("\n".join(lines)), sep="\t", engine="python")

    if "KIC" not in df.columns:
        raise RuntimeError(f"Missing KIC column. Columns: {list(df.columns)[:50]}. See raw: {OUT_RAW}")

    df["KIC"] = pd.to_numeric(df["KIC"], errors="coerce").astype("Int64")

    df.to_csv(OUT_CSV, index=False)
    df.to_parquet(OUT_PARQ, index=False)

    print(f"saved: {OUT_PARQ} rows={len(df)} cols={len(df.columns)}")
    print(f"saved: {OUT_CSV}")
    print(f"raw: {OUT_RAW}")


if __name__ == "__main__":
    main()
