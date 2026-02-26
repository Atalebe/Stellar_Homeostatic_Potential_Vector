#!/usr/bin/env python3
"""
Scan APOKASC3 Table 5 for plausible age columns.

Heuristic: a real age in Gyr should have a wide range (e.g., ~0.1 to ~14),
not be trapped in [0,1]. We also look for log-age columns.

Input:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet

Output:
- /mnt/g/STAR_HPV/raw/ages/apokasc3_age_column_diagnostics.yaml
"""

from __future__ import annotations
import os
import yaml
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/apokasc3_table5_clean.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/apokasc3_age_column_diagnostics.yaml"

def finite_stats(x: pd.Series) -> dict:
    v = pd.to_numeric(x, errors="coerce").astype(float)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return {"n": 0}
    return {
        "n": int(len(v)),
        "min": float(np.min(v)),
        "p05": float(np.quantile(v, 0.05)),
        "p50": float(np.quantile(v, 0.50)),
        "p95": float(np.quantile(v, 0.95)),
        "max": float(np.max(v)),
    }

def main() -> None:
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP)

    # candidates by name
    name_hits = [c for c in df.columns if ("age" in c.lower()) or ("tau" in c.lower())]
    # also include some common APOKASC-ish labels
    extra = [c for c in df.columns if c.lower() in {"fagecor","agecor","age","logage","log_age","tage","tage_gyr"}]
    cand = sorted(set(name_hits + extra))

    diag = {"input_rows": int(len(df)), "candidates": {}}

    for c in cand:
        diag["candidates"][c] = finite_stats(df[c])

    # rank plausible Gyr-like columns:
    # wide spread and max>2 and p95>3
    scored = []
    for c, st in diag["candidates"].items():
        if st.get("n", 0) < 100:
            continue
        mx = st.get("max", np.nan)
        p95 = st.get("p95", np.nan)
        p05 = st.get("p05", np.nan)
        if np.isfinite(mx) and np.isfinite(p95) and np.isfinite(p05):
            wide = (p95 - p05)
            score = 0.0
            if mx > 5: score += 2
            if p95 > 3: score += 2
            if wide > 2: score += 2
            if mx > 10: score += 1
            scored.append((score, c, st))

    scored.sort(reverse=True, key=lambda t: t[0])
    diag["top_plausible_age_columns"] = [
        {"col": c, "score": float(s), "stats": st} for (s, c, st) in scored[:15]
    ]

    with open(OUT, "w") as f:
        yaml.safe_dump(diag, f, sort_keys=False)

    print(f"saved: {OUT}")
    if diag["top_plausible_age_columns"]:
        print("top plausible age columns:")
        for r in diag["top_plausible_age_columns"][:8]:
            print(" ", r["col"], "score", r["score"], "stats", r["stats"])

if __name__ == "__main__":
    main()
