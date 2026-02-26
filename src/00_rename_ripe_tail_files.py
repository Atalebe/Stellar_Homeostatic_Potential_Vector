#!/usr/bin/env python3
"""
Rename ripe-tail outputs to consistent names.

This is optional. It does NOT change data, only filenames.
"""

from __future__ import annotations
import os
from pathlib import Path

ROOT = Path("/mnt/g/STAR_HPV/results/anchor_catalogs")

RENAMES = {
    # global post-gate tails (old names)
    "kepler_ripe_tail_q90.csv": "kepler_ripe_tail_global_postgate_q90.csv",
    "kepler_ripe_tail_q90.parquet": "kepler_ripe_tail_global_postgate_q90.parquet",
    "kepler_ripe_tail_q95.csv": "kepler_ripe_tail_global_postgate_q95.csv",
    "kepler_ripe_tail_q95.parquet": "kepler_ripe_tail_global_postgate_q95.parquet",
    "kepler_ripe_tail_summary.yaml": "kepler_ripe_tail_global_postgate_summary.yaml",

    # window-only tails (already good in your listing)
    # per-bin window tails (already good)
}

def main():
    ROOT.mkdir(parents=True, exist_ok=True)

    moved = 0
    for old, new in RENAMES.items():
        p_old = ROOT / old
        p_new = ROOT / new
        if not p_old.exists():
            continue
        if p_new.exists():
            print(f"skip (target exists): {p_new}")
            continue
        p_old.rename(p_new)
        moved += 1
        print(f"renamed: {p_old.name} -> {p_new.name}")

    print(f"done. renamed={moved}")

if __name__ == "__main__":
    main()
