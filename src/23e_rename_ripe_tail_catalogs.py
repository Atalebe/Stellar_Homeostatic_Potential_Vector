#!/usr/bin/env python3
from pathlib import Path
import shutil

SRC_DIR = Path("/mnt/g/STAR_HPV/results/anchor_catalogs")

RENAMES = {
    # clarify old naming
    "kepler_ripe_tail_postgate_q90.csv": "kepler_ripe_tail_global_postgate_q90.csv",
    "kepler_ripe_tail_postgate_q90.parquet": "kepler_ripe_tail_global_postgate_q90.parquet",
    "kepler_ripe_tail_postgate_q95.csv": "kepler_ripe_tail_global_postgate_q95.csv",
    "kepler_ripe_tail_postgate_q95.parquet": "kepler_ripe_tail_global_postgate_q95.parquet",
    "kepler_ripe_tail_postgate_summary.yaml": "kepler_ripe_tail_global_postgate_summary.yaml",
}

def main():
    done = []
    skipped = []
    for src_name, dst_name in RENAMES.items():
        src = SRC_DIR / src_name
        dst = SRC_DIR / dst_name
        if not src.exists():
            skipped.append((src_name, "missing"))
            continue
        if dst.exists():
            skipped.append((dst_name, "already exists"))
            continue
        shutil.move(str(src), str(dst))
        done.append((src_name, dst_name))

    print("moved:", len(done))
    for a, b in done:
        print(" ", a, "->", b)
    print("skipped:", len(skipped))
    for a, why in skipped[:20]:
        print(" ", a, why)

if __name__ == "__main__":
    main()
