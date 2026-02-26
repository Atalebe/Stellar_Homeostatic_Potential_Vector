#!/usr/bin/env bash
set -euo pipefail

D="/mnt/g/STAR_HPV/results/anchor_catalogs"

# old names -> clearer names
mv -v "$D/kepler_ripe_tail_postgate_q90.csv"      "$D/kepler_ripe_tail_global_postgate_q90.csv"
mv -v "$D/kepler_ripe_tail_postgate_q90.parquet"  "$D/kepler_ripe_tail_global_postgate_q90.parquet"
mv -v "$D/kepler_ripe_tail_postgate_q95.csv"      "$D/kepler_ripe_tail_global_postgate_q95.csv"
mv -v "$D/kepler_ripe_tail_postgate_q95.parquet"  "$D/kepler_ripe_tail_global_postgate_q95.parquet"
mv -v "$D/kepler_ripe_tail_postgate_summary.yaml" "$D/kepler_ripe_tail_global_postgate_summary.yaml"

echo "done"
ls -lh "$D" | grep kepler_ripe_tail | sed -e 's/  */ /g'
