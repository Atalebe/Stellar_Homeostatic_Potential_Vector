#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/g/STAR_HPV/raw/apogee_dr17"
mkdir -p "$BASE"
cd "$BASE"

# SDSS SAS direct
wget -c https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStar-dr17-synspec_rev1.fits
wget -c https://data.sdss.org/sas/dr17/apogee/spectro/aspcap/dr17/synspec_rev1/allStarLite-dr17-synspec_rev1.fits
