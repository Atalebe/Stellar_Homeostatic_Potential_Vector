#!/usr/bin/env bash
set -euo pipefail

BASE="/mnt/g/STAR_HPV/raw/galah_dr3"
mkdir -p "$BASE"
cd "$BASE"

# Main catalog (recommended)
wget -c https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/GALAH_DR3_main_allstar_v2.fits

# Ages VAC
wget -c https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/GALAH_DR3_VAC_ages_v2.fits

# Gaia cross match VAC (optional)
wget -c https://cloud.datacentral.org.au/teamdata/GALAH/public/GALAH_DR3/GALAH_DR3_VAC_GaiaEDR3_v2.fits
