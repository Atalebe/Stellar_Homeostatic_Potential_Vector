# src/34i_recover_missing_ids_by_position.py
import pandas as pd
import pyvo
import yaml
import os
import numpy as np
from astropy.coordinates import SkyCoord
import astropy.units as u

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
MISSING_IDS_FILE = "/mnt/g/STAR_HPV/raw/gaia_dr3/debug_legacy_ids/missing_ids_with_coords.csv"
OUT_DIR = "/mnt/g/STAR_HPV/raw/gaia_dr3/recovered_ids"

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    
    # Load missing IDs from your debug output
    missing_ids = [
        2051714265379610368,
        2052125895049468928,
        # ... add all 29 missing IDs
    ]
    
    # You'll need a file with coordinates for these IDs
    # If you don't have coordinates, we need to find them from your legacy catalog
    
    # Example: Search by position with 2 arcsec radius
    service = pyvo.dal.TAPService(TAP_URL)
    
    recovered = []
    still_missing = []
    
    for source_id in missing_ids:
        # You need RA/Dec for each missing ID
        # This assumes you have a file mapping IDs to coordinates
        ra, dec = get_coords_for_id(source_id)  # You need to implement this
        
        query = f"""
        SELECT source_id, ra, dec, parallax, phot_g_mean_mag, 
               bp_rp, ruwe, teff_gspphot, logg_gspphot,
               DISTANCE(POINT('ICRS', ra, dec), 
                       POINT('ICRS', {ra}, {dec})) AS dist
        FROM gaiadr3.gaia_source
        WHERE 1=CONTAINS(POINT('ICRS', ra, dec),
                        CIRCLE('ICRS', {ra}, {dec}, 2.0/3600))
        ORDER BY dist ASC
        LIMIT 1
        """
        
        try:
            result = service.run_sync(query)
            table = result.to_table()
            if len(table) > 0:
                row = table[0]
                recovered.append({
                    'original_id': source_id,
                    'matched_source_id': row['source_id'],
                    'ra': row['ra'],
                    'dec': row['dec'],
                    'dist_arcsec': row['dist'] * 3600,
                    'parallax': row['parallax']
                })
                print(f"✓ Recovered {source_id} → {row['source_id']}")
            else:
                still_missing.append(source_id)
                print(f"✗ No match for {source_id}")
        except Exception as e:
            print(f"Error querying {source_id}: {e}")
            still_missing.append(source_id)
    
    # Save results
    results = {
        'original_missing': len(missing_ids),
        'recovered': len(recovered),
        'still_missing': len(still_missing),
        'recovered_details': recovered,
        'still_missing_ids': still_missing
    }
    
    with open(f"{OUT_DIR}/recovery_results.yaml", 'w') as f:
        yaml.dump(results, f)
    
    if recovered:
        pd.DataFrame(recovered).to_csv(f"{OUT_DIR}/recovered_matches.csv", index=False)
    
    print(f"\nRecovered: {len(recovered)}/{len(missing_ids)}")
    print(f"Still missing: {len(still_missing)}/{len(missing_ids)}")

if __name__ == "__main__":
    main()
