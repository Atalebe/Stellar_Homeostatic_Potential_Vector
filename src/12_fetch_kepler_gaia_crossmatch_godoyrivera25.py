# src/12_fetch_kepler_gaia_crossmatch_godoyrivera25.py
import os
import requests

OUT = "/mnt/g/STAR_HPV/raw/kepler/godoyrivera25_tableA1.csv"

# Zenodo record contains "GodoyRivera25_TableA1.csv"
URL = "https://zenodo.org/records/14774100/files/GodoyRivera25_TableA1.csv?download=1"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    r = requests.get(URL, timeout=120)
    r.raise_for_status()
    with open(OUT, "wb") as f:
        f.write(r.content)
    print(f"saved: {OUT} bytes={len(r.content)}")

if __name__ == "__main__":
    main()
