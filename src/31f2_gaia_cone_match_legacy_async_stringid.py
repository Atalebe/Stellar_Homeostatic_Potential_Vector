import os, time
import pandas as pd
import pyvo

TAP_URL = "https://gaia.aip.de/tap"   # keep your working endpoint
LANG = "ADQL"

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string.csv"

RADIUS_DEG = 1.0 / 3600.0  # 1 arcsec
RETRIES = 6

def run_one(service, ra, dec):
    # KEY: CAST source_id to VARCHAR so it cannot become float anywhere
    q = f"""
    SELECT
      CAST(gs.source_id AS VARCHAR) AS source_id_str,
      gs.ra, gs.dec,
      gs.phot_g_mean_mag, gs.ruwe,
      DISTANCE(POINT('ICRS', gs.ra, gs.dec),
               POINT('ICRS', {ra}, {dec})) AS dist_deg
    FROM gaiadr3.gaia_source AS gs
    WHERE 1=CONTAINS(
      POINT('ICRS', gs.ra, gs.dec),
      CIRCLE('ICRS', {ra}, {dec}, {RADIUS_DEG})
    )
    ORDER BY dist_deg ASC
    """
    res = service.run_sync(q, language=LANG)
    df = res.to_table().to_pandas()
    return df

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    leg = pd.read_parquet(INP).copy()
    leg["kic"] = pd.to_numeric(leg["kic"], errors="coerce").astype("Int64")
    leg["ra"] = pd.to_numeric(leg["ra"], errors="coerce")
    leg["dec"] = pd.to_numeric(leg["dec"], errors="coerce")
    leg = leg.dropna(subset=["kic","ra","dec"]).copy()

    service = pyvo.dal.TAPService(TAP_URL)

    rows = []
    ok = 0
    fail = 0

    for _, r in leg.iterrows():
        kic = int(r["kic"])
        ra = float(r["ra"])
        dec = float(r["dec"])

        got = None
        for a in range(RETRIES):
            try:
                df = run_one(service, ra, dec)
                got = df
                break
            except Exception as e:
                time.sleep(min(60, 2**a))

        if got is None or len(got) == 0:
            fail += 1
            continue

        best = got.iloc[0]
        rows.append({
            "kic": kic,
            "ra": ra,
            "dec": dec,
            "source_id": str(best["source_id_str"]).strip(),
            "phot_g_mean_mag": float(best["phot_g_mean_mag"]) if "phot_g_mean_mag" in got.columns else None,
            "ruwe": float(best["ruwe"]) if "ruwe" in got.columns else None,
            "dist_deg": float(best["dist_deg"]) if "dist_deg" in got.columns else None,
        })
        ok += 1
        if ok % 10 == 0:
            print(f"progress: ok={ok} fail={fail}")

    out = pd.DataFrame(rows)
    out.to_csv(OUT, index=False)
    print("saved:", OUT, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
