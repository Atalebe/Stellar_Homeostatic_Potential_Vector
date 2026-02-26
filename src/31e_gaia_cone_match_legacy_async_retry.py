import os
import time
import numpy as np
import pandas as pd
import pyvo

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid.csv"

# AIP is flaky. ESA is usually steadier, but may require login sometimes.
# Try AIP first; if it keeps failing, swap to ESA:
# TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
TAP_URL = "https://gaia.aip.de/tap"
LANG = "ADQL"

R_ARCSEC = 1.0
MAX_RETRIES = 6

def run_one(service, kic, ra, dec, radius_deg):
    # Order by angular distance; TOP 1 keeps it fast.
    q = f"""
    SELECT TOP 1
      gs.source_id, gs.ra, gs.dec, gs.phot_g_mean_mag, gs.ruwe,
      DISTANCE(POINT('ICRS', gs.ra, gs.dec), POINT('ICRS', {ra}, {dec})) AS dist_deg
    FROM gaiadr3.gaia_source AS gs
    WHERE 1 = CONTAINS(
      POINT('ICRS', gs.ra, gs.dec),
      CIRCLE('ICRS', {ra}, {dec}, {radius_deg})
    )
    ORDER BY dist_deg ASC
    """

    # Async job avoids sync statement timeouts on busy servers
    job = service.submit_job(q, language=LANG)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR", "ABORTED"], timeout=300)

    if job.phase != "COMPLETED":
        raise RuntimeError(f"job phase={job.phase}")

    tab = job.fetch_result().to_table().to_pandas()
    if len(tab) == 0:
        return None

    d = tab.iloc[0].to_dict()
    d["kic"] = int(kic)
    return d

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_parquet(INP, columns=["kic","ra","dec"]).dropna().copy()
    df["kic"] = pd.to_numeric(df["kic"], errors="coerce").astype("Int64")
    df["ra"] = pd.to_numeric(df["ra"], errors="coerce")
    df["dec"] = pd.to_numeric(df["dec"], errors="coerce")
    df = df.dropna(subset=["kic","ra","dec"]).copy()

    service = pyvo.dal.TAPService(TAP_URL)
    radius_deg = R_ARCSEC / 3600.0

    rows = []
    n_ok = 0
    n_fail = 0

    for i, r in df.iterrows():
        kic = int(r["kic"])
        ra = float(r["ra"])
        dec = float(r["dec"])

        ok = False
        for attempt in range(MAX_RETRIES):
            try:
                out = run_one(service, kic, ra, dec, radius_deg)
                if out is not None:
                    rows.append(out)
                ok = True
                n_ok += 1
                break
            except Exception as e:
                wait = min(60, 2 ** attempt)
                print(f"[KIC {kic}] attempt {attempt+1}/{MAX_RETRIES} failed: {e} ; sleep {wait}s")
                time.sleep(wait)

        if not ok:
            n_fail += 1
            print(f"[KIC {kic}] FAILED after {MAX_RETRIES} retries")

        if (n_ok + n_fail) % 10 == 0:
            print(f"progress: ok={n_ok} fail={n_fail}")

    if not rows:
        raise RuntimeError("No matches recovered. Consider increasing radius or switching TAP_URL to ESA.")

    out = pd.DataFrame(rows)
    # One match per KIC already by TOP 1, but keep safe
    out = out.drop_duplicates(subset=["kic"], keep="first").copy()
    out.to_csv(OUT, index=False)

    print("saved:", OUT, "rows:", len(out))
    print(out.head(10))

if __name__ == "__main__":
    main()
