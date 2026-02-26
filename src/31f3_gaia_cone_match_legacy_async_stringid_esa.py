import os, time, sys
import pandas as pd
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string.csv"
OUT_TMP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string.partial.csv"

RADIUS_DEG = 1.0 / 3600.0  # 1 arcsec
RETRIES = 6
SLEEP_BETWEEN = 0.2
PER_TARGET_TIMEOUT = 120  # ESA can be slower but stable

# IMPORTANT: ESA DR3 schema/table
GAIA_TABLE = "gaiadr3.gaia_source"

def build_query(ra, dec):
    return f"""
    SELECT
      CAST(gs.source_id AS VARCHAR) AS source_id_str,
      gs.ra, gs.dec,
      gs.phot_g_mean_mag, gs.ruwe,
      DISTANCE(POINT('ICRS', gs.ra, gs.dec),
               POINT('ICRS', {ra}, {dec})) AS dist_deg
    FROM {GAIA_TABLE} AS gs
    WHERE 1=CONTAINS(
      POINT('ICRS', gs.ra, gs.dec),
      CIRCLE('ICRS', {ra}, {dec}, {RADIUS_DEG})
    )
    ORDER BY dist_deg ASC
    """

def run_one(service, ra, dec):
    q = build_query(ra, dec)
    job = service.submit_job(q, language=LANG)
    job.run()
    job.wait(phases=["COMPLETED","ERROR","ABORTED"], timeout=PER_TARGET_TIMEOUT)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"phase={job.phase}")
    return job.fetch_result().to_table().to_pandas()

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return None

def load_done_kic():
    done = set()
    for p in [OUT, OUT_TMP]:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p, dtype={"source_id":"string"})
                if "kic" in df.columns:
                    done |= set(pd.to_numeric(df["kic"], errors="coerce").dropna().astype(int).tolist())
            except Exception:
                pass
    return done

def append_row(rowdict):
    df = pd.DataFrame([rowdict])
    header = not os.path.exists(OUT_TMP)
    df.to_csv(OUT_TMP, mode="a", index=False, header=header)

def finalize():
    if os.path.exists(OUT_TMP):
        df = pd.read_csv(OUT_TMP, dtype={"source_id":"string"})
        df = df.drop_duplicates(subset=["kic"])
        df.to_csv(OUT, index=False)
        print("finalized ->", OUT, "rows:", len(df))

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    leg = pd.read_parquet(INP).copy()
    leg["kic"] = pd.to_numeric(leg["kic"], errors="coerce").astype("Int64")
    leg["ra"] = pd.to_numeric(leg["ra"], errors="coerce")
    leg["dec"] = pd.to_numeric(leg["dec"], errors="coerce")
    leg = leg.dropna(subset=["kic","ra","dec"]).copy()
    leg["kic_i"] = leg["kic"].astype(int)

    done = load_done_kic()
    todo = leg[~leg["kic_i"].isin(done)].copy()

    print(f"targets total={len(leg)} done={len(done)} todo={len(todo)}")
    sys.stdout.flush()

    service = pyvo.dal.TAPService(TAP_URL)

    ok = 0
    fail = 0
    t0 = time.time()

    try:
        for i, r in enumerate(todo.itertuples(index=False), start=1):
            kic = int(r.kic_i)
            ra = float(r.ra)
            dec = float(r.dec)

            print(f"[{i}/{len(todo)}] KIC={kic} ...", end=" ", flush=True)

            got = None
            last_err = None
            for a in range(RETRIES):
                try:
                    got = run_one(service, ra, dec)
                    break
                except Exception as e:
                    last_err = str(e)
                    time.sleep(min(60, 0.5 * (2**a)))

            if got is None or len(got) == 0:
                fail += 1
                print(f"FAIL ({last_err})")
                append_row({
                    "kic": kic,
                    "ra": ra,
                    "dec": dec,
                    "source_id": "",
                    "phot_g_mean_mag": "",
                    "ruwe": "",
                    "dist_deg": "",
                    "status": "fail",
                    "error": (last_err or "")[:200],
                })
                continue

            best = got.iloc[0]
            sid = str(best.get("source_id_str","")).strip()

            append_row({
                "kic": kic,
                "ra": ra,
                "dec": dec,
                "source_id": sid,
                "phot_g_mean_mag": safe_float(best.get("phot_g_mean_mag")),
                "ruwe": safe_float(best.get("ruwe")),
                "dist_deg": safe_float(best.get("dist_deg")),
                "status": "ok",
                "error": "",
            })
            ok += 1
            print("OK", "source_id=", sid)

            if ok % 10 == 0:
                finalize()
                dt = time.time() - t0
                print(f"checkpoint: ok={ok} fail={fail} elapsed={dt:.1f}s")
                sys.stdout.flush()

            time.sleep(SLEEP_BETWEEN)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial results kept at:", OUT_TMP)

    finalize()
    print("done. ok=", ok, "fail=", fail)

if __name__ == "__main__":
    main()
