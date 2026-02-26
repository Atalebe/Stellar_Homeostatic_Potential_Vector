import os, time, sys
import pandas as pd
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_coords.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"
OUT_TMP = OUT.replace(".csv", ".partial.csv")

RADIUS_ARCSEC = 5.0
RADIUS_DEG = RADIUS_ARCSEC / 3600.0

RETRIES = 6
SLEEP_BETWEEN = 0.2
PER_TARGET_TIMEOUT = 180
GAIA_TABLE = "gaiadr3.gaia_source"

def build_query(ra, dec):
    return f"""
    SELECT
      CAST(gs.source_id AS VARCHAR) AS source_id_str,
      gs.ra, gs.dec,
      gs.parallax, gs.phot_g_mean_mag, gs.ruwe,
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
        # try to expose server-side message if present
        try:
            msg = str(job.error_summary)
        except Exception:
            msg = ""
        raise RuntimeError(f"phase={job.phase} {msg}".strip())
    return job.fetch_result().to_table().to_pandas()

def load_done():
    done = set()
    for p in [OUT, OUT_TMP]:
        if os.path.exists(p):
            df = pd.read_csv(p, dtype={"source_id":"string"}, low_memory=False)
            if "kic" in df.columns:
                done |= set(pd.to_numeric(df["kic"], errors="coerce").dropna().astype(int).tolist())
    return done

def append_row(d):
    df = pd.DataFrame([d])
    header = not os.path.exists(OUT_TMP)
    df.to_csv(OUT_TMP, mode="a", index=False, header=header)

def finalize():
    if os.path.exists(OUT_TMP):
        df = pd.read_csv(OUT_TMP, dtype={"source_id":"string"}, low_memory=False)
        df = df.drop_duplicates(subset=["kic"])
        df.to_csv(OUT, index=False)
        print("finalized ->", OUT, "rows:", len(df))

def choose_best(cands):
    # already ordered by distance ASC, but add a soft quality preference:
    # prefer ruwe < 1.4 if possible among the nearest few
    c = cands.copy()
    c["ruwe"] = pd.to_numeric(c.get("ruwe"), errors="coerce")
    c["dist_deg"] = pd.to_numeric(c.get("dist_deg"), errors="coerce")

    top = c.head(5).copy()
    good = top[top["ruwe"].fillna(99) < 1.4]
    if len(good) > 0:
        return good.sort_values("dist_deg").iloc[0]
    return top.sort_values("dist_deg").iloc[0]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    leg = pd.read_parquet(INP).copy()
    leg["kic"] = pd.to_numeric(leg["kic"], errors="coerce").astype("Int64")
    leg["ra"] = pd.to_numeric(leg["ra"], errors="coerce")
    leg["dec"] = pd.to_numeric(leg["dec"], errors="coerce")
    leg = leg.dropna(subset=["kic","ra","dec"]).copy()
    leg["kic_i"] = leg["kic"].astype(int)

    done = load_done()
    todo = leg[~leg["kic_i"].isin(done)].copy()

    print(f"targets total={len(leg)} done={len(done)} todo={len(todo)} radius={RADIUS_ARCSEC}\"")
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
                    time.sleep(min(60, 0.5*(2**a)))

            if got is None or len(got) == 0:
                fail += 1
                print(f"FAIL ({last_err})")
                append_row({
                    "kic": kic, "ra": ra, "dec": dec,
                    "source_id": "",
                    "dist_deg": "",
                    "ruwe": "",
                    "phot_g_mean_mag": "",
                    "parallax": "",
                    "status": "fail",
                    "error": (last_err or "")[:220],
                })
                continue

            best = choose_best(got)
            sid = str(best.get("source_id_str","")).strip()

            append_row({
                "kic": kic, "ra": ra, "dec": dec,
                "source_id": sid,
                "dist_deg": float(best.get("dist_deg")),
                "ruwe": float(best.get("ruwe")) if pd.notna(best.get("ruwe")) else "",
                "phot_g_mean_mag": float(best.get("phot_g_mean_mag")) if pd.notna(best.get("phot_g_mean_mag")) else "",
                "parallax": float(best.get("parallax")) if pd.notna(best.get("parallax")) else "",
                "status": "ok",
                "error": "",
            })
            ok += 1
            print("OK", "sid=", sid)

            if ok % 10 == 0:
                finalize()
                dt = time.time() - t0
                print(f"checkpoint: ok={ok} fail={fail} elapsed={dt:.1f}s")
                sys.stdout.flush()

            time.sleep(SLEEP_BETWEEN)

    except KeyboardInterrupt:
        print("\nInterrupted. Partial saved:", OUT_TMP)

    finalize()
    print("done. ok=", ok, "fail=", fail)

if __name__ == "__main__":
    main()
