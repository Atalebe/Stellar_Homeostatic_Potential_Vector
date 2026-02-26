import os, time, yaml
import pandas as pd
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"

MAP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_to_gaia_sourceid_string_r5arcsec.csv"
OUT_DIR = "/mnt/g/STAR_HPV/raw/gaia_dr3/debug_legacy_ids_r5map"
OUT_YAML = f"{OUT_DIR}/legacy_ids_presence.yaml"
OUT_MISSING_CSV = f"{OUT_DIR}/legacy_ids_missing_by_table.csv"

TABLES = [
    "gaiadr3.gaia_source",
    "gaiaedr3.gaia_source",
    "gaiadr2.gaia_source",
]

CHUNK = 50
RETRIES = 5
BACKOFF0 = 1.0

def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n], i//n

def fetch_found_ids(service, table_fq, ids_sql):
    q = f"SELECT source_id FROM {table_fq} WHERE source_id IN ({ids_sql})"
    job = service.submit_job(q, language=LANG)
    job.run()
    job.wait(phases=["COMPLETED","ERROR","ABORTED"], timeout=240)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"{table_fq} phase={job.phase}")
    df = job.fetch_result().to_table().to_pandas()
    s = pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("Int64").astype(int)
    return set(s.tolist())

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    mp = pd.read_csv(MAP, dtype={"source_id":"string"}, low_memory=False)
    mp = mp[mp["status"].astype(str) == "ok"].copy()

    ids = mp["source_id"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True)
    ids_int = pd.to_numeric(ids, errors="coerce").dropna().astype("Int64").astype(int).unique().tolist()
    ids_set = set(ids_int)

    service = pyvo.dal.TAPService(TAP_URL)

    presence = {}
    for t in TABLES:
        found_all = set()
        for chunk, idx in chunker(ids_int, CHUNK):
            ids_sql = ",".join(str(x) for x in chunk)
            ok = False
            last = None
            for a in range(RETRIES):
                try:
                    found = fetch_found_ids(service, t, ids_sql)
                    found_all |= found
                    ok = True
                    break
                except Exception as e:
                    last = str(e)
                    time.sleep(min(60, BACKOFF0 * (2**a)))
            if not ok:
                print("chunk failed:", t, idx, last)
        presence[t] = found_all
        print(t, "found", len(found_all), "/", len(ids_int))

    out = {
        "ids_requested": int(len(ids_int)),
        "tables": {},
    }

    rows_missing = []
    for t in TABLES:
        found = presence[t]
        missing = sorted(list(ids_set - found))
        out["tables"][t] = {
            "found_n": int(len(found)),
            "missing_n": int(len(missing)),
            "missing_ids_first50": missing[:50],
        }
        for m in missing:
            rows_missing.append({"table": t, "missing_source_id": str(m)})

    best = max(TABLES, key=lambda t: len(presence[t]))
    out["best_table_by_coverage"] = best
    out["best_found_n"] = int(len(presence[best]))
    out["tap_url"] = TAP_URL
    out["map_used"] = MAP

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    pd.DataFrame(rows_missing).to_csv(OUT_MISSING_CSV, index=False)

    print("saved:", OUT_YAML)
    print("saved:", OUT_MISSING_CSV)
    print("best:", best, out["best_found_n"], "/", out["ids_requested"])

if __name__ == "__main__":
    main()
