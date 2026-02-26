import os, yaml, time
import pandas as pd
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"

IDS_CSV = "/mnt/g/STAR_HPV/interim/legacy/legacy_gaia_ids_fixed.csv"
OUT_DIR = "/mnt/g/STAR_HPV/raw/gaia_dr3/debug_legacy_ids"
OUT_YAML = f"{OUT_DIR}/legacy_ids_presence.yaml"

TABLES = [
    "gaiadr3.gaia_source",
    "gaiadr3.gaia_source_lite",
    "gaiaedr3.gaia_source",
    "gaiadr2.gaia_source",
]

CHUNK = 25
RETRIES = 4

def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n], i//n

def query_ids(service, table_fq, ids_sql):
    q = f"SELECT source_id FROM {table_fq} WHERE source_id IN ({ids_sql})"
    job = service.submit_job(q, language=LANG)
    job.run()
    job.wait(phases=["COMPLETED","ERROR","ABORTED"], timeout=300)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"{table_fq} phase={job.phase}")
    df = job.fetch_result().to_table().to_pandas()
    return set(pd.to_numeric(df["source_id"], errors="coerce").dropna().astype("Int64").astype(int).tolist())

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    ids = pd.read_csv(IDS_CSV, dtype={"source_id":"string"})["source_id"].dropna().astype("string")
    ids = ids.str.strip().str.replace(r"\.0$", "", regex=True)
    ids = pd.to_numeric(ids, errors="coerce").dropna().astype("Int64").astype(int).unique().tolist()
    ids_set = set(ids)

    service = pyvo.dal.TAPService(TAP_URL)

    presence = {t: set() for t in TABLES}

    for t in TABLES:
        for chunk, idx in chunker(ids, CHUNK):
            ids_sql = ",".join(str(x) for x in chunk)
            ok = False
            for a in range(RETRIES):
                try:
                    found = query_ids(service, t, ids_sql)
                    presence[t] |= found
                    ok = True
                    break
                except Exception as e:
                    time.sleep(min(30, 2**a))
                    last = str(e)
            if not ok:
                print("chunk failed:", t, idx, last)

        print(t, "found", len(presence[t]), "/", len(ids))

    out = {
        "ids_requested": len(ids),
        "tables": {},
    }
    for t in TABLES:
        found = presence[t]
        missing = sorted(list(ids_set - found))
        out["tables"][t] = {
            "found_n": len(found),
            "missing_n": len(missing),
            "missing_ids": missing[:200],  # keep yaml small
        }

    # Also compute the “best” table by coverage
    best = max(TABLES, key=lambda t: len(presence[t]))
    out["best_table_by_coverage"] = best
    out["best_found_n"] = len(presence[best])

    with open(OUT_YAML, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT_YAML)
    print("best:", best, len(presence[best]), "/", len(ids))

if __name__ == "__main__":
    main()
