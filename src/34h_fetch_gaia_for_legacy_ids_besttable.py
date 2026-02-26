import os, glob, time, yaml
import pandas as pd
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"

IDS_CSV = "/mnt/g/STAR_HPV/interim/legacy/legacy_gaia_ids_fixed.csv"
PRES_YAML = "/mnt/g/STAR_HPV/raw/gaia_dr3/debug_legacy_ids/legacy_ids_presence.yaml"

OUT_DIR = "/mnt/g/STAR_HPV/raw/gaia_dr3/legacy_ids_parts_besttable"
OUT_MERGED = "/mnt/g/STAR_HPV/raw/gaia_dr3/gaia_legacy_ids_besttable.parquet"
OUT_SUM = "/mnt/g/STAR_HPV/raw/gaia_dr3/gaia_legacy_ids_besttable_summary.yaml"

CHUNK = 25
RETRIES = 5
SLEEP = 0.5

SELECT_COLS = [
    "source_id","ra","dec",
    "parallax","parallax_error",
    "phot_g_mean_mag","bp_rp","ruwe",
    "teff_gspphot","logg_gspphot","mh_gspphot",
]

def chunker(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n], i//n

def fetch(service, table_fq, ids_sql):
    cols_sql = ", ".join(f"gs.{c}" for c in SELECT_COLS)
    q = f"""
    SELECT {cols_sql}
    FROM {table_fq} AS gs
    WHERE gs.source_id IN ({ids_sql})
    """
    job = service.submit_job(q, language=LANG)
    job.run()
    job.wait(phases=["COMPLETED","ERROR","ABORTED"], timeout=300)
    if job.phase != "COMPLETED":
        raise RuntimeError(f"phase={job.phase}")
    return job.fetch_result().to_table().to_pandas()

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    with open(PRES_YAML, "r") as f:
        meta = yaml.safe_load(f)
    table_fq = meta["best_table_by_coverage"]
    print("using table:", table_fq)

    ids = pd.read_csv(IDS_CSV, dtype={"source_id":"string"})["source_id"].dropna().astype("string")
    ids = ids.str.strip().str.replace(r"\.0$", "", regex=True)
    ids = pd.to_numeric(ids, errors="coerce").dropna().astype("Int64").astype(int).unique().tolist()
    print("ids:", len(ids))

    service = pyvo.dal.TAPService(TAP_URL)

    for chunk, idx in chunker(ids, CHUNK):
        outp = os.path.join(OUT_DIR, f"gaia_legacy_ids_part{idx:03d}.parquet")
        if os.path.exists(outp):
            continue
        ids_sql = ",".join(str(x) for x in chunk)

        ok = False
        for a in range(RETRIES):
            try:
                df = fetch(service, table_fq, ids_sql)
                df.to_parquet(outp, index=False)
                print("saved", outp, "rows", len(df))
                ok = True
                break
            except Exception as e:
                wait = min(60, 2**a)
                print(f"part {idx} attempt {a+1}/{RETRIES} failed: {e} ; sleep {wait}s")
                time.sleep(wait)

        if not ok:
            raise RuntimeError(f"part {idx} failed after retries")

        time.sleep(SLEEP)

    files = sorted(glob.glob(os.path.join(OUT_DIR, "gaia_legacy_ids_part*.parquet")))
    g = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True) if files else pd.DataFrame()
    g.to_parquet(OUT_MERGED, index=False)

    summary = {
        "ids_requested": int(len(ids)),
        "rows_returned": int(len(g)),
        "table_used": table_fq,
        "columns": list(g.columns),
        "out_merged": OUT_MERGED,
        "n_parts": len(files),
        "tap_url": TAP_URL,
    }
    with open(OUT_SUM, "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("merged:", OUT_MERGED, "rows", len(g))
    print("summary:", OUT_SUM)

if __name__ == "__main__":
    main()
