import os, glob, time, yaml
import pandas as pd
import pyvo

CFG_DEFAULT = {
    "ids_csv": "/mnt/g/STAR_HPV/interim/legacy/legacy_gaia_ids.csv",
    "tap_url": "https://gaia.aip.de/tap",
    "language": "ADQL",
    "chunk_size": 200,   # safe for IN clause
    "sleep_s": 0.5,
    "out_dir": "/mnt/g/STAR_HPV/raw/gaia_dr3/legacy_ids_parts",
    "out_merged": "/mnt/g/STAR_HPV/raw/gaia_dr3/gaia_legacy_ids.parquet",
    "out_summary": "/mnt/g/STAR_HPV/raw/gaia_dr3/gaia_legacy_ids_summary.yaml",
}

# Columns needed for the state vector (plus some diagnostics)
SELECT_COLS = [
    "source_id", "ra", "dec",
    "parallax", "parallax_error",
    "phot_g_mean_mag", "bp_rp",
    "ruwe",
    "teff_gspphot", "logg_gspphot", "mh_gspphot",
]

def chunks(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n], i//n

def main():
    cfg = CFG_DEFAULT
    os.makedirs(cfg["out_dir"], exist_ok=True)

    ids = pd.read_csv(cfg["ids_csv"], dtype={"source_id":"string"})["source_id"].dropna().astype("string")
    ids = ids.str.strip().str.replace(r"\.0$", "", regex=True)
    ids = pd.to_numeric(ids, errors="coerce").dropna().astype("Int64").unique().tolist()
    print("ids:", len(ids))

    service = pyvo.dal.TAPService(cfg["tap_url"])

    part_paths = []
    for chunk, idx in chunks(ids, cfg["chunk_size"]):
        outp = os.path.join(cfg["out_dir"], f"gaia_legacy_ids_part{idx:03d}.parquet")
        part_paths.append(outp)
        if os.path.exists(outp):
            continue

        ids_sql = ",".join(str(int(x)) for x in chunk)
        cols_sql = ", ".join(f"gs.{c}" for c in SELECT_COLS)
        q = f"""
        SELECT {cols_sql}
        FROM gaiadr3.gaia_source AS gs
        WHERE gs.source_id IN ({ids_sql})
        """

        # Use async to avoid timeouts
        job = service.submit_job(q, language=cfg["language"])
        job.run()
        job.wait(phases=["COMPLETED","ERROR","ABORTED"], timeout=300)

        if job.phase != "COMPLETED":
            raise RuntimeError(f"chunk {idx} job phase={job.phase}")

        df = job.fetch_result().to_table().to_pandas()
        df.to_parquet(outp, index=False)
        print("saved", outp, "rows", len(df))
        time.sleep(cfg["sleep_s"])

    # Merge
    files = sorted(glob.glob(os.path.join(cfg["out_dir"], "gaia_legacy_ids_part*.parquet")))
    if not files:
        raise RuntimeError("No parts found.")
    g = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
    g.to_parquet(cfg["out_merged"], index=False)

    summary = {
        "ids_requested": int(len(ids)),
        "rows_returned": int(len(g)),
        "columns": list(g.columns),
        "out_merged": cfg["out_merged"],
    }
    with open(cfg["out_summary"], "w") as f:
        yaml.safe_dump(summary, f, sort_keys=False)

    print("merged:", cfg["out_merged"], "rows", len(g))
    print("summary:", cfg["out_summary"])

if __name__ == "__main__":
    main()
