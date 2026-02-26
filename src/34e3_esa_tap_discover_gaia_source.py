import os, yaml
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"
OUT = "/mnt/g/STAR_HPV/raw/gaia_dr3/esa_gaia_source_table.yaml"

def fq_name(schema_name, table_name):
    s = (schema_name or "").strip()
    t = (table_name or "").strip()
    # If table_name already contains a dot, treat as already qualified.
    if "." in t:
        return t
    if s:
        return f"{s}.{t}"
    return t

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    service = pyvo.dal.TAPService(TAP_URL)

    q = "SELECT schema_name, table_name FROM TAP_SCHEMA.tables"
    res = service.run_sync(q, language=LANG)
    tab = res.to_table()

    rows = []
    for r in tab:
        s = str(r["schema_name"]) if "schema_name" in tab.colnames else ""
        t = str(r["table_name"]) if "table_name" in tab.colnames else ""
        if "gaia_source" in t.lower():
            rows.append({"schema_name": s, "table_name": t})

    if not rows:
        raise RuntimeError("No tables containing 'gaia_source' found.")

    def rank(x):
        s = x["schema_name"].lower()
        t = x["table_name"].lower()
        # prefer DR3 main table
        if "gaiadr3" in s and t.endswith("gaia_source"):
            return 0
        if "gaiadr3" in s and "gaia_source_lite" in t:
            return 1
        if "gaiaedr3" in s and t.endswith("gaia_source"):
            return 2
        return 9

    rows = sorted(rows, key=lambda x: (rank(x), x["schema_name"], x["table_name"]))
    chosen = rows[0]
    chosen_fq = fq_name(chosen["schema_name"], chosen["table_name"])

    out = {
        "tap_url": TAP_URL,
        "chosen_schema_name": chosen["schema_name"],
        "chosen_table_name": chosen["table_name"],
        "chosen_table_fq": chosen_fq,
        "n_candidates": len(rows),
        "top_candidates": rows[:30],
    }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print("chosen:", chosen_fq)
    print("top candidates:")
    for r in rows[:12]:
        print(" ", r["schema_name"], r["table_name"])

if __name__ == "__main__":
    main()
