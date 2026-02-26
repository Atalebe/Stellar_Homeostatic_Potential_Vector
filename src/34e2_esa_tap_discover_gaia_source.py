import os, yaml
import pyvo

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"
OUT = "/mnt/g/STAR_HPV/raw/gaia_dr3/esa_gaia_source_table.yaml"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    service = pyvo.dal.TAPService(TAP_URL)

    # TAP_SCHEMA.tables usually has schema_name + table_name on many services
    q = """
    SELECT schema_name, table_name
    FROM TAP_SCHEMA.tables
    WHERE 1=1
    """
    res = service.run_sync(q, language=LANG)
    # pyvo returns astropy table; avoid pandas dependency here
    tab = res.to_table()

    # Filter in Python to avoid SQL function portability issues
    rows = []
    for r in tab:
        s = str(r["schema_name"]) if "schema_name" in tab.colnames else ""
        t = str(r["table_name"]) if "table_name" in tab.colnames else ""
        if "gaia_source" in t.lower():
            rows.append({"schema_name": s, "table_name": t})

    if not rows:
        raise RuntimeError("No tables containing 'gaia_source' found in TAP_SCHEMA.tables")

    # Rank candidates: prefer schema containing 'gaiadr3' and exact table_name 'gaia_source'
    def rank(x):
        s = x["schema_name"].lower()
        t = x["table_name"].lower()
        if t == "gaia_source" and "gaiadr3" in s:
            return 0
        if t == "gaia_source" and "gaia" in s:
            return 1
        if "gaiadr3" in s:
            return 2
        return 3

    rows = sorted(rows, key=lambda x: (rank(x), x["schema_name"], x["table_name"]))
    chosen = rows[0]
    table_fq = f"{chosen['schema_name']}.{chosen['table_name']}" if chosen["schema_name"] else chosen["table_name"]

    out = {
        "tap_url": TAP_URL,
        "chosen_schema_name": chosen["schema_name"],
        "chosen_table_name": chosen["table_name"],
        "chosen_table_fq": table_fq,
        "n_candidates": len(rows),
        "top_candidates": rows[:30],
    }
    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print("chosen:", table_fq)
    print("top candidates:")
    for r in rows[:12]:
        print(" ", r["schema_name"], r["table_name"])

if __name__ == "__main__":
    main()
