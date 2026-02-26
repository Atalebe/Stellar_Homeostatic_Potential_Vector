import os, yaml
import pyvo
import pandas as pd

TAP_URL = "https://gea.esac.esa.int/tap-server/tap"
LANG = "ADQL"

OUT = "/mnt/g/STAR_HPV/raw/gaia_dr3/esa_gaia_source_table.yaml"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    service = pyvo.dal.TAPService(TAP_URL)

    # Tap schema: list tables with name like '%gaia_source%'
    q = """
    SELECT table_schema, table_name
    FROM TAP_SCHEMA.tables
    WHERE LOWER(table_name) LIKE '%gaia_source%'
    ORDER BY table_schema, table_name
    """

    res = service.run_sync(q, language=LANG)
    df = res.to_table().to_pandas()
    if len(df) == 0:
        raise RuntimeError("No gaia_source-like tables found in TAP_SCHEMA.tables")

    # Prefer exact 'gaia_source' within a schema containing 'gaiadr3'
    df["schema_l"] = df["table_schema"].astype(str).str.lower()
    df["name_l"] = df["table_name"].astype(str).str.lower()

    # Candidates ranked
    df["rank"] = 999
    df.loc[(df["name_l"] == "gaia_source") & (df["schema_l"].str.contains("gaiadr3")), "rank"] = 0
    df.loc[(df["name_l"] == "gaia_source") & (df["schema_l"].str.contains("gaia")), "rank"] = 1
    df.loc[(df["name_l"].str.contains("gaia_source")) & (df["schema_l"].str.contains("gaiadr3")), "rank"] = 2
    df.loc[(df["name_l"].str.contains("gaia_source")) & (df["schema_l"].str.contains("gaia")), "rank"] = 3
    df = df.sort_values(["rank","table_schema","table_name"])

    chosen = df.iloc[0]
    table_fq = f"{chosen['table_schema']}.{chosen['table_name']}"

    out = {
        "tap_url": TAP_URL,
        "chosen_table_schema": str(chosen["table_schema"]),
        "chosen_table_name": str(chosen["table_name"]),
        "chosen_table_fq": table_fq,
        "candidates": df[["table_schema","table_name","rank"]].head(30).to_dict(orient="records"),
    }
    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print("chosen:", table_fq)
    print("top candidates:")
    print(df[["table_schema","table_name","rank"]].head(12).to_string(index=False))

if __name__ == "__main__":
    main()
