import os
import requests
import pyvo

TAP_URL = "https://gaia.aip.de/tap"

def tap_service():
    token = os.getenv("GAIA_AIP_TOKEN", "").strip()
    if token:
        sess = requests.Session()
        sess.headers["Authorization"] = f"Token {token}"
        return pyvo.dal.TAPService(TAP_URL, session=sess)
    return pyvo.dal.TAPService(TAP_URL)

def main():
    s = tap_service()

    # 1) simplest possible query
    r1 = s.run_sync("SELECT TOP 1 source_id FROM gaiadr3.vari_rotation_modulation", language="ADQL")
    print("ok vari_rotation_modulation:", len(r1.to_table()))
    # 2) check if gaia_source has parallax_over_error, do not guess
    q = """
    SELECT c.column_name
    FROM gaia_tap_schema.columns AS c
    JOIN gaia_tap_schema.tables  AS t
      ON c.table_name = t.table_name
    WHERE t.schema_name='gaiadr3'
      AND c.table_name='gaia_source'
      AND c.column_name ILIKE '%over%';
    """
    r2 = s.run_sync(q, language="ADQL")
    print(r2.to_table())

if __name__ == "__main__":
    main()
