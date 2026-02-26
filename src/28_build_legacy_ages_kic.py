import os
import numpy as np
import pandas as pd

INP = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_table4_per_pipeline.parquet"
OUT = "/mnt/g/STAR_HPV/raw/ages/legacy_2017_kic_ages.csv"

def mad(x):
    x = np.asarray(x, dtype=float)
    m = np.nanmedian(x)
    return np.nanmedian(np.abs(x - m))

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)

    df = pd.read_parquet(INP, columns=["kic","pipe","age_gyr"]).copy()
    df = df.dropna(subset=["kic","age_gyr"]).copy()

    g = df.groupby("kic")["age_gyr"]
    out = pd.DataFrame({
        "kic": g.size().index.astype("Int64"),
        "n_pipelines": g.size().values.astype(int),
        "age_gyr": g.median().values.astype(float),
        "age_gyr_mad": g.apply(mad).values.astype(float),
        "age_gyr_q16": g.quantile(0.16).values.astype(float),
        "age_gyr_q84": g.quantile(0.84).values.astype(float),
    })

    # A simple uncertainty: half-width of (q84-q16)
    out["age_gyr_err"] = 0.5 * (out["age_gyr_q84"] - out["age_gyr_q16"])

    out["age_source"] = "Kepler LEGACY asteroseismology (Silva Aguirre+ 2017)"
    # quality flag: require at least 4 pipelines contributing (tune if you want)
    out["age_quality_flag"] = (out["n_pipelines"] >= 4).astype(int)

    out.to_csv(OUT, index=False)
    print(f"saved: {OUT} rows={len(out)}")
    print(out.head(10))

if __name__ == "__main__":
    main()
