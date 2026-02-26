import os, yaml
import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture

INP = "/mnt/g/STAR_HPV/processed/state_vectors/kepler_led_state_vector.parquet"
OUT = "/mnt/g/STAR_HPV/results/gate/kepler_gmm_gate.yaml"

BINS = ["3200-4000","4000-5200","5200-6000","6000-7500"]

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df = pd.read_parquet(INP)

    out = {"bins": {}}

    for tb in BINS:
        sub = df[df["teff_bin"].astype(str) == tb].copy()
        sub = sub[np.isfinite(sub["prot_days"].astype(float)) & (sub["prot_days"].astype(float) > 0)]
        if len(sub) < 200:
            continue

        logP = np.log10(sub["prot_days"].astype(float).to_numpy()).reshape(-1,1)

        gmm = GaussianMixture(n_components=2, random_state=123).fit(logP)
        means = gmm.means_.flatten()
        sds = np.sqrt(gmm.covariances_.flatten())
        w = gmm.weights_.flatten()

        # order by mean
        idx = np.argsort(means)
        means, sds, w = means[idx], sds[idx], w[idx]

        # crude "transition": midpoint of the two means in log space
        gate_logP = float(0.5*(means[0] + means[1]))

        out["bins"][tb] = {
            "rows": int(len(sub)),
            "gmm_means_days": [float(10**means[0]), float(10**means[1])],
            "gmm_sigmas_logP": [float(sds[0]), float(sds[1])],
            "gmm_weights": [float(w[0]), float(w[1])],
            "gate_logP_mid": gate_logP,
            "gate_days_mid": float(10**gate_logP),
            "median_days": float(np.nanmedian(sub["prot_days"].astype(float))),
        }

    with open(OUT, "w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT)
    print(out)

if __name__ == "__main__":
    main()
