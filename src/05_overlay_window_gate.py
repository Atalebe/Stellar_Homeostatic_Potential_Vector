import os, yaml
import numpy as np
import pandas as pd

NULL_YAML = "/mnt/g/STAR_HPV/results/nulls/null_metrics.yaml"
STATE_PARQ = "/mnt/g/STAR_HPV/processed/state_vectors/gaia_shv_state_vector.parquet"
OUT_YAML = "/mnt/g/STAR_HPV/results/gate/window_gate_overlay.yaml"

def main():
    with open(NULL_YAML,"r") as f:
        nm = yaml.safe_load(f)

    df = pd.read_parquet(STATE_PARQ)
    df = df[df["class_stage"]=="dwarf"].copy()
    df["log_prot"] = np.log10(df["prot"].astype(float))

    out = {"bins": {}}

    for tb, binfo in nm["bins"].items():
        gate = binfo["observed"]["gate_logprot"]
        if gate is None:
            continue

        sub = df[df["teff_bin"].astype(str)==tb].copy()
        if sub.empty:
            continue

        slow = sub["log_prot"] >= gate
        fast = sub["log_prot"] < gate

        # window membership concentration test
        f_win_slow = float(sub.loc[slow, "in_window"].mean()) if slow.any() else None
        f_win_fast = float(sub.loc[fast, "in_window"].mean()) if fast.any() else None

        # enrichment ratio, >1 means window prefers slow side
        enrich = None
        if f_win_fast not in (None, 0.0) and f_win_slow is not None:
            enrich = float(f_win_slow / f_win_fast)

        out["bins"][tb] = {
            "rows": int(len(sub)),
            "gate_logprot": float(gate),
            "gate_period_days": float(10**gate),
            "frac_slow_side": float(slow.mean()),
            "window_frac_slow_side": f_win_slow,
            "window_frac_fast_side": f_win_fast,
            "window_enrichment_slow_over_fast": enrich,
        }

    os.makedirs(os.path.dirname(OUT_YAML), exist_ok=True)
    with open(OUT_YAML,"w") as f:
        yaml.safe_dump(out, f, sort_keys=False)

    print("saved:", OUT_YAML)

if __name__=="__main__":
    main()
