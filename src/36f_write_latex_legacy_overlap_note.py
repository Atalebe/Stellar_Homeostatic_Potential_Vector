import os
import yaml

INP = "/mnt/g/STAR_HPV/results/ripeness/legacy_vs_kepler_teffbin_comparison.yaml"
OUT = "/mnt/g/STAR_HPV/results/ripeness/legacy_overlap_note.tex"

def pct(x):
    return f"{100.0*x:.1f}\\%"

def main():
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    d = yaml.safe_load(open(INP, "r"))

    n_leg = int(d["legacy_total_ids"])
    n_overlap = int(d["legacy_matched_to_kepler_rows"])

    kep = d["teffbin_distribution_kepler_gmmgate"]
    leg = d["teffbin_distribution_legacy_in_kepler"]

    # pull fractions safely
    fk = kep.get("fractions", {})
    fl = leg.get("fractions", {})
    ck = kep.get("counts", {})
    cl = leg.get("counts", {})

    # ordering
    bins = ["3200-4000", "4000-5200", "5200-6000", "6000-7500"]

    # build short inline text (no tables, keep it robust)
    kep_parts = []
    leg_parts = []
    for b in bins:
        if b in fk:
            kep_parts.append(f"{b}: {ck.get(b,0)} ({pct(float(fk[b]))})")
        if b in fl:
            leg_parts.append(f"{b}: {cl.get(b,0)} ({pct(float(fl[b]))})")

    kep_line = "; ".join(kep_parts)
    leg_line = "; ".join(leg_parts) if leg_parts else "none"

    tex = rf"""
\FindingBlock{{Asteroseismic age calibration: overlap and selection bias}}{{
An intersection was attempted between Kepler LEGACY asteroseismic ages ($N={n_leg}$) and the Kepler--Gaia GMM-gated rotator sample ($N={int(kep["rows"])}$).
Only $N={n_overlap}$ targets overlapped after enforcing rotation-period availability and the post-gate selection, implying negligible age coverage for calibrating the extreme homeostatic tail.

The Kepler GMM-gated temperature-bin composition is: {kep_line}.
The overlapping LEGACY subset is strongly hot-biased: {leg_line}.
This reflects selection effects in asteroseismic target lists and does not indicate an identifier or query failure.

Accordingly, the present pipeline treats $P_{{\rm rot}}$ as the operational time proxy for ripeness, while large-$N$ absolute age calibration is deferred to broader Kepler asteroseismic age catalogs with published Gaia crossmatches.
}}
""".strip()

    with open(OUT, "w") as f:
        f.write(tex + "\n")

    print("saved:", OUT)

if __name__ == "__main__":
    main()
