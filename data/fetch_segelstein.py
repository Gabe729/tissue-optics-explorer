"""Regenerate segelstein_1981_water_absorption.csv from the public OMLC table.

Downloads https://omlc.org/spectra/water/data/segelstein81.txt (Segelstein 1981,
compiled by OMLC), keeps 400-2500 nm, and writes the CSV used by optics_nir.py:

    source_row, wavelength_nm, mu_a_cm_inv, mu_a_mm_inv   (mu_a_mm_inv = mu_a_cm_inv / 10)

source_row is the 1-based line number in the source file. Values match the
bundled CSV numerically (trailing-zero formatting may differ). Standard library only:

    python data/fetch_segelstein.py
"""
import csv
import urllib.request
from pathlib import Path

URL = "https://omlc.org/spectra/water/data/segelstein81.txt"
OUT = Path(__file__).resolve().parent / "segelstein_1981_water_absorption.csv"


def main():
    text = urllib.request.urlopen(URL, timeout=30).read().decode("latin-1")
    rows = []
    for n, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 2:
            continue
        try:
            wl, mua = float(parts[0]), parts[1]
            float(mua)
        except ValueError:
            continue
        if 400 <= wl <= 2500:
            rows.append([n, parts[0], mua, f"{float(mua) / 10:.6g}"])
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["source_row", "wavelength_nm", "mu_a_cm_inv", "mu_a_mm_inv"])
        w.writerows(rows)
    print(f"wrote {OUT} ({len(rows)} rows)")


if __name__ == "__main__":
    main()
