# Data

Two literature spectral tables used by `optics_nir.py`. Both are loaded relative to the repository, so no download is needed to run the code.

## `roblyer_2025_stitched_extinction_dataset1.csv`

Stitched chromophore spectra, 430–2000 nm at 1 nm (1571 rows), converted unchanged from the published `Dataset 1.xlsx` to CSV.

| Column | Units |
| --- | --- |
| `source_row` | row number in the source workbook |
| `wavelength_nm` | nm |
| `oxy_extinction_mm_inv_per_uM` | mm⁻¹ µM⁻¹ (oxyhaemoglobin) |
| `deoxy_extinction_mm_inv_per_uM` | mm⁻¹ µM⁻¹ (deoxyhaemoglobin) |
| `water_mu_a_mm_inv` | mm⁻¹ (pure water) |
| `lipid_mu_a_mm_inv` | mm⁻¹ (lipid) |

**Source.** D. Roblyer, A. Pilvar, T. Pham, S. Spink, C. Hayakawa, V. Venugopalan and M. Pierce, "Dataset 1.xlsx", Optica Publishing Group, figshare (2025). https://doi.org/10.6084/m9.figshare.30005485.v1 — dataset page: https://opticapublishing.figshare.com/articles/dataset/Dataset_1_xlsx/30005485

Supplementary to: D. Roblyer, A. Pilvar, T. Pham, S. Spink, C. K. Hayakawa, V. Venugopalan and M. C. Pierce, "Review of shortwave infrared imaging and spectroscopy in tissue [Invited]," *Biomedical Optics Express* 16(12), 5028–5062 (2025). https://doi.org/10.1364/BOE.577163

**Licence.** Creative Commons Attribution 4.0 International (CC BY 4.0), https://creativecommons.org/licenses/by/4.0/, as stated on the figshare record. Changes made here: format conversion from XLSX to CSV and renamed column headers; values unchanged.

The table is a stitched compilation of earlier literature spectra, not a primary measurement. Its water column is Segelstein 1981 resampled.

## `segelstein_1981_water_absorption.csv`

Absorption coefficient of liquid water, 400–2500 nm (271 rows, the source spacing), from the table compiled by the Oregon Medical Laser Center (OMLC).

| Column | Units |
| --- | --- |
| `source_row` | line number in the OMLC text file |
| `wavelength_nm` | nm |
| `mu_a_cm_inv` | cm⁻¹ (as published) |
| `mu_a_mm_inv` | mm⁻¹ (= `mu_a_cm_inv` / 10) |

**Source.** D. J. Segelstein, "The complex refractive index of water," M.S. thesis, University of Missouri–Kansas City (1981). Numeric table as distributed by OMLC (S. Prahl): https://omlc.org/spectra/water/data/segelstein81.txt , listed at https://omlc.org/spectra/water/abs/index.html . Temperature not stated.

**Licence.** No licence is stated for the OMLC table (the OMLC site carries a general copyright notice). The values are factual measurements from a 1981 thesis and are widely redistributed in the tissue-optics literature and software, but redistribution terms could not be confirmed. If you prefer not to redistribute the file, delete it and regenerate it locally with:

```bash
python data/fetch_segelstein.py
```

which downloads the OMLC table and rewrites the identical CSV (standard library only).

## How the code extends the spectra

`optics_nir.py` uses the Roblyer table from 430 to 2000 nm and appends 2001–2500 nm: water from Segelstein (a seamless join, since the Roblyer water column is Segelstein), with lipid and haemoglobin held at their 2000 nm values. Lipid and haemoglobin are minor absorbers next to water in that band, but the flat extension is an approximation, not measured structure.
