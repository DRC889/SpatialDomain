# Data availability

This repository contains the analysis code for SpatialDomainAE. The raw spatial
transcriptomics data are obtained from public repositories as described below.

## Primary dataset — mouse MCAO ischemic stroke

- **Source:** Gene Expression Omnibus, accession **GSE233815** (Zucha et al., 2024, *PNAS*).
- **Platform:** 10x Genomics Visium.
- **Sections:** four time points — control, 1, 3, and 7 days post-injury (DPI).
- **Content used here:** 10,173 spots and 22 manually annotated spatial domains
  (covering normal brain anatomy and ischemic-damage subtypes).
- Download the raw Space Ranger / Seurat outputs from GEO and export them with
  `scripts/00_export_from_seurat.R`, then run `scripts/01_preprocess.py`.

## External benchmark — human DLPFC

- **Source:** the spatialLIBD dataset (Maynard et al., 2021, *Nature Neuroscience*).
- **Sections:** 12 dorsolateral prefrontal cortex sections with manual cortical-layer
  annotations.
- **Access:** via the `spatialLIBD` Bioconductor package, or http://spatial.libd.org/spatialLIBD/ .
- Used by `scripts/rev05_dlpfc.py` for the external-validation experiment.

## Preprocessing summary

All methods receive the same input: per-sample top-3,000 highly variable genes,
library-size normalized and log1p-transformed (see `scripts/01_preprocess.py` and
Section 2.2 of the manuscript). Spatial coordinates are taken from the Visium
metadata. No raw counts are redistributed in this repository; only derived,
non-identifying result tables are included under `results/`.
