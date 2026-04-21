Mercury raw-data folder structure
================================

literature/
- Stores paper PDFs, article files, or literature-origin source files.

raw_data/
- Stores CSV, TXT, supplementary tables, or machine-readable raw inputs.

registry/
- Stores source registry snapshots used at execution time.

manifest/
- Stores execution manifests describing download results, local paths, and hashes.

This folder is intended to preserve a strict separation between literature provenance
and raw numerical/source data before later processing into derived/input stages.
