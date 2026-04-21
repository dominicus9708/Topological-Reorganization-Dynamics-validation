# raw

This directory stores source-level observational materials before they are
reorganized into validation-ready input tables.

The raw layer is intended to preserve provenance as explicitly as possible.
Its role is not to provide the final pipeline inputs directly, but to keep
track of where the source materials came from, how they were downloaded,
and how they were separated before later restructuring in `data/derived/`.

In this repository, the raw layer is organized by validation case rather
than by one universal file type, because the source structures themselves
are not identical across Mercury and gravitational-wave cases.

---

## Purpose of the raw layer

The raw layer exists to preserve four things:

1. source provenance,
2. original or minimally touched source files,
3. execution-time registry records,
4. download and file-state manifests.

Accordingly, raw data in this repository should be understood as
**provenance-preserving storage**, not as the final common input consumed
by later validation pipelines.

---

## Case structure overview

### Mercury

The Mercury raw-data structure is organized as:

- `literature/`
- `raw_data/`
- `registry/`
- `manifest/`

Their roles are:

- `literature/`  
  Stores paper PDFs, article files, or literature-origin source files.

- `raw_data/`  
  Stores CSV, TXT, supplementary tables, or other machine-readable raw inputs.

- `registry/`  
  Stores source-registry snapshots used at execution time.

- `manifest/`  
  Stores execution manifests describing download results, local paths, and hashes.

This branch is intentionally separated so that literature provenance and
raw numerical inputs remain distinct before later processing into the
derived and input stages.