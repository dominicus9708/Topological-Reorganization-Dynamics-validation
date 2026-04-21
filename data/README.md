# data

This directory stores the observational inputs used in the validation repository.

The data layer is divided into two main parts:

- `raw/`  
  Preserves source-level materials before validation-ready restructuring.

- `derived/`  
  Stores processed tables and fixed input CSV files that are used by the
  standard, skeleton, and topological pipelines.

The purpose of this directory is not only to hold files, but also to preserve
the distinction between source provenance, intermediate restructuring, and
final pipeline-ready inputs.

---

## Structure

### `raw/`

The `raw/` branch stores source-level materials before they are reorganized
into validation-ready input tables.

Its role is to preserve provenance and source separation.

#### Mercury raw structure

The Mercury raw-data structure is organized as:

- `literature/`
- `raw_data/`
- `registry/`
- `manifest/`

This separation is intentional.

- `literature/` stores paper PDFs, article files, and literature-origin source material.
- `raw_data/` stores CSV, TXT, supplementary tables, and machine-readable raw inputs.
- `registry/` stores source-registry snapshots used at execution time.
- `manifest/` stores execution manifests describing download results, local paths, and hashes.

The Mercury raw branch is therefore designed to preserve a strict distinction
between literature provenance and raw numerical inputs before later processing
into derived and input stages. :contentReference[oaicite:6]{index=6}

#### GW150914 raw structure

The GW150914 raw-data structure is organized as:

- `metadata/`
- `strain/`
- `registry/`
- `manifest/`

Its roles are as follows:

- `metadata/` stores event metadata JSON obtained from the official GWOSC event detail API.
- `strain/` stores raw detector strain files for GW150914
  (H1 and L1, 32 s, 4 kHz, HDF5).
- `registry/` stores the internal source-registry snapshot used by the pipeline.
- `manifest/` stores download status, file paths, sizes, and hashes. :contentReference[oaicite:7]{index=7}

#### GW170817 raw structure

The GW170817 raw-data structure is organized as:

- `metadata/`
- `strain/`
- `registry/`
- `manifest/`

Its roles are as follows:

- `metadata/` stores event metadata JSON obtained from the official GWOSC event detail API,
  GRB 170817A timing notes from NASA GCN, and strain listing debug dumps.
- `strain/` stores the H1, L1, and V1 strain files downloaded from the confirmed GWOSC listing.
  The current default target is 4096 s, 4 kHz, HDF.
- `registry/` stores the internal source-registry snapshot used by the pipeline.
- `manifest/` stores download status, file paths, sizes, and hashes. :contentReference[oaicite:8]{index=8}

---

### `derived/`

The `derived/` branch stores reorganized data products created from the raw branch.

Its purpose is to separate:

1. source-preserving preprocessing,
2. final input fixing,
3. later pipeline execution.

This means the repository follows the practical rule:

- `raw -> derived -> input -> results`

rather than reading source files directly from the raw layer inside the
standard or topological stages. This is consistent with the validation
workflow rule that final input CSV files should be fixed first and then
reused across later stages without re-selection or redefinition. 

#### Mercury derived structure

For Mercury, the derived branch currently fixes the final orbital input in:

- `derived/mercury/input/`

Representative files include:

- `mercury_elements_input_2026.csv`
- `mercury_vectors_input_2026.csv`
- `mercury_orbit_summary_input_2026.csv`

The corresponding input manifest records 13 element rows and 73 vector rows,
together with the fixed orbital summary values such as mean eccentricity,
mean semi-major axis, and the perihelion-like and aphelion-like dates. :contentReference[oaicite:10]{index=10}

#### GW150914 derived structure

For GW150914, the derived branch is split into:

- `derived/gravitational waves/GW150914/input/`
- `derived/gravitational waves/GW150914/derived/`

The fixed input branch contains:

- `gw150914_h1_input_4khz_32s.csv`
- `gw150914_l1_input_4khz_32s.csv`
- `gw150914_h1_l1_input_4khz_32s.csv`
- `gw150914_event_summary_input.csv`

The derived branch contains:

- detector-level strain CSV exports,
- merged strain exports,
- metadata summary output.

The current manifests indicate 131072 rows for H1, L1, and merged input,
with 32 s duration and 4 kHz sampling. 

#### GW170817 derived structure

For GW170817, the derived branch is split into:

- `derived/gravitational waves/GW170817/input/`
- `derived/gravitational waves/GW170817/derived/`

The fixed input branch contains:

- `gw170817_h1_input_4khz_4096s.csv`
- `gw170817_l1_input_4khz_4096s.csv`
- `gw170817_v1_input_4khz_4096s.csv`
- `gw170817_h1_l1_v1_input_4khz_4096s.csv`
- `gw170817_event_summary_input.csv`

The derived branch contains:

- detector-level strain CSV exports,
- merged H1-L1-V1 strain output,
- metadata summary output.

The current manifests indicate 16777216 rows for each detector input and for the merged input,
with 4096 s duration and 4 kHz sampling. 

---

## Data Handling Principle

The `data/` directory is designed around the following handling principle:

- preserve source provenance at the raw stage,
- reorganize data conservatively at the derived stage,
- fix official input CSV files before later execution,
- prevent standard and topological pipelines from redefining the input itself.

In this repository, the final input CSV files are treated as the official
common input for later validation stages.
After the input stage is fixed, later stages are expected to add computed
columns, coordinate transformations, or structural proxy quantities,
but not to redefine or reselect the input sample itself. :contentReference[oaicite:13]{index=13}

---

## Relationship to the Repository

This `data/` directory should be read together with the repository-level
validation logic.

- `raw/` preserves provenance and source separation.
- `derived/` fixes validation-ready inputs.
- later pipeline stages read these fixed inputs and write their outputs into `results/`.

Accordingly, the `data/` branch is the repository layer that connects
source materials to reproducible validation inputs.