# derived

This directory stores reorganized data products created from the raw layer.

The role of the derived layer is to separate:

1. source-preserving raw storage,
2. intermediate restructuring,
3. final pipeline-ready input fixing.

In this repository, the derived layer is the stage that stands between
source provenance and executable validation input.

It should therefore be read as the **data preparation layer**
rather than as either the source archive or the final output layer.

---

## Purpose of the derived layer

The derived layer exists so that later pipeline stages do **not** read
directly from the raw source materials.

Instead, the workflow is:

- preserve source provenance in `data/raw/`
- reorganize or preprocess data in `data/derived/`
- fix official input CSV files under case-specific `input/` branches
- run skeleton, standard, topological, and integration stages from those fixed inputs

This means that the repository follows the practical flow:

- raw
- derived
- derived/input
- python execution
- results

This ordering is part of the repository design and is maintained so that
later validation stages can remain reproducible and structurally explicit.