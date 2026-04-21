# results

This directory stores the outputs produced by the validation pipelines.

The role of the `results/` layer is to preserve the execution outputs of
the repository in a stage-separated and case-separated form.

In practical terms:

- `data/` preserves source materials and fixed inputs,
- `src/` performs the execution,
- `results/` stores what that execution produced.

This means the `results/` layer should be read as the repository’s
output-preservation branch.

---

## Purpose of the results layer

The results layer exists to preserve:

1. baseline output,
2. topological output,
3. stage-by-stage validation summaries,
4. integration reports and comparison figures.

It is not merely a storage folder for images or CSV files.
It is the execution record of the validation program.

For that reason, results are grouped both by case and by stage.

---

## Stage structure

Depending on the case, the output branch may contain the following stages.

### `skeleton`

The `skeleton` stage stores path-verification and structural readiness outputs.

Its purpose is to confirm that:

- input loading worked,
- output folders were created correctly,
- stage connectivity is valid,
- later stages can be executed on a stable basis.

This stage is a structural execution check rather than a physical
interpretation stage.

### `standard`

The `standard` stage stores the preserved observational or detector-space baseline.

Its purpose is to record the conventional reference layer before any
topological interpretation is added.

Typical standard outputs include:

- baseline summaries,
- reference figures,
- derived baseline CSV tables,
- standard report text files.

### `topological`

The `topological` stage stores the structural-response interpretation layer.

Its purpose is to preserve outputs generated after the topological
reorganization framework is applied to the same fixed input.

Typical topological outputs include:

- structural-response CSV summaries,
- event-selection or structural-gap tables,
- response-model reports,
- topological figures.

### `Integration`

The `Integration` stage stores direct comparison outputs between the
standard and topological branches.

Its role is to preserve:

- comparison summaries,
- difference curves,
- integration figures,
- combined reports,
- final validation-oriented comparisons.

This stage is especially important because it makes the repository’s
consistency-oriented design explicit: the standard and topological layers
are interpreted together only after both have been preserved separately.

---

## Case overview

### Mercury

The Mercury results branch is organized under:

- `results/mercury/output/`

Its current main output stages are:

- `standard/`
- `topological/`
- `Integration/`

Representative outputs include:

- standard orbital baseline summaries,
- topological structural-response summaries,
- integration reports comparing orbital radius and structural response,
- Mercury comparison figures used in the paper. :contentReference[oaicite:4]{index=4}

Mercury is the weak-field projected orbital validation case of the repository.

---

### GW150914

The GW150914 results branch is organized under:

- `results/gravitational_wave/GW150914/output/`

Its current main output stages are:

- `skeleton/`
- `standard/`
- `topological/`
- `Integration/`

Representative outputs include:

- skeleton readiness reports,
- detector-space baseline summaries,
- derived local and envelope structural-response outputs,
- post-peak relaxation outputs,
- integration figures comparing baseline and topological response. 

GW150914 is the first detector-mediated gravitational-wave validation case
in the repository.

---

### GW170817

The GW170817 results branch is organized under:

- `results/gravitational_wave/GW170817/output/`

Its current main output stages are:

- `skeleton/`
- `standard/`
- `topological/`
- `Integration/`

Representative outputs include:

- skeleton readiness reports,
- standard detector-mean baseline summaries,
- topological event-selection outputs,
- reconstruction and structural-gap outputs,
- response-model comparison outputs,
- integration summaries and comparison figures. 

Typical integration outputs for GW170817 include files such as:

- `gw170817_event_selection_difference.png`
- `gw170817_reconstruction_comparison.png`
- `gw170817_response_model_difference.png`
- `gw170817_integration_summary.csv`
- `gw170817_integration_report.txt` :contentReference[oaicite:7]{index=7}

GW170817 is currently the most developed gravitational-wave validation case
in the repository.

---

## Output handling principle

The `results/` layer follows the same structural rule as the rest of the repository:

- preserve stages separately,
- keep the baseline explicit,
- keep the topological interpretation explicit,
- compare them only at the integration stage.

This means:

- standard and topological outputs should not be mixed into one opaque branch,
- integration outputs should remain identifiable as comparison products,
- stage names should reflect the role of the output, not just its file type.

The output philosophy of this repository is therefore not “one final result,”
but a structured set of execution products that preserve how each case was
validated.

---

## Relationship to the paper

The `results/` directory is the direct output counterpart of the paper’s
validation section.

In particular, the branches under this directory correspond to the
validation program developed in:

- **Validation of Topological Reorganization Dynamics**
  - Mercury perihelion precession
  - Gravitational-wave propagation
  - Integrated interpretation of the validation cases

The figures and summaries stored here should therefore be read as the
computational output layer of that part of the paper.

---

## Summary

The `results/` directory is the repository’s output-preservation layer.

It stores:

- stage-separated execution results,
- case-separated validation products,
- standard and topological outputs,
- integration reports and comparison figures.

Mercury emphasizes weak-field orbital validation.

GW150914 emphasizes detector-mediated baseline and envelope response.

GW170817 emphasizes event selection, reconstruction, structural-gap response,
and post-focus comparison.

Together, these branches preserve the executable validation record of the
repository.