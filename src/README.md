# src

This directory stores the executable Python code used in the validation repository.

The role of the `src/` layer is to connect the fixed input zone under
`data/derived/.../input/` to the output-producing stages under `results/`.

In practical terms, `src/` is the repository’s execution layer.

It is the place where fixed validation-ready inputs are read,
structural or baseline computations are performed,
and reproducible outputs are written into the corresponding results branches.

---

## Code placement principle

In this repository, the basic code path follows the rule:

- `src/<validation target>/<observational case>/`

The practical meaning of this rule is:

- code is grouped first by validation domain,
- then by observational target or case,
- and finally by execution stage.

This structure is intentional.
It keeps the execution logic separated by case,
so that standard and topological interpretations do not become entangled
inside one oversized script branch.

---

## Stage structure

Each observational case under `src/` is expected to follow the same basic stage logic.

### 1. `skeleton`

The `skeleton` stage is the pre-execution structural check.

Its purpose is to verify that:

- paths are connected correctly,
- input files can be loaded,
- output folders can be created,
- the basic execution chain is working.

This stage is not a physical interpretation stage.
It is a structural readiness check.

In practice, the repository treats the skeleton stage as mandatory.
If the skeleton stage is unstable,
the standard and topological stages should not be treated as reliable.

---

### 2. `standard`

The `standard` stage preserves the observational or detector-space baseline.

Its purpose is to reconstruct the conventional reference layer first.

This means:

- no topological correction is introduced at this stage,
- the observational baseline is summarized in a reproducible form,
- later interpretation is anchored to the same fixed input.

For this reason, the standard stage is treated as the first meaningful success condition.
Even if the result looks simple,
the stage is considered valid if it loads the fixed input correctly
and reproduces the baseline relation in a stable and traceable way.

---

### 3. `topological`

The `topological` stage introduces the structural-response layer
based on the topological reorganization framework.

Its purpose is not to replace the standard layer,
but to interpret the same fixed input through an added structural contrast,
response quantity, or reorganization variable.

Accordingly:

- the standard stage preserves the baseline,
- the topological stage adds the structural interpretation,
- both stages should remain comparable because they begin from the same official input.

This separation is one of the core practical rules of the repository.

---

## Optional extension stages

Some observational cases may require additional stages beyond the basic three.

Examples include:

- mass-volume trial stages,
- bridge-variable stages,
- integration-only support scripts,
- comparison and report-generation scripts.

These are not treated as universal default stages.
They are conditional extensions used only when the corresponding case requires them.

In other words:

- `skeleton`
- `standard`
- `topological`

are the basic structure,

whereas trial or bridge stages are case-dependent extensions.

---

## Relationship to the data layer

The `src/` layer is expected to begin from the fixed input zone,
not from the raw source files.

The intended workflow is:

- preserve source provenance in `data/raw/`
- reorganize and fix reusable input in `data/derived/.../input/`
- execute validation logic in `src/`
- write outputs into `results/`

This means the executable scripts in `src/` should treat the fixed input CSV files
as the official common starting point.

After the input stage has been fixed,
later execution stages may compute additional columns,
structural proxy quantities, or response summaries,
but they should not silently redefine the input sample itself. :contentReference[oaicite:3]{index=3}

---

## Relationship to the results layer

The output layer of the repository is organized under `results/`.

In practical terms:

- `src/` performs execution,
- `results/` stores the outputs of that execution.

Typical output branches include:

- `skeleton`
- `standard`
- `topological`
- `Integration`

depending on the case.

This means the `src/` layer should be read together with the `results/` layer.
The former defines how the computation is carried out,
and the latter preserves what was produced.

---

## Case interpretation

The meaning of the execution stages is already visible in the current repository cases.

### Mercury

Mercury already shows the basic pattern:

- skeleton stage for path and input verification,
- standard stage for the weak-field orbital baseline,
- topological stage for the derived structural-response quantity,
- integration stage for standard versus topological comparison. 

### GW150914

GW150914 shows the same basic structure in a detector-mediated propagation setting:

- detector-space preprocessing,
- baseline preservation,
- local-envelope structural response,
- integration across standard and topological outputs. :contentReference[oaicite:5]{index=5}

### GW170817

GW170817 extends the same execution logic toward a heavier and more layered case:

- multi-detector preprocessing,
- fixed input generation,
- event-selection layer,
- reconstruction and structural-gap layer,
- response and relaxation layer,
- integration across standard and topological outputs. :contentReference[oaicite:6]{index=6}

---

## Summary

The `src/` directory is the repository’s execution layer.

Its purpose is to:

- read fixed validation-ready input,
- preserve the distinction between baseline and structural-response stages,
- execute reproducible case-specific validation logic,
- and write outputs into the `results/` hierarchy.

The most important practical rule is simple:

- do not begin from raw files at execution time,
- do not merge standard and topological logic into one opaque stage,
- and do not redefine the official input after it has been fixed.