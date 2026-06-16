# Pathology Analysis, Detection, and Intervention

This module contains the Figure 3 and Figure 4 code for hallucination, social bias, jailbreak/refusal failure, and sycophancy.

Recommended environment from the release root:

```bash
pip install -r requirements/requirements_sae.txt
pip install -e third_party/sae_lens_neurocogmap
```

## Layout

Each pathology folder follows the same broad organization:

- `code/`: task generation, evaluation, activation extraction, and analysis scripts.
- `code/detection/`: NeuroCogMap-based detection code.
- `code/intervention/`: mechanism-guided steering and intervention evaluation.
- `*_graphs/` and `intervention_graph/`: plotting and aggregation utilities.

Raw public inputs are released under `../../data/pathology_raw/`.

## Workflow

The retained workflow is:

1. Generate or load task responses from the released raw inputs.
2. Derive task labels and split normative versus pathological examples.
3. Extract parcel-level activation signatures.
4. Run multilevel pathology analysis and detection.
5. Apply mechanism-guided intervention using the selected pathological parcels.

Generated model outputs, token-level activation caches, cross-validation outputs, and intervention result directories are intentionally excluded from this release.

## Intervention Settings

Default intervention launchers use fixed paper-selected strengths and do not perform a default strength search. To run an optional sensitivity sweep, explicitly set:

```bash
INTERVENTION_STRENGTHS="0.1 0.3 0.5" bash run_intervention.sh
```

Paths in the original shell scripts may need local adjustment before full reruns.
