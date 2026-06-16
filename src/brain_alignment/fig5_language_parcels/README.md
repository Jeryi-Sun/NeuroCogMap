# Figure 5 Language-Parcel Brain Alignment

This module contains the code needed to reproduce the Figure 5 language-related cortical parcel selection and NeuroCogMap encoding workflow.

Recommended environment from the release root:

```bash
pip install -r requirements/requirements_lit.txt
```

Key pieces:

- `litcoder_core/`: minimal LITcoder-style training and encoding entry points.
- `human_profile_generation/`: script used to convert Neurosynth/Cognitive Atlas term z-score profiles into natural-language human cortical parcel descriptions.
- `selection/select_language_overlapping_parcels_accuracy.py`: selects language-related Schaefer parcels by overlap with language ROI masks.
- `selection/filter_metrics_by_language_parcels.py`: filters prediction metrics to the selected language parcels.
- `data_preparation/` and `draw_code/`: helper scripts used for prediction matrices, top human--LLM matches, semantic similarity, and RSA analyses.

Small panel artifacts are stored in `../../../artifacts/figure_selections/fig5_language_parcels/`.

## Included Small Artifacts

- `language_parcel_overlap_and_accuracy.json`: final ten language-overlapping cortical parcels, retaining only parcel id/name and vertex-count fields.
- `panel_a_data/summary_results.csv`: compact NeuroCogMap-only panel-A result; baseline rows are not included in the release artifact.
- `merged_top_human_parcels_per_llm_whereisthesmoke_saeact.csv`: merged top human--LLM matches for the held-out story.
- `human_cortical_profiles/ns_scale100.csv`: Neurosynth/Cognitive Atlas term z-score profiles for Schaefer100 parcels.
- `human_cortical_profiles/human_cortical_parcel_descriptions.json`: LLM-naturalized function names, descriptions, and brain-role summaries for the 100 human cortical parcels.

## Human Cortical Profile Naturalization

For Figure 5c/d/e, each human Schaefer100 parcel was represented by a Neurosynth/Cognitive Atlas term-association vector. The generation script converts the top absolute z-score terms into natural-language fields:

- `function_name`
- `function_description`
- `role_in_human_brain`

These text profiles were used for human--LLM semantic similarity, LLM-judge correspondence scoring, and network-level RSA. They should be treated as a reproducibility artifact derived from the term z-score profiles, not as independent human annotation.

## Quick Audit

From the release root:

```bash
python - <<'PY'
import json
from pathlib import Path
path = Path("artifacts/figure_selections/fig5_language_parcels/language_parcel_overlap_and_accuracy.json")
data = json.load(open(path))
selected = data["selected_parcels"]
print(len(selected), [item["parcel_idx"] for item in selected])
PY
```

The selection should contain ten parcels.

## Full Refitting Requirements

Full model refitting requires LeBel/LITcoder-compatible fMRI assemblies and feature caches, which are not bundled.
