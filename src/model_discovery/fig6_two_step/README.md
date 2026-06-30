# Figure 6 Two-Step Model Discovery

This module contains Figure 6 code for two-step fMRI prediction and cognitive model discovery.

Recommended environment from the release root:

```bash
pip install -r requirements/requirements_lit.txt
```

## Neural Prediction

- `neural/fit.py`: ridge regression on Schaefer-100 ROI beta estimates.
- `neural/extraction/`: feature extractors for language, embedding, SAE, and NeuroCogMap parcel features.
- `neural/analysis_results/`: panel-level aggregation and plotting helpers.

Inputs are under `../../../data/model_discovery/two_step/neural/`.

Included neural inputs:

- `schaefer_parcels_100.csv`: Schaefer-100 ROI beta table.
- `prompts_reformatted.jsonl`: reformatted two-step prompts used for feature extraction.

## Cognitive Models

- `openloop/models.py`: original Dual-systems Model and random baseline.
- `openloop/improved_cog_model_cogneuromap_simple.py`: behaviour-only discovered model.
- `openloop/improved_cog_model_cogneuromap_full.py`: NeuroCogMap-guided discovered model.
- `openloop/trainers.py` and `openloop/openloop_together.py`: Centaur-style fitting and AIC computation.

Held-out two-step task inputs are under `../../../data/model_discovery/two_step/openloop/`.

Included held-out inputs:

- `kool2016when/exp2.csv`
- `kool2017cost/exp2.csv`

## Model Roles

- Original model: `DualSystems` in `openloop/models.py`.
- Behaviour-only discovered model: `DualSystemsHierarchicalBanditCogneuromapSimple`.
- NeuroCogMap-guided discovered model: `DualSystemsHierarchicalBanditCogneuromapFull`.

## Typical Workflow

1. Use `neural/extract.py` or `neural/extract_with_token_limit.py` to extract model features.
2. Use `neural/fit.py` for participant-level ridge fitting and ROI-wise prediction evaluation.
3. Use `openloop/openloop_together.py` with the desired model entries enabled in the `experiments` list to fit held-out two-step datasets.
4. Compare participant-level AIC values with the provided comparison scripts.

Bulk fit outputs are intentionally excluded from this release.
