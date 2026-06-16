# Code Release Manifest

This manifest records the intended GitHub release contents.

## Included In GitHub

- `README.md`: code-release overview and installation notes.
- `docs/`: external-resource notes and smoke-test utilities.
- `requirements/`: curated dependency files for the SAE/pathology and LIT/brain-alignment environments.
- `src/neurocogmap_construction/`: NeuroCogMap construction code.
- `src/pathology/`: pathology analysis, detection, intervention, and plotting code.
- `src/brain_alignment/fig5_language_parcels/`: human cortical alignment and language-parcel code.
- `src/model_discovery/fig6_two_step/`: two-step fMRI prediction and model-discovery code.
- `src/neurocogmap_release/`: release-relative path helpers.
- `third_party/sae_lens_neurocogmap/`: patched SAELens source required by the SAE environment.

## Excluded From GitHub

The following categories are excluded and should be accessed through Zenodo or original providers:

- processed analysis artifacts and figure source data
- NeuroCogMap atlas JSON/CSV artifacts
- public benchmark datasets and raw pathology inputs
- generated model responses
- hidden-state, token-activation, and SAE/model cache files
- intervention run outputs and cross-validation outputs
- model weights and sparse-autoencoder weights

## Companion Zenodo Data Package

The companion Zenodo data package is available at https://zenodo.org/records/20629857. It is organized under `NeuroCogMap_Data/` and contains the small processed artifacts needed to audit and reproduce paper figures without redistributing restricted or third-party raw resources.
