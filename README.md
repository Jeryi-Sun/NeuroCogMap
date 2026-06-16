# NeuroCogMap Code Release

This repository contains the code release for the NeuroCogMap study. It includes scripts for NeuroCogMap construction, pathology analyses, detection and intervention experiments, human cortical alignment analyses, and NeuroCogMap-guided model discovery.

Processed data, figure source data, atlas artifacts, and aggregate evaluation summaries are provided separately in the companion Zenodo data package: https://zenodo.org/records/20629857.

## Included Components

- `src/neurocogmap_construction/`: NeuroCogMap construction code, including functional parcel extraction, structural-connectome construction, and parcel-capability mapping.
- `src/pathology/`: hallucination, social-bias, refusal-failure/jailbreak, and sycophancy analysis code.
- `src/brain_alignment/fig5_language_parcels/`: human cortical alignment and language-parcel analysis code.
- `src/model_discovery/fig6_two_step/`: two-step fMRI prediction and model-discovery code.
- `src/neurocogmap_release/`: release-relative path helpers.
- `docs/`: external-resource notes and smoke-test utilities.
- `requirements/`: curated Python dependency files.
- `third_party/sae_lens_neurocogmap/`: NeuroCogMap-patched SAELens source package used by the analysis code.

## Not Included

This code package does not redistribute:

- third-party model weights
- sparse-autoencoder weights
- public benchmark datasets
- generated model responses
- token-level hidden-state or activation caches
- bulk cross-validation or intervention outputs

These resources should be obtained from their original providers or regenerated locally where permitted.

## Installation

For NeuroCogMap construction and pathology analyses:

```bash
pip install -r requirements/requirements_sae.txt
pip install -e third_party/sae_lens_neurocogmap
```

For human cortical alignment and two-step model-discovery analyses:

```bash
pip install -r requirements/requirements_lit.txt
```

## External Resources

See `docs/EXTERNAL_RESOURCES.md` for required external model, SAE, embedding, and neuroimaging resources. Several scripts use environment variables to locate these resources, including:

- `NEUROCOGMAP_RELEASE_ROOT`
- `NEUROCOGMAP_OUTPUT_DIR`
- `NEUROCOGMAP_GEMMA2_SAE_DIR`
- `NEUROCOGMAP_LLAMA3_8B_SAE_DIR`
- `NEUROCOGMAP_GEMMA2_9B_SAE_DIR`
- `NEUROCOGMAP_QWEN_EMBEDDING_DIR`
- `NEUROCOGMAP_WORD2VEC_PATH`
- `NEUROCOGMAP_LEBEL_DATASET_DIR`

## Companion Data

The companion data package is available on Zenodo at https://zenodo.org/records/20629857.

The companion Zenodo data package contains:

- NeuroCogMap atlas artifacts
- parcel assignments and annotations
- parcel-capability mappings
- structural connectome matrices
- pathology-analysis summaries and figure source data
- human-evaluation summaries and rubrics
- Figure 5 and Figure 6 processed analysis inputs

Raw public benchmark datasets and sensitive harmful-instruction or bias-prompt generations are not redistributed in the companion data package.
