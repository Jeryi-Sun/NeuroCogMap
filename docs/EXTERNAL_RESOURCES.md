# External Resources

The release does not include model weights, SAE weights, local inference servers, large fMRI assemblies, or generated activation/result caches.

Python package requirements are bundled separately under `requirements/`. Figure 1 and Figures 3/4 use `requirements_sae.txt` plus the local patched SAELens package in `third_party/sae_lens_neurocogmap/`; Figure 5 and Figure 6 use `requirements_lit.txt`.

## Models and SAE Weights

The scripts refer to these model families:

- Gemma-2-2B
- Gemma-2-9B-IT
- Llama-3.1-8B
- BERT base uncased
- Word2Vec embeddings
- Qwen3-Embedding-8B
- GPT-OSS-20B or another OpenAI-compatible local judge endpoint for annotation/evaluation steps

The NeuroCogMap construction and intervention code expects Gemma Scope or Llama Scope residual-stream SAE weights to be available locally or through the configured SAE-loading mechanism.

The release includes patched SAELens source code, but it does not include any downloaded SAE parameters.

## Environment Variables

Portable public entrypoints understand these variables:

- `NEUROCOGMAP_RELEASE_ROOT`: release root override; normally auto-detected.
- `NEUROCOGMAP_OUTPUT_DIR`: generated-output root; defaults to `/tmp/neurocogmap_release_outputs`.
- `NEUROCOGMAP_GEMMA2_SAE_DIR`: Gemma-2B SAE root for construction/intervention smoke tests.
- `NEUROCOGMAP_LLAMA3_8B_SAE_DIR`: optional Llama-3.1-8B SAE root for structural-connectome reruns.
- `NEUROCOGMAP_GEMMA2_9B_SAE_DIR`: optional Gemma-2-9B-IT SAE root for structural-connectome reruns.
- `NEUROCOGMAP_QWEN_EMBEDDING_DIR`: local Qwen embedding model directory.
- `NEUROCOGMAP_WORD2VEC_PATH`: local Word2Vec vector file.
- `NEUROCOGMAP_LEBEL_DATASET_DIR`: LeBel/LIT dataset directory containing annotations, ROI masks, and assemblies.
- `NEUROCOGMAP_CAP_PARCEL_DIR`: intermediate capability-parcel ranking directory for capability mapping.
- `NEUROCOGMAP_SIMILARITY_DETAILED_CSV`: detailed capability-parcel semantic similarity CSV.
- `NEUROCOGMAP_SIMILARITY_MATRIX_CSV`: matrix-form capability-parcel semantic similarity CSV.

## Brain Data

Figure 5 LeBel story-listening fMRI assemblies are not bundled. The release includes code and small selection artifacts required to recover the final ten language-related cortical parcels, but full model refitting requires the LeBel/LITcoder-compatible assemblies and feature caches.

Figure 6 two-step neural prediction includes the Schaefer-100 ROI input CSV used by the released scripts. If replacing it with a newly downloaded or reprocessed version, keep the expected column names used by `src/model_discovery/fig6_two_step/neural/fit.py`.

## Generated Files

Generated outputs should be written outside tracked source directories, for example under `/tmp/neurocogmap_release_outputs` or a local `outputs/` directory. Do not commit:

- token-level parcel activations
- generated/evaluated model responses
- cross-validation outputs
- intervention run outputs
- model/SAE cache directories
- `wandb` directories
