# NeuroCogMap Construction

This module contains the Figure 1 construction pipeline.

Released construction inputs are under `../../data/neurocogmap_construction/`, and released small artifacts are under `../../artifacts/neurocogmap_atlas/` from the repository root. Full reconstruction also requires model weights and SAE weights.

Recommended environment from the release root:

```bash
pip install -r requirements/requirements_sae.txt
pip install -e third_party/sae_lens_neurocogmap
```

## Pipeline

1. Extract sentence-level SAE activations:
   - `functional_atlas/get_sae_act.py`
2. Cluster SAE latents into functional parcels:
   - `functional_atlas/gwMRF_latent_clustering_optimized.py`
3. Generate parcel cognitive descriptions:
   - `functional_atlas/analyze_parcel_functionality.py`
4. Build the directed structural connectome:
   - `structural_connectome/build_parcel_connections.py`
5. Build parcel-to-capability mappings:
   - `capability_mapping/aggregate_final_capability_parcel.py`

Public entrypoints prefer CLI paths, then environment variables, then release-relative defaults. Generated files default to `/tmp/neurocogmap_release_outputs` via `NEUROCOGMAP_OUTPUT_DIR`.

## Released Atlas Artifacts

Preferred model-specific artifacts are organized as:

- `gemma2_2b/`: Gemma-2-2B atlas, 270 parcels.
- `llama3_1_8b/`: Llama-3.1-8B atlas, 240 parcels.
- `gemma2_9b_it/`: Gemma-2-9B-IT atlas, 270 parcels.

Each model directory contains the available parcel assignments, parcel descriptions, capability-to-parcel mapping, and structural connectome. Gemma-2-2B also includes supplemental preprocessing, SVD, edge-list, and detailed parcel-functionality artifacts.

## Released Construction Inputs

- `capability_qa/`: QA-style construction corpora used for SAE activation extraction.
- `capability_test/`: held-out/test QA inputs used for construction audits and downstream checks.
- `metadata/`: capability-to-dataset statistics, dataset descriptions, and coverage metadata.

## Typical Execution Order

For a portable smoke-style run, call the Python entrypoints directly and pass external resources explicitly:

```bash
python functional_atlas/get_sae_act.py --list_datasets
python functional_atlas/gwMRF_latent_clustering_optimized.py --help
python structural_connectome/build_parcel_connections.py --help
python capability_mapping/aggregate_final_capability_parcel.py --help
```

Full reconstruction additionally requires local model/SAE weights and intermediate capability-mapping inputs. Original shell launchers are kept as provenance helpers, but the portable public interface is the parameterized Python CLI above.

Large SAE activations, SAE weights, model weights, and generated intermediate outputs are intentionally not included.
