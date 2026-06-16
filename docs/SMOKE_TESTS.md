# Smoke Tests

Use `docs/run_smoke_tests.py` to check that the release is usable without writing generated files into the repository.

## Public Smoke Test

Run after installing the release dependencies:

```bash
python docs/run_smoke_tests.py \
  --level public \
  --output-dir /tmp/neurocogmap_release_smoke
```

This level does not require model weights, SAE weights, vLLM, or large fMRI assemblies. It checks release integrity, dependency imports, CLI help, vendored SAELens packaging, and secret/path hygiene. When a Zenodo data root is provided, it also checks selected processed data and figure-source artifacts.

In the GitHub code package, data-backed checks are reported as `SKIPPED_EXTERNAL_RESOURCE` unless the companion Zenodo data package is mounted:

```bash
python docs/run_smoke_tests.py \
  --level public \
  --data-root /path/to/NeuroCogMap_Data \
  --output-dir /tmp/neurocogmap_release_smoke
```

The public check treats README/smoke-covered entrypoints as portable: they must not contain local absolute paths. Older non-public scripts can still appear in `source_path_scan` as `WARN` while they are kept for provenance.

## Internal GPU Smoke Test

On the project server, use the tested conda environments and pass local external resources explicitly:

```bash
CUDA_VISIBLE_DEVICES=4 python docs/run_smoke_tests.py \
  --level gpu \
  --sae-python /path/to/sae/bin/python \
  --lit-python /path/to/lit/bin/python \
  --cuda-device 4 \
  --gemma2-sae-dir /path/to/gemma-scope-2b-pt-res \
  --qwen-embedding-dir /path/to/Qwen3-Embedding-8B \
  --word2vec-path /path/to/nlwiki_20180420_300d.txt \
  --lebel-dataset-dir /path/to/litcoder_core/dataset \
  --vllm-url http://127.0.0.1:8001/v1 \
  --vllm-api-key "$OPENAI_API_KEY" \
  --output-dir /tmp/neurocogmap_release_smoke
```

Missing external resources are reported as `SKIPPED_EXTERNAL_RESOURCE`, not as release-code failures. Any `FAIL` row should be fixed before claiming that the corresponding reproduction layer is runnable.

All smoke outputs are written to the requested `--output-dir`; the harness refuses to write reports inside the release tree.
