# Runtime Requirements

The release uses two lightweight requirement sets rather than a full server `pip freeze`.

## Main NeuroCogMap / Pathology Environment

Use `requirements_sae.txt` for:

- Figure 1 NeuroCogMap construction;
- Figures 3 and 4 pathology analysis, detection, and intervention;
- general release checks.

From the release root:

```bash
python -m venv .venv-sae
source .venv-sae/bin/activate
pip install -r requirements/requirements_sae.txt
pip install -e third_party/sae_lens_neurocogmap
```

The final command installs the NeuroCogMap-patched local SAELens package. It provides the normal `sae_lens` Python import.

## Figure 5 / Figure 6 Environment

Use `requirements_lit.txt` for:

- Figure 5 brain alignment and LITcoder-style encoding code;
- Figure 6 two-step neural prediction and model-discovery code.

From the release root:

```bash
python -m venv .venv-lit
source .venv-lit/bin/activate
pip install -r requirements/requirements_lit.txt
```

## External Resources

These requirements do not include model weights, SAE weights, vLLM servers, Hugging Face caches, or large fMRI assemblies. See `../docs/EXTERNAL_RESOURCES.md`.
