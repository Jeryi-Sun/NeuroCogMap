# NeuroCogMap-Patched SAELens

This package vendors the NeuroCogMap-local patched copy of `sae_lens` used by the release experiments.

The installable distribution name is `sae-lens-neurocogmap`, but the Python import remains:

```python
import sae_lens
```

Install from the release root with:

```bash
pip install -e third_party/sae_lens_neurocogmap
```

This package contains source code only. It does not include SAE weights, model weights, Hugging Face caches, or generated activation outputs.

The vendored source was copied from the local `sae` conda environment's `sae_lens 5.11.0` installation after NeuroCogMap-specific data-loading edits.
