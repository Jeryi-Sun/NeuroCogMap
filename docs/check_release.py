#!/usr/bin/env python3
"""Lightweight integrity checks for the NeuroCogMap GitHub code release."""

from __future__ import annotations

import py_compile
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

REQUIRED_PATHS = [
    "README.md",
    "MANIFEST.md",
    "docs/EXTERNAL_RESOURCES.md",
    "docs/SMOKE_TESTS.md",
    "docs/run_smoke_tests.py",
    "docs/check_release.py",
    "requirements/README.md",
    "requirements/requirements_sae.txt",
    "requirements/requirements_lit.txt",
    "src/neurocogmap_release/__init__.py",
    "src/neurocogmap_release/paths.py",
    "src/neurocogmap_construction/README.md",
    "src/pathology/README.md",
    "src/brain_alignment/fig5_language_parcels/README.md",
    "src/model_discovery/fig6_two_step/README.md",
    "src/neurocogmap_construction/functional_atlas/get_sae_act.py",
    "src/neurocogmap_construction/functional_atlas/gwMRF_latent_clustering_optimized.py",
    "src/neurocogmap_construction/structural_connectome/build_parcel_connections.py",
    "src/neurocogmap_construction/capability_mapping/aggregate_final_capability_parcel.py",
    "src/brain_alignment/fig5_language_parcels/selection/select_language_overlapping_parcels_accuracy.py",
    "src/model_discovery/fig6_two_step/neural/fit.py",
    "src/model_discovery/fig6_two_step/openloop/compare_aic_kool2017cost_exp2.py",
    "third_party/sae_lens_neurocogmap/README.md",
    "third_party/sae_lens_neurocogmap/LICENSE",
    "third_party/sae_lens_neurocogmap/pyproject.toml",
    "third_party/sae_lens_neurocogmap/sae_lens/__init__.py",
]

FORBIDDEN_TOP_LEVEL_DIRS = {
    "artifacts",
    "data",
}

FORBIDDEN_NAMES = {
    "__pycache__",
    ".pytest_cache",
    "wandb",
}

FORBIDDEN_PATH_SUBSTRINGS = {
    ".egg-info",
    "baseline_results",
    "parcel_activation_results",
    "qa_sae_output",
    "sycophancy__answer_gemma",
}

FORBIDDEN_LOCAL_PATH_STRINGS = [
    "/new_" + "disk3",
    "/Users/" + "sunzhongxiang",
    "gaoling" + "_138",
    "gaoling" + "_138_vpn",
]

WEIGHT_LIKE_SUFFIXES = {
    ".bin",
    ".ckpt",
    ".h5",
    ".npy",
    ".npz",
    ".onnx",
    ".pkl",
    ".pt",
    ".pth",
    ".safetensors",
}

TEXT_SUFFIXES = {
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".sh",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


def fail(message: str) -> None:
    print(f"FAIL: {message}", file=sys.stderr)
    raise SystemExit(1)


def iter_files() -> list[Path]:
    return [path for path in ROOT.rglob("*") if path.is_file() and ".git" not in path.relative_to(ROOT).parts]


def check_required_paths() -> None:
    missing = [rel for rel in REQUIRED_PATHS if not (ROOT / rel).exists()]
    if missing:
        fail("missing required code-release paths:\n" + "\n".join(missing))


def check_package_boundary() -> None:
    bad_top = sorted(rel for rel in FORBIDDEN_TOP_LEVEL_DIRS if (ROOT / rel).exists())
    if bad_top:
        fail("GitHub code release should not include top-level data/artifact dirs:\n" + "\n".join(bad_top))

    bad_paths: list[str] = []
    for path in ROOT.rglob("*"):
        rel = path.relative_to(ROOT)
        if ".git" in rel.parts:
            continue
        rel_text = str(rel)
        if any(part in FORBIDDEN_NAMES for part in rel.parts):
            bad_paths.append(rel_text)
        if any(fragment in rel_text for fragment in FORBIDDEN_PATH_SUBSTRINGS):
            bad_paths.append(rel_text)
        if path.is_file() and path.suffix.lower() in WEIGHT_LIKE_SUFFIXES:
            bad_paths.append(rel_text)
    if bad_paths:
        fail("forbidden generated, cache, model, or bulk-data paths found:\n" + "\n".join(bad_paths[:50]))


def check_local_path_hygiene() -> None:
    hits: list[str] = []
    for path in iter_files():
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for needle in FORBIDDEN_LOCAL_PATH_STRINGS:
            if needle in text:
                line_no = next(
                    i for i, line in enumerate(text.splitlines(), start=1) if needle in line
                )
                hits.append(f"{path.relative_to(ROOT)}:{line_no}:{needle}")
                break
    if hits:
        fail("local machine or SSH paths remain in public files:\n" + "\n".join(hits[:50]))


def check_python_syntax() -> None:
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_root = Path(tmpdir)
        for index, py_file in enumerate(ROOT.rglob("*.py")):
            try:
                py_compile.compile(
                    str(py_file),
                    cfile=str(tmp_root / f"{index}.pyc"),
                    doraise=True,
                )
            except py_compile.PyCompileError as exc:
                failures.append(f"{py_file.relative_to(ROOT)}: {exc.msg}")
    if failures:
        fail("python syntax failures:\n" + "\n".join(failures[:50]))


def check_shell_syntax() -> None:
    shell_files = list(ROOT.rglob("*.sh"))
    failures: list[str] = []
    for path in shell_files:
        text = path.read_text(encoding="utf-8", errors="ignore")
        import subprocess

        proc = subprocess.run(
            ["bash", "-n", str(path)],
            cwd=str(ROOT),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if proc.returncode != 0:
            failures.append(f"{path.relative_to(ROOT)}:\n{proc.stdout[-1000:]}")
        if not text.strip():
            failures.append(f"{path.relative_to(ROOT)}: empty shell script")
    if failures:
        fail("shell syntax failures:\n" + "\n".join(failures[:30]))


def check_runtime_packaging() -> None:
    for rel in ["requirements/requirements_sae.txt", "requirements/requirements_lit.txt"]:
        text = (ROOT / rel).read_text(encoding="utf-8")
        requirement_lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        ]
        requirement_text = "\n".join(requirement_lines)
        bad_fragments = ["/new_disk", "/home/", "file://", "-e "]
        found = [fragment for fragment in bad_fragments if fragment in requirement_text]
        if found:
            fail(f"{rel} contains local or editable dependency fragments: {found}")

    sae_requirements = (ROOT / "requirements/requirements_sae.txt").read_text(encoding="utf-8")
    if "sae-lens" in sae_requirements.lower():
        fail("requirements_sae.txt should not install upstream SAELens; use third_party/sae_lens_neurocogmap")

    pyproject = (ROOT / "third_party/sae_lens_neurocogmap/pyproject.toml").read_text(encoding="utf-8")
    if 'name = "sae-lens-neurocogmap"' not in pyproject:
        fail("vendored SAELens pyproject.toml has unexpected package name")

    init_file = ROOT / "third_party/sae_lens_neurocogmap/sae_lens/__init__.py"
    init_text = init_file.read_text(encoding="utf-8")
    if "__version__" not in init_text:
        fail("vendored SAELens package is missing __version__ in sae_lens/__init__.py")

    license_text = (ROOT / "third_party/sae_lens_neurocogmap/LICENSE").read_text(encoding="utf-8").strip()
    if len(license_text) < 100:
        fail("vendored SAELens LICENSE appears to be missing or truncated")


def main() -> None:
    check_required_paths()
    check_package_boundary()
    check_local_path_hygiene()
    check_runtime_packaging()
    check_python_syntax()
    check_shell_syntax()
    print("OK: NeuroCogMap GitHub code-release integrity checks passed.")


if __name__ == "__main__":
    main()
