#!/usr/bin/env python3
"""Layered smoke tests for the NeuroCogMap release."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/tmp/neurocogmap_release_smoke")
VENDORED_SAE_LENS = ROOT / "third_party/sae_lens_neurocogmap"
SECRET_RE = re.compile(r"\bhf_[A-Za-z0-9]{20,}\b")

TEXT_EXTENSIONS = {
    ".py",
    ".sh",
    ".md",
    ".txt",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
}

PUBLIC_DOC_PATHS = [
    "README.md",
    "MANIFEST.md",
    "docs/EXTERNAL_RESOURCES.md",
    "docs/SMOKE_TESTS.md",
    "requirements/README.md",
    "requirements/requirements_sae.txt",
    "requirements/requirements_lit.txt",
]

PUBLIC_ENTRY_PATHS = [
    "src/neurocogmap_construction/functional_atlas/get_sae_act.py",
    "src/neurocogmap_construction/functional_atlas/gwMRF_latent_clustering_optimized.py",
    "src/neurocogmap_construction/structural_connectome/build_parcel_connections.py",
    "src/neurocogmap_construction/capability_mapping/aggregate_final_capability_parcel.py",
    "src/brain_alignment/fig5_language_parcels/selection/select_language_overlapping_parcels_accuracy.py",
    "src/model_discovery/fig6_two_step/neural/fit.py",
    "src/model_discovery/fig6_two_step/openloop/compare_aic_kool2017cost_exp2.py",
]


def status_rank(status: str) -> int:
    return {
        "PASS": 0,
        "WARN": 1,
        "SKIPPED_EXTERNAL_RESOURCE": 2,
        "FAIL": 3,
    }[status]


class SmokeRunner:
    def __init__(self, output_dir: Path, verbose: bool = False) -> None:
        self.output_dir = output_dir.resolve()
        self.verbose = verbose
        self.results: list[dict[str, Any]] = []

    def add(
        self,
        name: str,
        status: str,
        message: str,
        details: dict[str, Any] | None = None,
    ) -> None:
        result = {
            "name": name,
            "status": status,
            "message": message,
            "details": details or {},
        }
        self.results.append(result)
        if self.verbose or status != "PASS":
            print(f"[{status}] {name}: {message}")

    def command_env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        existing_pythonpath = env.get("PYTHONPATH", "")
        paths = [
            str(ROOT / "src"),
            str(ROOT / "src/brain_alignment/fig5_language_parcels/litcoder_core"),
            str(VENDORED_SAE_LENS),
        ]
        if existing_pythonpath:
            paths.append(existing_pythonpath)
        env["PYTHONPATH"] = os.pathsep.join(paths)
        return env

    def run_command(
        self,
        name: str,
        cmd: list[str],
        timeout: int = 60,
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        start = time.time()
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(cwd or ROOT),
                env=env or self.command_env(),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=timeout,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            self.add(
                name,
                "FAIL",
                f"command timed out after {timeout}s",
                {"cmd": cmd, "output": (exc.stdout or "")[-3000:]},
            )
            return

        elapsed = round(time.time() - start, 3)
        output = proc.stdout or ""
        if proc.returncode == 0:
            self.add(name, "PASS", f"command succeeded in {elapsed}s")
        else:
            self.add(
                name,
                "FAIL",
                f"command failed with exit code {proc.returncode}",
                {"cmd": cmd, "elapsed_s": elapsed, "output": output[-5000:]},
            )

    def write_report(self) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        report_path = self.output_dir / "smoke_report.json"
        summary = {
            "root": str(ROOT),
            "output_dir": str(self.output_dir),
            "results": self.results,
            "counts": {
                status: sum(r["status"] == status for r in self.results)
                for status in ["PASS", "WARN", "SKIPPED_EXTERNAL_RESOURCE", "FAIL"]
            },
        }
        report_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        return report_path

    def print_table(self) -> None:
        print("\nSmoke test summary")
        print("==================")
        for result in sorted(self.results, key=lambda r: (status_rank(r["status"]), r["name"])):
            print(f"{result['status']:27} {result['name']} - {result['message']}")

    def has_failures(self) -> bool:
        return any(result["status"] == "FAIL" for result in self.results)


def ensure_output_dir_is_external(output_dir: Path) -> None:
    resolved = output_dir.resolve()
    try:
        inside_root = resolved.is_relative_to(ROOT)
    except AttributeError:
        inside_root = str(resolved).startswith(str(ROOT))
    if inside_root:
        raise SystemExit(f"Refusing to write smoke outputs inside the release tree: {resolved}")


def iter_text_files() -> list[Path]:
    skip_parts = {"__pycache__", ".git", ".pytest_cache"}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if any(part in skip_parts for part in rel.parts):
            continue
        if path.suffix in TEXT_EXTENSIONS:
            files.append(path)
    return files


def check_secret_scan(runner: SmokeRunner) -> None:
    hits: list[str] = []
    for path in iter_text_files():
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for match in SECRET_RE.finditer(text):
            hits.append(f"{path.relative_to(ROOT)}:{text.count(chr(10), 0, match.start()) + 1}")
            break
    if hits:
        runner.add("secret_scan", "FAIL", "hardcoded Hugging Face token-like strings found", {"hits": hits[:50]})
    else:
        runner.add("secret_scan", "PASS", "no hardcoded Hugging Face token-like strings found")


def check_path_scan(runner: SmokeRunner) -> None:
    public_hits: list[str] = []
    for rel in PUBLIC_DOC_PATHS:
        path = ROOT / rel
        if path.exists() and "LEGACY_ABSOLUTE_PATH" in path.read_text(encoding="utf-8", errors="ignore"):
            public_hits.append(rel)

    code_hits: list[str] = []
    for path in (ROOT / "src").rglob("*"):
        if path.suffix not in {".py", ".sh", ".md"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "LEGACY_ABSOLUTE_PATH" in text:
            line_no = next(
                i for i, line in enumerate(text.splitlines(), start=1) if "LEGACY_ABSOLUTE_PATH" in line
            )
            code_hits.append(f"{path.relative_to(ROOT)}:{line_no}")

    if public_hits:
        runner.add(
            "public_path_scan",
            "FAIL",
            "public docs or requirements contain local absolute paths",
            {"hits": public_hits},
        )
    else:
        runner.add("public_path_scan", "PASS", "public docs and requirements avoid local absolute paths")

    if code_hits:
        runner.add(
            "source_path_scan",
            "WARN",
            f"{len(code_hits)} source files still contain local absolute paths; ensure public launchers override them",
            {"examples": code_hits[:30]},
        )
    else:
        runner.add("source_path_scan", "PASS", "source files do not contain local absolute paths")


def check_public_entry_path_scan(runner: SmokeRunner) -> None:
    hits: list[str] = []
    for rel in PUBLIC_ENTRY_PATHS:
        path = ROOT / rel
        if not path.exists():
            hits.append(f"{rel}:missing")
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for line_no, line in enumerate(text.splitlines(), start=1):
            if "LEGACY_ABSOLUTE_PATH" in line:
                hits.append(f"{rel}:{line_no}")
                break
    if hits:
        runner.add(
            "public_entry_path_scan",
            "FAIL",
            "public entrypoints contain local absolute paths",
            {"hits": hits},
        )
    else:
        runner.add("public_entry_path_scan", "PASS", "public entrypoints avoid local absolute paths")


def check_release_script(runner: SmokeRunner, python: str) -> None:
    runner.run_command(
        "release_integrity_check",
        [python, "docs/check_release.py"],
        timeout=120,
    )


def check_imports(runner: SmokeRunner, sae_python: str, lit_python: str) -> None:
    sae_code = (
        "import numpy,pandas,sklearn,scipy,torch,transformers,datasets,sae_lens;"
        "print('sae imports ok', torch.__version__, sae_lens.__version__)"
    )
    lit_code = (
        "import numpy,pandas,sklearn,scipy,torch,transformers,nibabel,nilearn,schedulefree;"
        "print('lit imports ok', torch.__version__)"
    )
    runner.run_command("sae_dependency_imports", [sae_python, "-c", sae_code], timeout=120)
    runner.run_command("lit_dependency_imports", [lit_python, "-c", lit_code], timeout=120)


def check_vendored_sae_lens_install(runner: SmokeRunner, python: str) -> None:
    package_copy = runner.output_dir / "package_dryrun" / "sae_lens_neurocogmap"
    if package_copy.exists():
        shutil.rmtree(package_copy)
    shutil.copytree(VENDORED_SAE_LENS, package_copy, ignore=shutil.ignore_patterns("*.egg-info", "__pycache__"))
    runner.run_command(
        "vendored_sae_lens_editable_dry_run",
        [
            python,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--no-deps",
            "--no-build-isolation",
            "-e",
            str(package_copy),
            "--report",
            str(runner.output_dir / "sae_lens_pip_dryrun_report.json"),
        ],
        timeout=180,
    )


def check_cli_help(runner: SmokeRunner, sae_python: str, lit_python: str) -> None:
    commands = [
        (
            "fig1_get_sae_act_help",
            [sae_python, "src/neurocogmap_construction/functional_atlas/get_sae_act.py", "--help"],
        ),
        (
            "fig1_gwmrf_help",
            [sae_python, "src/neurocogmap_construction/functional_atlas/gwMRF_latent_clustering_optimized.py", "--help"],
        ),
        (
            "fig1_structural_connectome_help",
            [sae_python, "src/neurocogmap_construction/structural_connectome/build_parcel_connections.py", "--help"],
        ),
        (
            "fig1_capability_mapping_help",
            [sae_python, "src/neurocogmap_construction/capability_mapping/aggregate_final_capability_parcel.py", "--help"],
        ),
        (
            "fig5_language_selection_help",
            [lit_python, "src/brain_alignment/fig5_language_parcels/selection/select_language_overlapping_parcels_accuracy.py", "--help"],
        ),
        (
            "fig6_neural_fit_help",
            [lit_python, "src/model_discovery/fig6_two_step/neural/fit.py", "--help"],
        ),
        (
            "fig6_aic_compare_help",
            [lit_python, "src/model_discovery/fig6_two_step/openloop/compare_aic_kool2017cost_exp2.py", "--help"],
        ),
    ]
    for name, cmd in commands:
        runner.run_command(name, cmd, timeout=120)


def check_get_sae_act_list_datasets(runner: SmokeRunner, sae_python: str) -> None:
    qa_dir = ROOT / "data/neurocogmap_construction/capability_qa"
    if not qa_dir.exists():
        runner.add(
            "fig1_get_sae_act_list_datasets",
            "SKIPPED_EXTERNAL_RESOURCE",
            "construction QA prompts are not redistributed in the public code/data packages",
        )
        return
    runner.run_command(
        "fig1_get_sae_act_list_datasets",
        [
            sae_python,
            "src/neurocogmap_construction/functional_atlas/get_sae_act.py",
            "--list_datasets",
            "--data_dir",
            str(qa_dir),
        ],
        timeout=120,
    )


def assert_nonempty_csv(path: Path) -> tuple[list[str], int]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = sum(1 for _ in reader)
    if not header or rows == 0:
        try:
            label = str(path.relative_to(ROOT))
        except ValueError:
            label = str(path)
        raise ValueError(f"{label} is empty")
    return header, rows


def resolve_data_root(args: argparse.Namespace) -> Path | None:
    configured = args.data_root or os.environ.get("NEUROCOGMAP_DATA_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    if (ROOT / "data").exists() or (ROOT / "artifacts").exists():
        return ROOT
    return None


def check_data_schema(runner: SmokeRunner, data_root: Path | None) -> None:
    if data_root is None:
        runner.add(
            "data_schema",
            "SKIPPED_EXTERNAL_RESOURCE",
            "Zenodo data package not mounted; pass --data-root to validate data schemas",
        )
        return
    try:
        metadata_files = [
            data_root / "data/neurocogmap_construction/metadata/final_merged_capability_dataset_stats.json",
            data_root / "data/neurocogmap_construction/metadata/dataset_introduction.json",
        ]
        for path in metadata_files:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if not payload:
                raise ValueError(f"{path} is empty")

        neural_csv = data_root / "data/model_discovery/two_step/neural/schaefer_parcels_100.csv"
        header, rows = assert_nonempty_csv(neural_csv)
        required = {"participant", "run_no", "sub_trial_no", "sub_trial_type"}
        if not required.issubset(set(header)):
            raise ValueError("schaefer_parcels_100.csv missing required metadata columns")
        if len(header) < 100 or rows == 0:
            raise ValueError("schaefer_parcels_100.csv appears too small")

        prompts = data_root / "data/model_discovery/two_step/neural/prompts_reformatted.jsonl"
        with prompts.open("r", encoding="utf-8") as f:
            first_prompt = json.loads(f.readline())
        if "text" not in first_prompt:
            raise ValueError("prompts_reformatted.jsonl first row lacks text")

        for rel in [
            "data/model_discovery/two_step/openloop/kool2016when/exp2.csv",
            "data/model_discovery/two_step/openloop/kool2017cost/exp2.csv",
        ]:
            header, rows = assert_nonempty_csv(data_root / rel)
            if not {"participant", "task", "trial"}.issubset(set(header)):
                raise ValueError(f"{rel} missing openloop columns")
            if rows < 10:
                raise ValueError(f"{rel} has too few rows")

        runner.add("data_schema", "PASS", "Zenodo metadata and Fig6 input data schemas are readable")
    except Exception as exc:
        runner.add("data_schema", "FAIL", str(exc))


def check_fig5_artifacts(runner: SmokeRunner, data_root: Path | None) -> None:
    if data_root is None:
        runner.add(
            "fig5_artifacts",
            "SKIPPED_EXTERNAL_RESOURCE",
            "Zenodo figure-selection artifacts not mounted; pass --data-root to validate them",
        )
        return
    try:
        selection_path = data_root / "artifacts/figure_selections/fig5_language_parcels/language_parcel_overlap_and_accuracy.json"
        data = json.loads(selection_path.read_text(encoding="utf-8"))
        selected = data.get("selected_parcels") if isinstance(data, dict) else data
        if not isinstance(selected, list) or len(selected) != 10:
            raise ValueError("Fig5 language parcel selection should contain exactly 10 parcels")
        allowed_keys = {"parcel_idx", "parcel_name", "parcel_vertex_count", "overlap_vertex_count"}
        for item in selected:
            if set(item) != allowed_keys:
                raise ValueError("Fig5 language parcel selection contains non-release fields")
        panel_summary = data_root / "artifacts/figure_selections/fig5_language_parcels/panel_a_data/summary_results.csv"
        with panel_summary.open("r", encoding="utf-8-sig", newline="") as f:
            rows = list(csv.DictReader(f))
        if len(rows) != 1 or rows[0].get("method") != "NeuroCogMap":
            raise ValueError("Fig5 panel_a_data summary should contain only NeuroCogMap")
        profile_path = data_root / "artifacts/figure_selections/fig5_language_parcels/human_cortical_profiles/human_cortical_parcel_descriptions.json"
        profiles = json.loads(profile_path.read_text(encoding="utf-8"))
        if not isinstance(profiles, (dict, list)) or len(profiles) < 100:
            raise ValueError("human cortical parcel descriptions should contain at least 100 parcels")
        runner.add("fig5_artifacts", "PASS", "Fig5 NeuroCogMap selection recovers 10 parcels")
    except Exception as exc:
        runner.add("fig5_artifacts", "FAIL", str(exc))


def check_fig6_model_roles(runner: SmokeRunner, lit_python: str) -> None:
    code = f"""
import sys
from pathlib import Path
root = Path({str(ROOT)!r})
openloop = root / 'src/model_discovery/fig6_two_step/openloop'
sys.path.insert(0, str(openloop))
from models import DualSystems
from improved_cog_model_cogneuromap_simple import DualSystemsHierarchicalBanditCogneuromapSimple
from improved_cog_model_cogneuromap_full import DualSystemsHierarchicalBanditCogneuromapFull
for cls in [DualSystems, DualSystemsHierarchicalBanditCogneuromapSimple, DualSystemsHierarchicalBanditCogneuromapFull]:
    model = cls(variant='two_step')
    assert model is not None
print('fig6 model roles import ok')
"""
    runner.run_command("fig6_model_roles", [lit_python, "-c", code], timeout=120)


def check_gpu_cuda(runner: SmokeRunner, python: str, name: str, cuda_device: str) -> None:
    env = runner.command_env()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    code = "import torch; assert torch.cuda.is_available(); print(torch.cuda.device_count(), torch.cuda.get_device_name(0))"
    runner.run_command(name, [python, "-c", code], timeout=120, env=env)


def external_path_arg(args: argparse.Namespace, attr: str, env_name: str) -> Path | None:
    value = getattr(args, attr) or os.environ.get(env_name)
    if not value:
        return None
    return Path(value).expanduser()


def check_external_resource_paths(runner: SmokeRunner, args: argparse.Namespace) -> dict[str, Path]:
    resources = {
        "gemma2_sae_dir": external_path_arg(args, "gemma2_sae_dir", "NEUROCOGMAP_GEMMA2_SAE_DIR"),
        "qwen_embedding_dir": external_path_arg(args, "qwen_embedding_dir", "NEUROCOGMAP_QWEN_EMBEDDING_DIR"),
        "word2vec_path": external_path_arg(args, "word2vec_path", "NEUROCOGMAP_WORD2VEC_PATH"),
        "lebel_dataset_dir": external_path_arg(args, "lebel_dataset_dir", "NEUROCOGMAP_LEBEL_DATASET_DIR"),
    }
    available: dict[str, Path] = {}
    missing: list[str] = []
    for name, path in resources.items():
        if path is None:
            missing.append(f"{name}=not_configured")
        elif path.exists():
            available[name] = path
        else:
            missing.append(f"{name}={path}")
    if missing:
        runner.add(
            "external_resource_paths",
            "SKIPPED_EXTERNAL_RESOURCE",
            "some optional external resources are missing or not configured",
            {"missing": missing},
        )
    if available:
        runner.add("external_resource_paths_available", "PASS", f"{len(available)} external resources found")
    return available


def check_sae_weight_load(runner: SmokeRunner, args: argparse.Namespace, resources: dict[str, Path]) -> None:
    sae_dir = resources.get("gemma2_sae_dir")
    if not sae_dir:
        runner.add("gpu_sae_weight_load", "SKIPPED_EXTERNAL_RESOURCE", "Gemma-2B SAE directory not configured")
        return
    sae_id = "layer_0/width_16k/average_l0_105"
    params_path = sae_dir / sae_id / "params.npz"
    if not params_path.exists():
        runner.add("gpu_sae_weight_load", "SKIPPED_EXTERNAL_RESOURCE", f"SAE params not found: {params_path}")
        return
    env = runner.command_env()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    code = f"""
from sae_lens import SAE
sae, cfg, sparsity = SAE.from_pretrained(
    release='gemma-scope-2b-pt-res',
    sae_id={sae_id!r},
    device='cuda',
    local_path={str(params_path)!r},
)
print('loaded', sae.cfg.d_sae)
"""
    runner.run_command("gpu_sae_weight_load", [args.sae_python, "-c", code], timeout=180, env=env)


def check_vllm_endpoint(runner: SmokeRunner, vllm_url: str | None, api_key: str | None = None) -> None:
    if not vllm_url:
        runner.add("vllm_endpoint", "SKIPPED_EXTERNAL_RESOURCE", "vLLM URL not configured")
        return
    base = vllm_url.rstrip("/")
    models_url = base + "/models"
    try:
        request = urllib.request.Request(models_url)
        if api_key:
            request.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(request, timeout=10) as response:
            payload = response.read(2000).decode("utf-8", errors="ignore")
        runner.add("vllm_endpoint", "PASS", f"endpoint responded at {models_url}", {"sample": payload[:500]})
    except (urllib.error.URLError, TimeoutError) as exc:
        runner.add("vllm_endpoint", "SKIPPED_EXTERNAL_RESOURCE", f"endpoint unavailable: {exc}")


def check_fig6_openloop_minifit(runner: SmokeRunner, args: argparse.Namespace, data_root: Path | None) -> None:
    if data_root is None:
        runner.add(
            "gpu_fig6_openloop_minifit",
            "SKIPPED_EXTERNAL_RESOURCE",
            "Zenodo Fig6 open-loop data not mounted; pass --data-root to run the mini-fit",
        )
        return
    env = runner.command_env()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_device
    code = f"""
import sys
from pathlib import Path
import pandas as pd
root = Path({str(ROOT)!r})
data_root = Path({str(data_root)!r})
openloop = root / 'src/model_discovery/fig6_two_step/openloop'
sys.path.insert(0, str(openloop))
from models import DualSystems
from improved_cog_model_cogneuromap_simple import DualSystemsHierarchicalBanditCogneuromapSimple
from improved_cog_model_cogneuromap_full import DualSystemsHierarchicalBanditCogneuromapFull
from trainers import Trainer
df = pd.read_csv(data_root / 'data/model_discovery/two_step/openloop/kool2016when/exp2.csv')
participant = df['participant'].dropna().iloc[0]
sample = df[df['participant'] == participant].head(40).copy()
for cls in [DualSystems, DualSystemsHierarchicalBanditCogneuromapSimple, DualSystemsHierarchicalBanditCogneuromapFull]:
    model = cls(variant='two_step')
    trainer = Trainer(model, num_iter=1)
    loss = trainer.fit_and_evaluate(sample, sample)
    print(cls.__name__, float(loss.detach().cpu()))
"""
    runner.run_command("gpu_fig6_openloop_minifit", [args.lit_python, "-c", code], timeout=240, env=env)


def check_fig5_external_probe(runner: SmokeRunner, resources: dict[str, Path]) -> None:
    lebel_dir = resources.get("lebel_dataset_dir")
    if not lebel_dir:
        runner.add("fig5_lebel_resource_probe", "SKIPPED_EXTERNAL_RESOURCE", "LeBel/LIT dataset directory not configured")
        return
    expected = ["assembly_lebel_uts02.pkl", "assembly_lebel_uts03.pkl"]
    missing = [name for name in expected if not (lebel_dir / name).exists()]
    if missing:
        runner.add("fig5_lebel_resource_probe", "SKIPPED_EXTERNAL_RESOURCE", "missing LeBel assembly files", {"missing": missing})
    else:
        runner.add("fig5_lebel_resource_probe", "PASS", "LeBel assembly files are present")


def run_public_checks(runner: SmokeRunner, args: argparse.Namespace) -> None:
    data_root = resolve_data_root(args)
    check_release_script(runner, args.sae_python)
    check_imports(runner, args.sae_python, args.lit_python)
    check_vendored_sae_lens_install(runner, args.sae_python)
    check_secret_scan(runner)
    check_path_scan(runner)
    check_public_entry_path_scan(runner)
    check_cli_help(runner, args.sae_python, args.lit_python)
    check_get_sae_act_list_datasets(runner, args.sae_python)
    check_data_schema(runner, data_root)
    check_fig5_artifacts(runner, data_root)
    check_fig6_model_roles(runner, args.lit_python)


def run_gpu_checks(runner: SmokeRunner, args: argparse.Namespace) -> None:
    run_public_checks(runner, args)
    check_gpu_cuda(runner, args.sae_python, "gpu_sae_cuda_visible", args.cuda_device)
    check_gpu_cuda(runner, args.lit_python, "gpu_lit_cuda_visible", args.cuda_device)
    resources = check_external_resource_paths(runner, args)
    check_sae_weight_load(runner, args, resources)
    check_fig5_external_probe(runner, resources)
    check_fig6_openloop_minifit(runner, args, resolve_data_root(args))
    check_vllm_endpoint(runner, args.vllm_url, args.vllm_api_key or os.environ.get("OPENAI_API_KEY"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run layered NeuroCogMap release smoke tests.")
    parser.add_argument("--level", choices=["public", "gpu"], default="public", help="Smoke-test depth.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for smoke reports.")
    parser.add_argument("--data-root", type=Path, default=None, help="Optional NeuroCogMap_Data root from the Zenodo package.")
    parser.add_argument("--sae-python", default=sys.executable, help="Python executable for Fig1/Fig3/Fig4 checks.")
    parser.add_argument("--lit-python", default=sys.executable, help="Python executable for Fig5/Fig6 checks.")
    parser.add_argument("--cuda-device", default="4", help="CUDA device index for GPU smoke checks.")
    parser.add_argument("--gemma2-sae-dir", default=None, help="Optional local Gemma-2B SAE directory.")
    parser.add_argument("--qwen-embedding-dir", default=None, help="Optional local Qwen embedding model directory.")
    parser.add_argument("--word2vec-path", default=None, help="Optional local Word2Vec vector file.")
    parser.add_argument("--lebel-dataset-dir", default=None, help="Optional local LeBel/LIT dataset directory.")
    parser.add_argument("--vllm-url", default=None, help="Optional OpenAI-compatible vLLM base URL, e.g. http://127.0.0.1:8001/v1.")
    parser.add_argument("--vllm-api-key", default=None, help="Optional API key for the OpenAI-compatible vLLM endpoint.")
    parser.add_argument("--verbose", action="store_true", help="Print passing checks as they run.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_output_dir_is_external(args.output_dir)
    runner = SmokeRunner(args.output_dir, verbose=args.verbose)

    if args.level == "public":
        run_public_checks(runner, args)
    else:
        run_gpu_checks(runner, args)

    report_path = runner.write_report()
    runner.print_table()
    print(f"\nReport written to: {report_path}")
    return 1 if runner.has_failures() else 0


if __name__ == "__main__":
    raise SystemExit(main())
