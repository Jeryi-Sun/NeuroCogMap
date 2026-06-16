"""Path helpers for the NeuroCogMap release.

CLI arguments should still take precedence in entrypoint scripts. These helpers
provide release-relative defaults and environment-variable fallbacks without
writing generated outputs into the release tree.
"""

from __future__ import annotations

import os
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("/tmp/neurocogmap_release_outputs")


def release_root() -> Path:
    configured = os.environ.get("NEUROCOGMAP_RELEASE_ROOT")
    if configured:
        return Path(configured).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


def data_path(*parts: str) -> Path:
    return release_root() / "data" / Path(*parts)


def artifact_path(*parts: str) -> Path:
    return release_root() / "artifacts" / Path(*parts)


def output_root() -> Path:
    configured = os.environ.get("NEUROCOGMAP_OUTPUT_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    return DEFAULT_OUTPUT_ROOT


def output_path(*parts: str) -> Path:
    return output_root() / Path(*parts)


def env_path(name: str, default: str | Path | None = None) -> Path | None:
    value = os.environ.get(name)
    if value:
        return Path(value).expanduser().resolve()
    if default is None:
        return None
    return Path(default).expanduser()


def env_path_str(name: str, default: str | Path | None = None) -> str:
    value = env_path(name, default)
    return "" if value is None else str(value)
