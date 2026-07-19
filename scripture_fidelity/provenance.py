"""Release-safe build and execution provenance."""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

SYSTEM_SCHEMA_VERSION = "2"
PROMPT_TEMPLATE_VERSION = "scripture-fidelity-prompts-v4"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_repo_root(),
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return ""
    return result.stdout.strip()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_identity() -> dict:
    commit = (
        os.environ.get("SOURCE_COMMIT")
        or os.environ.get("GITHUB_SHA")
        or _git("rev-parse", "HEAD")
        or "unknown"
    )
    revision = os.environ.get("K_REVISION") or ""
    dirty = (
        bool(_git("status", "--porcelain"))
        if (_repo_root() / ".git").exists()
        else None
    )
    remote = (
        os.environ.get("SOURCE_REPOSITORY_URL")
        or _git("remote", "get-url", "origin")
        or None
    )
    lock_path = _repo_root() / "uv.lock"
    system_version = revision or f"git-{commit[:12]}"
    return {
        "system_version": system_version,
        "schema_version": SYSTEM_SCHEMA_VERSION,
        "repository_url": remote,
        "git_commit": commit,
        "git_dirty": dirty,
        "dependency_lock": lock_path.name if lock_path.is_file() else None,
        "dependency_lock_sha256": _file_sha256(lock_path),
        "prompt_template_version": PROMPT_TEMPLATE_VERSION,
    }


def execution_environment() -> dict:
    packages = {}
    for package in ("scripture-fidelity", "inspect-ai", "pydantic"):
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            packages[package] = None
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": packages,
    }
