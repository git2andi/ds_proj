"""Shared helpers for local configuration experiments.

These helpers intentionally edit only scalar YAML values while preserving the
comments and formatting in config.yaml. They also keep a crash-recovery backup
so experimental scripts cannot silently leave the main configuration patched.
"""

from __future__ import annotations

import atexit
import csv
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT / "config.yaml"
BACKUP_PATH = ROOT / "config.yaml.experiment_backup"

PathKey = tuple[str, ...]


def _yaml_scalar(value: Any) -> str:
    """Serialize a scalar in compact YAML syntax without PyYAML document markers."""
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        # JSON strings are valid YAML scalars and safely preserve spaces and punctuation.
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, (int, float)):
        return repr(value)
    raise ValueError(f"only scalar config values are supported, got {value!r}")


def patch_yaml_scalars(text: str, updates: dict[PathKey, Any]) -> str:
    """Patch exact scalar paths without removing comments or reformatting YAML."""
    remaining = dict(updates)
    stack: list[tuple[int, str]] = []
    output: list[str] = []
    key_pattern = re.compile(r"^(?P<indent>\s*)(?P<key>[^#][^:]*?):(?P<tail>.*?)(?P<newline>\r?\n)?$")

    for raw_line in text.splitlines(keepends=True):
        stripped = raw_line.lstrip()
        if not stripped or stripped.startswith("#"):
            output.append(raw_line)
            continue

        match = key_pattern.match(raw_line)
        if not match:
            output.append(raw_line)
            continue

        indent_text = match.group("indent")
        indent = len(indent_text.replace("\t", "    "))
        key_token = match.group("key").strip()
        key = key_token.strip("\"'")
        tail = match.group("tail") or ""
        newline = match.group("newline") or ""

        while stack and stack[-1][0] >= indent:
            stack.pop()
        path = tuple(item[1] for item in stack) + (key,)

        value_part = tail
        comment = ""
        if "#" in tail:
            value_part, comment_tail = tail.split("#", 1)
            comment = "#" + comment_tail

        if path in remaining:
            if not value_part.strip():
                raise ValueError(f"cannot replace mapping section {'.'.join(path)} with a scalar")
            spacing = " "
            if comment:
                comment = "  " + comment.lstrip()
            output.append(
                f"{indent_text}{match.group('key')}: { _yaml_scalar(remaining.pop(path))}{comment}{newline}"
            )
            continue

        output.append(raw_line)
        if not value_part.strip():
            stack.append((indent, key))

    if remaining:
        missing = ", ".join(".".join(path) for path in remaining)
        raise KeyError(f"config path(s) not found: {missing}")
    return "".join(output)


def read_config() -> dict[str, Any]:
    data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("config.yaml must contain a mapping")
    return data


def value_at(data: dict[str, Any], path: PathKey) -> Any:
    current: Any = data
    for part in path:
        if not isinstance(current, dict):
            raise KeyError(".".join(path))
        if part in current:
            current = current[part]
        elif part.isdigit() and int(part) in current:
            current = current[int(part)]
        else:
            raise KeyError(".".join(path))
    return current


class ConfigExperimentSession:
    """Back up config.yaml, write temporary scalar patches, and restore safely."""

    def __init__(self) -> None:
        self.original_text = ""
        self.active = False

    @staticmethod
    def restore_stale_backup() -> bool:
        if not BACKUP_PATH.exists():
            return False
        shutil.copy2(BACKUP_PATH, CONFIG_PATH)
        BACKUP_PATH.unlink()
        return True

    def __enter__(self) -> "ConfigExperimentSession":
        if BACKUP_PATH.exists():
            raise RuntimeError(
                f"Found {BACKUP_PATH.name}. A prior experiment may have stopped unexpectedly. "
                "Run this script with --restore-config before starting a new experiment."
            )
        self.original_text = CONFIG_PATH.read_text(encoding="utf-8")
        BACKUP_PATH.write_text(self.original_text, encoding="utf-8")
        self.active = True
        atexit.register(self.restore)
        return self

    def write(self, updates: dict[PathKey, Any]) -> None:
        if not self.active:
            raise RuntimeError("configuration experiment session is not active")
        CONFIG_PATH.write_text(patch_yaml_scalars(self.original_text, updates), encoding="utf-8")

    def restore(self) -> None:
        if not self.active:
            return
        CONFIG_PATH.write_text(self.original_text, encoding="utf-8")
        if BACKUP_PATH.exists():
            BACKUP_PATH.unlink()
        self.active = False

    def apply_to_original(self, updates: dict[PathKey, Any]) -> None:
        """Persist selected scalar updates while still removing the backup."""
        if not self.active:
            raise RuntimeError("configuration experiment session is not active")
        patched = patch_yaml_scalars(self.original_text, updates)
        CONFIG_PATH.write_text(patched, encoding="utf-8")
        self.original_text = patched
        if BACKUP_PATH.exists():
            BACKUP_PATH.write_text(patched, encoding="utf-8")

    def __exit__(self, exc_type, exc, tb) -> None:
        self.restore()


@dataclass
class ProcessResult:
    ok: bool
    returncode: int
    stdout: str
    stderr: str
    run_dir: str = ""
    run_json: dict[str, Any] | None = None


def run_topic_subprocess(topic: str, *, timeout_seconds: int) -> ProcessResult:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    try:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "main.py"), topic],
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ProcessResult(
            ok=False,
            returncode=124,
            stdout=exc.stdout or "",
            stderr=(exc.stderr or "") + f"\nTimed out after {timeout_seconds} seconds.",
        )

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    match = re.search(r"(?m)^Logs:\s*(.+?)\s*$", stdout)
    run_dir = match.group(1).strip() if match else ""
    payload: dict[str, Any] | None = None
    if run_dir:
        path = Path(run_dir)
        if not path.is_absolute():
            path = ROOT / path
        json_path = path / "run.json"
        if json_path.exists():
            try:
                payload = json.loads(json_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                stderr += f"\nCould not read {json_path}: {exc}"

    return ProcessResult(
        ok=completed.returncode == 0 and payload is not None,
        returncode=completed.returncode,
        stdout=stdout,
        stderr=stderr,
        run_dir=run_dir,
        run_json=payload,
    )


def validate_current_config(timeout_seconds: int = 30) -> tuple[bool, str]:
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(ROOT / 'src')!r}); "
        "import config_loader; print('ok')"
    )
    try:
        completed = subprocess.run(
            [sys.executable, "-c", code],
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=timeout_seconds,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return False, "configuration validation timed out"
    message = (completed.stderr or completed.stdout or "").strip()
    return completed.returncode == 0, message


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def zip_directory(root: Path, target: Path) -> Path:
    if target.exists():
        target.unlink()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(root.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(root.parent))
    return target


def slugify(value: str, limit: int = 80) -> str:
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", value).strip("_.-")
    return (slug or "experiment")[:limit]
