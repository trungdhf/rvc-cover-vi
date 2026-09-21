"""Tiện ích dùng chung cho pipeline cover."""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError(f"Config không hợp lệ: {path}")
    return cfg


def resolve(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def workdir_of(cfg: dict[str, Any]) -> Path:
    wd = resolve(cfg.get("workdir", "output/run"))
    wd.mkdir(parents=True, exist_ok=True)
    return wd


def run(cmd: list[str]) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def ffmpeg(args: list[str]) -> None:
    run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args])
