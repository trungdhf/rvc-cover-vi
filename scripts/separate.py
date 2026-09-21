"""Bước 1: tách vocal và nhạc nền khỏi bài hát gốc bằng Demucs."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from common import ffmpeg, load_config, resolve, run, workdir_of


def separate(song: Path, workdir: Path, model: str = "htdemucs", two_stems: str = "vocals") -> tuple[Path, Path]:
    raw = workdir / "raw.wav"
    ffmpeg(["-i", str(song), "-ac", "2", "-ar", "44100", str(raw)])

    stems_dir = workdir / "stems"
    cmd = [sys.executable, "-m", "demucs", "-n", model, "-o", str(stems_dir)]
    if two_stems:
        cmd += ["--two-stems", two_stems]
    cmd.append(str(raw))
    run(cmd)

    produced = stems_dir / model / raw.stem
    vocals = workdir / "vocals.wav"
    inst = workdir / "instrumental.wav"
    shutil.copy(produced / "vocals.wav", vocals)
    shutil.copy(produced / f"no_{two_stems}.wav", inst)
    return vocals, inst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    sep = cfg.get("separate", {})
    vocals, inst = separate(
        resolve(cfg["song"]),
        workdir_of(cfg),
        model=sep.get("model", "htdemucs"),
        two_stems=sep.get("two_stems", "vocals"),
    )
    print(f"vocals  -> {vocals}")
    print(f"nhạc nền -> {inst}")


if __name__ == "__main__":
    main()
