"""Chạy toàn bộ pipeline: tách vocal -> đổi giọng -> mix."""

from __future__ import annotations

import argparse
from pathlib import Path

import convert as convert_step
import mix as mix_step
import separate as separate_step
from common import ffmpeg, load_config, resolve, workdir_of


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--chunk-sec", type=int, default=0)
    ap.add_argument("--skip-separate", action="store_true")
    ap.add_argument("--skip-convert", action="store_true", help="Chỉ mix lại, giữ vocal đã convert")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    workdir = workdir_of(cfg)
    sep = cfg.get("separate", {})

    if not args.skip_separate:
        separate_step.separate(
            resolve(cfg["song"]),
            workdir,
            model=sep.get("model", "htdemucs"),
            two_stems=sep.get("two_stems", "vocals"),
        )

    converted = workdir / "vocals_converted.wav"
    if not args.skip_convert:
        rvc = convert_step.build_engine(cfg["convert"], resolve("models"))
        if args.chunk_sec > 0:
            convert_step.convert_chunked(rvc, workdir / "vocals.wav", converted, args.chunk_sec, workdir)
        else:
            rvc.infer_file(str(workdir / "vocals.wav"), str(converted))

    out = mix_step.mix(converted, workdir / "instrumental.wav", workdir / "cover.wav", cfg.get("mix", {}))
    ffmpeg(["-i", str(out), "-b:a", "320k", str(out.with_suffix(".mp3"))])
    print(f"xong: {out.with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
