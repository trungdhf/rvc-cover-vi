"""Bước 3: ghép vocal đã đổi giọng với nhạc nền và chuẩn hoá loudness."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyloudnorm as pyln
import soundfile as sf

from common import ffmpeg, load_config, resolve, workdir_of

REVERB = {
    "none": None,
    "light": "aecho=0.8:0.85:40:0.18",
    "hall": "aecho=0.8:0.9:60|90:0.3|0.22",
}


def mix(vocals: Path, inst: Path, out: Path, cfg: dict) -> Path:
    reverb = REVERB.get(cfg.get("reverb", "light"))
    vocal_chain = [f"volume={cfg.get('vocal_gain_db', 0.0)}dB"]
    if reverb:
        vocal_chain.append(reverb)

    filter_complex = (
        f"[0:a]{','.join(vocal_chain)},aresample=44100[v];"
        f"[1:a]volume={cfg.get('inst_gain_db', 0.0)}dB,aresample=44100[i];"
        f"[v][i]amix=inputs=2:duration=longest:normalize=0[a]"
    )
    ffmpeg(["-i", str(vocals), "-i", str(inst), "-filter_complex", filter_complex, "-map", "[a]", str(out)])

    target = cfg.get("target_lufs")
    if target is not None:
        normalize_loudness(out, float(target))
    return out


def normalize_loudness(path: Path, target_lufs: float) -> None:
    data, rate = sf.read(path)
    meter = pyln.Meter(rate)
    loudness = meter.integrated_loudness(data)
    normalized = pyln.normalize.loudness(data, loudness, target_lufs)
    peak = float(np.max(np.abs(normalized)))
    if peak > 0.99:
        normalized = normalized * (0.99 / peak)
    sf.write(path, normalized, rate)
    print(f"loudness {loudness:.1f} LUFS -> {target_lufs:.1f} LUFS")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    workdir = workdir_of(cfg)
    vocals = workdir / "vocals_converted.wav"
    if not vocals.exists():
        vocals = workdir / "vocals.wav"
    out = mix(vocals, workdir / "instrumental.wav", workdir / "cover.wav", cfg.get("mix", {}))
    ffmpeg(["-i", str(out), "-b:a", "320k", str(out.with_suffix(".mp3"))])
    print(f"bản cover -> {out.with_suffix('.mp3')}")


if __name__ == "__main__":
    main()
