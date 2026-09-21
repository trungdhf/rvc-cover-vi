"""Bước 2: chuyển vocal gốc sang giọng của bạn bằng model RVC."""

from __future__ import annotations

import argparse
from pathlib import Path

from common import ffmpeg, load_config, resolve, workdir_of

from rvc_python.infer import RVCInference


def build_engine(conv: dict, models_dir: Path) -> RVCInference:
    rvc = RVCInference(models_dir=str(models_dir), device=conv.get("device", "cpu:0"))
    rvc.load_model(
        str(resolve(conv["model"])),
        version=conv.get("version", "v2"),
        index_path=str(resolve(conv["index"])) if conv.get("index") else "",
    )
    rvc.set_params(
        f0method=conv.get("f0method", "rmvpe"),
        f0up_key=conv.get("f0up_key", 0),
        index_rate=conv.get("index_rate", 0.6),
        filter_radius=conv.get("filter_radius", 3),
        rms_mix_rate=conv.get("rms_mix_rate", 0.25),
        protect=conv.get("protect", 0.33),
    )
    return rvc


def convert_chunked(rvc: RVCInference, vocals: Path, out: Path, chunk_sec: int, workdir: Path) -> Path:
    """Chia nhỏ vocal trước khi infer — cần thiết khi chạy CPU với bài dài."""
    parts_dir = workdir / "chunks"
    parts_dir.mkdir(parents=True, exist_ok=True)
    ffmpeg(
        [
            "-i", str(vocals),
            "-f", "segment",
            "-segment_time", str(chunk_sec),
            "-c", "copy",
            str(parts_dir / "part_%04d.wav"),
        ]
    )
    converted = []
    for part in sorted(parts_dir.glob("part_*.wav")):
        dst = parts_dir / f"conv_{part.name}"
        print(f"[rvc] {part.name}", flush=True)
        rvc.infer_file(str(part), str(dst))
        converted.append(dst)

    listfile = parts_dir / "concat.txt"
    listfile.write_text("".join(f"file '{p.name}'\n" for p in converted), encoding="utf-8")
    ffmpeg(["-f", "concat", "-safe", "0", "-i", str(listfile), "-c", "copy", str(out)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument(
        "--chunk-sec",
        type=int,
        default=0,
        help="Chia vocal thành đoạn N giây trước khi infer (0 = xử lý cả bài). Dùng khi chạy CPU.",
    )
    args = ap.parse_args()

    cfg = load_config(resolve(args.config))
    workdir = workdir_of(cfg)
    vocals = workdir / "vocals.wav"
    if not vocals.exists():
        raise SystemExit(f"Chưa có {vocals}. Chạy scripts/separate.py trước.")

    out = workdir / "vocals_converted.wav"
    rvc = build_engine(cfg["convert"], resolve("models"))
    if args.chunk_sec > 0:
        convert_chunked(rvc, vocals, out, args.chunk_sec, workdir)
    else:
        rvc.infer_file(str(vocals), str(out))
    print(f"vocal đã đổi giọng -> {out}")


if __name__ == "__main__":
    main()
