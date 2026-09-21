"""Chuẩn bị dataset giọng nói/hát để train RVC.

Nhận các file thu âm thô, chuyển về mono 44.1kHz, cắt theo khoảng lặng thành các
đoạn 4-10 giây và ghi ra thư mục dataset (có thể zip lên Colab để train).
"""

from __future__ import annotations

import argparse
import re
import subprocess
from pathlib import Path

from common import ffmpeg

SILENCE_RE = re.compile(r"silence_(start|end): (-?\d+\.?\d*)")
AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac", ".opus"}


def detect_silences(path: Path, noise_db: float, min_silence: float) -> list[tuple[float, float]]:
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-i", str(path),
            "-af", f"silencedetect=noise={noise_db}dB:d={min_silence}",
            "-f", "null", "-",
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    starts: list[float] = []
    ends: list[float] = []
    for kind, value in SILENCE_RE.findall(proc.stderr):
        (starts if kind == "start" else ends).append(float(value))
    return list(zip(starts, ends + [0.0] * (len(starts) - len(ends))))


def duration_of(path: Path) -> float:
    proc = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
        capture_output=True,
        text=True,
        check=True,
    )
    return float(proc.stdout.strip())


def segments_from_silences(
    silences: list[tuple[float, float]], total: float, min_len: float, max_len: float
) -> list[tuple[float, float]]:
    cuts = [0.0]
    for start, end in silences:
        cuts.append((start + end) / 2 if end else start)
    cuts.append(total)

    segments: list[tuple[float, float]] = []
    seg_start = cuts[0]
    for cut in cuts[1:]:
        if cut - seg_start < min_len:
            continue
        while cut - seg_start > max_len:
            segments.append((seg_start, seg_start + max_len))
            seg_start += max_len
        if cut - seg_start >= min_len:
            segments.append((seg_start, cut))
        seg_start = cut
    return segments


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="File hoặc thư mục chứa bản thu giọng của bạn")
    ap.add_argument("--out", default="dataset", help="Thư mục dataset đầu ra")
    ap.add_argument("--min-len", type=float, default=4.0)
    ap.add_argument("--max-len", type=float, default=10.0)
    ap.add_argument("--noise-db", type=float, default=-35.0, help="Ngưỡng coi là khoảng lặng")
    ap.add_argument("--min-silence", type=float, default=0.35)
    args = ap.parse_args()

    src = Path(args.input)
    files = sorted(p for p in (src.rglob("*") if src.is_dir() else [src]) if p.suffix.lower() in AUDIO_EXTS)
    if not files:
        raise SystemExit(f"Không tìm thấy file audio trong {src}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    index = 0
    for f in files:
        mono = out_dir / f"_tmp_{f.stem}.wav"
        ffmpeg(["-i", str(f), "-ac", "1", "-ar", "44100", "-af", "highpass=f=60,loudnorm", str(mono)])
        segments = segments_from_silences(
            detect_silences(mono, args.noise_db, args.min_silence),
            duration_of(mono),
            args.min_len,
            args.max_len,
        )
        for start, end in segments:
            index += 1
            ffmpeg(["-ss", f"{start:.3f}", "-to", f"{end:.3f}", "-i", str(mono), str(out_dir / f"{index:04d}.wav")])
        mono.unlink()

    total_min = sum(duration_of(p) for p in out_dir.glob("[0-9]*.wav")) / 60
    print(f"{index} đoạn, tổng {total_min:.1f} phút -> {out_dir}")
    if total_min < 10:
        print("Cảnh báo: nên có ít nhất 10 phút audio sạch để model ổn định.")


if __name__ == "__main__":
    main()
