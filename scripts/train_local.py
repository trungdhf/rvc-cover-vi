"""Train model RVC ngay trên máy (CPU hoặc GPU), không cần Colab/web UI.

Dùng repo RVC gốc ở chế độ dòng lệnh: preprocess -> f0 -> hubert -> train -> index.
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from random import shuffle

HF_BASE = "https://huggingface.co/lj1995/VoiceConversionWebUI/resolve/main"
RVC_REPO = "https://github.com/RVC-Project/Retrieval-based-Voice-Conversion-WebUI.git"
SR_MAP = {"32k": 32000, "40k": 40000, "48k": 48000}


def sh(cmd: list[str], cwd: Path, env: dict[str, str] | None = None) -> None:
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=cwd, env=env, check=True)


def wget(url: str, dst: Path) -> None:
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["wget", "-q", "-O", str(dst), url], check=True)


def ensure_rvc(rvc_dir: Path) -> None:
    if not rvc_dir.exists():
        sh(["git", "clone", "--depth", "1", RVC_REPO, str(rvc_dir)], cwd=rvc_dir.parent)
    for name in ("config.json", "preprocessor_config.json", "pytorch_model.bin"):
        wget(f"{HF_BASE}/hubert_base/{name}", rvc_dir / "assets" / "hubert_base" / name)
    wget(f"{HF_BASE}/rmvpe.pt", rvc_dir / "assets" / "rmvpe" / "rmvpe.pt")
    for name in ("f0G40k.pth", "f0D40k.pth"):
        wget(f"{HF_BASE}/pretrained_v2/{name}", rvc_dir / "assets" / "pretrained_v2" / name)
    if not (rvc_dir / "logs" / "mute").exists():
        zip_path = rvc_dir / "mute.zip"
        wget(f"{HF_BASE}/mute.zip", zip_path)
        shutil.unpack_archive(str(zip_path), str(rvc_dir / "logs"))


def write_filelist(rvc_dir: Path, name: str, sr: str) -> None:
    exp_dir = rvc_dir / "logs" / name
    gt, feat = exp_dir / "0_gt_wavs", exp_dir / "3_feature768"
    f0, f0nsf = exp_dir / "2a_f0", exp_dir / "2b-f0nsf"
    names = (
        {p.stem for p in gt.iterdir()}
        & {p.stem for p in feat.iterdir()}
        & {p.name.split(".")[0] for p in f0.iterdir()}
        & {p.name.split(".")[0] for p in f0nsf.iterdir()}
    )
    lines = [
        f"{gt}/{n}.wav|{feat}/{n}.npy|{f0}/{n}.wav.npy|{f0nsf}/{n}.wav.npy|0"
        for n in sorted(names)
    ]
    mute = rvc_dir / "logs" / "mute"
    lines += 2 * [
        f"{mute}/0_gt_wavs/mute{sr}.wav|{mute}/3_feature768/mute.npy"
        f"|{mute}/2a_f0/mute.wav.npy|{mute}/2b-f0nsf/mute.wav.npy|0"
    ]
    shuffle(lines)
    (exp_dir / "filelist.txt").write_text("\n".join(lines), encoding="utf-8")

    sys.path.insert(0, str(rvc_dir))
    from configs.config import Config

    cfg = copy.deepcopy(Config().json_config[f"v1/{sr}.json" if sr == "40k" else f"v2/{sr}.json"])
    cfg.pop("speaker_info", None)
    (exp_dir / "config.json").write_text(
        json.dumps(cfg, ensure_ascii=False, indent=4, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, help="Thư mục wav đã cắt (output của prepare_dataset.py)")
    ap.add_argument("--rvc-dir", default="rvc-src", help="Nơi clone repo RVC gốc")
    ap.add_argument("--name", default="my-voice")
    ap.add_argument("--sr", default="40k", choices=list(SR_MAP))
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--device", default="cpu", help="cpu hoặc cuda:0")
    ap.add_argument("--workers", type=int, default=os.cpu_count() or 4)
    ap.add_argument("--out", default="models", help="Thư mục chép .pth + .index sau khi train")
    args = ap.parse_args()

    rvc_dir = Path(args.rvc_dir).resolve()
    rvc_dir.parent.mkdir(parents=True, exist_ok=True)
    ensure_rvc(rvc_dir)

    exp_dir = rvc_dir / "logs" / args.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    env = {**os.environ, "PYTHONPATH": str(rvc_dir)}
    py = [sys.executable, "-m"]
    dataset = Path(args.dataset).resolve()
    cuda = args.device.startswith("cuda")

    sh(py + ["train.preprocess", str(dataset), str(SR_MAP[args.sr]), str(args.workers),
             str(exp_dir), "False", "3.7"], rvc_dir, env)
    if cuda:
        sh(py + ["train.dataset.extract_f0", "cuda", "1", "0", "0", str(exp_dir), "True"], rvc_dir, env)
    else:
        sh(py + ["train.dataset.extract_f0", "cpu", str(exp_dir), str(args.workers), "rmvpe"], rvc_dir, env)
    sh(py + ["train.dataset.extract_hubert_feature", args.device, "1", "0", "0",
             str(exp_dir), "v2", "True" if cuda else "False"], rvc_dir, env)

    write_filelist(rvc_dir, args.name, args.sr)

    train_cmd = py + [
        "train.train", "-e", args.name, "-sr", args.sr, "-f0", "1",
        "-bs", str(args.batch_size), "-te", str(args.epochs), "-se", str(args.save_every),
        "-pg", "assets/pretrained_v2/f0G40k.pth", "-pd", "assets/pretrained_v2/f0D40k.pth",
        "-l", "1", "-c", "0", "-sw", "1", "-v", "v2",
    ]
    if cuda:
        train_cmd += ["-g", "0"]
    (rvc_dir / "assets" / "weights").mkdir(parents=True, exist_ok=True)
    (rvc_dir / "assets" / "indices").mkdir(parents=True, exist_ok=True)
    sh(train_cmd, rvc_dir, env)

    weight = rvc_dir / "assets" / "weights" / f"{args.name}.pth"
    if not weight.exists():
        sh(py[:1] + ["-c",
                     "from train.process_ckpt import extract_small_model;"
                     f"extract_small_model('logs/{args.name}/G_2333333.pth','{args.name}',"
                     f"'{args.sr}',1,'{args.epochs} epochs','v2')"], rvc_dir, env)
    sh(py + ["train.train_index", args.name, "v2", "assets/indices", str(args.workers), "auto"],
       rvc_dir, env)

    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy(weight, out / f"{args.name}.pth")
    index = sorted(exp_dir.glob("added_*.index"))
    if index:
        shutil.copy(index[0], out / f"{args.name}.index")
    print(f"model -> {out / (args.name + '.pth')}")


if __name__ == "__main__":
    main()
