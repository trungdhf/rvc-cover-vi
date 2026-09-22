# rvc-cover-vi

Pipeline cover nhạc bằng giọng của chính bạn: tách vocal khỏi bài gốc (Demucs) → đổi giọng bằng model RVC → mix lại với nhạc nền.

```
bài hát gốc ──► separate.py ──┬─► vocals.wav ──► convert.py ──► vocals_converted.wav ──┐
                              └─► instrumental.wav ───────────────────────────────────┴─► mix.py ──► cover.mp3
```

## Cài đặt

```bash
sudo apt install ffmpeg python3-dev build-essential   # fairseq/pyworld build từ source

python -m venv .venv && source .venv/bin/activate
pip install "setuptools<70" wheel "cython<3"          # fairseq 0.12.2 không build được với setuptools mới
# CPU
pip install torch==2.3.1 torchaudio==2.3.1 --index-url https://download.pytorch.org/whl/cpu
# hoặc GPU: --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Lần chạy `convert.py` đầu tiên sẽ tự tải `hubert_base.pt` và `rmvpe.pt` (~600MB).

## Quy trình

### 1. Thu âm và chuẩn bị dataset

10-30 phút giọng bạn, thu cùng một mic, phòng ít vang, không nhạc nền. Nên có cả phần **hát** (không chỉ nói) và trải đủ quãng giọng bạn sẽ dùng khi cover.

```bash
python scripts/prepare_dataset.py --input recordings/ --out dataset
zip -r dataset.zip dataset
```

### 2. Train model

**Trên máy (CPU hoặc GPU), không cần web UI:**

```bash
python scripts/train_local.py --dataset dataset --epochs 100 --batch-size 4
# GPU: thêm --device cuda:0
```

Script tự clone repo RVC gốc, tải hubert/rmvpe/pretrained v2, chạy preprocess → f0 → feature → train → index, rồi chép `phien-singer.pth` + `phien-singer.index` vào `models/`. Dataset ~2 phút, 100 epochs mất ~35 phút trên 8 CPU.

**Trên Colab GPU:** mở `notebooks/RVC_train_colab.ipynb` (T4), upload `dataset.zip`, làm theo các cell. Lưu ý Colab free có thể tự ngắt runtime giữa chừng khi tài khoản hết compute unit.

### 3. Cover một bài

```bash
cp config.example.yaml config.yaml   # sửa đường dẫn bài hát + tên model
python scripts/run_cover.py --config config.yaml
# CPU với bài dài thì chia nhỏ để đỡ tốn RAM:
python scripts/run_cover.py --config config.yaml --chunk-sec 30
```

Chạy từng bước riêng nếu muốn nghe thử giữa chừng:

```bash
python scripts/separate.py --config config.yaml
python scripts/convert.py  --config config.yaml
python scripts/mix.py      --config config.yaml
```

Kết quả nằm trong `output/<tên bài>/`: `vocals.wav`, `instrumental.wav`, `vocals_converted.wav`, `cover.wav`, `cover.mp3`.

## Model sẵn có trong repo

`models/phien-singer.pth` + `models/phien-singer.index` (lưu bằng Git LFS) là model đã train sẵn. Clone xong nhớ kéo file LFS về:

```bash
git lfs install
git lfs pull
```

| File | Dataset | Ghi chú |
|---|---|---|
| `phien-singer.pth` / `.index` | 6.7 phút, 80 epochs | bản đang dùng |
| `phien-singer-v1.pth` / `.index` | 2.1 phút, 100 epochs | bản đầu, giữ để so sánh |
| `phien-speaker.pth` / `.index` | 1.6 phút giọng đọc, 120 epochs | dùng cho giọng nói/đọc |

Với giọng đọc nên hạ `index_rate` ~0.2-0.3 và nâng `protect` ~0.4-0.5 cho tự nhiên hơn.

## Tạo mp3 giọng model từ một file audio

Dùng khi đã có sẵn vocal (a cappella, file thu, hoặc stem vocal) và chỉ muốn đổi sang giọng model:

```bash
source .venv/bin/activate
mkdir -p output/mybai
ffmpeg -y -i nguon.mp3 -ar 44100 -ac 2 output/mybai/vocals.wav   # bỏ qua bước separate

cat > config.mybai.yaml <<'YAML'
song: input/nguon.mp3
workdir: output/mybai
separate:
  model: htdemucs
  two_stems: vocals
convert:
  model: models/phien-singer.pth
  index: models/phien-singer.index
  device: cpu:0        # GPU: cuda:0
  f0method: rmvpe
  f0up_key: 0
  index_rate: 0.35
  protect: 0.33
  filter_radius: 3
  rms_mix_rate: 0.25
mix:
  vocal_gain_db: 0.0
  inst_gain_db: -1.0
  target_lufs: -14.0
  reverb: light
YAML

python scripts/convert.py --config config.mybai.yaml --chunk-sec 30
```

Ra `output/mybai/vocals_converted.wav`. Xuất mp3:

```bash
# vocal khô
ffmpeg -y -i output/mybai/vocals_converted.wav -b:a 320k cover_vocal.mp3
# thêm reverb nhẹ + chuẩn hoá loudness
ffmpeg -y -i output/mybai/vocals_converted.wav \
  -af "aecho=0.8:0.85:40:0.18,loudnorm=I=-14:TP=-1.5:LRA=11" -b:a 320k cover_vocal.mp3
```

Có nhạc nền riêng thì chép vào `output/mybai/instrumental.wav` rồi chạy `python scripts/mix.py --config config.mybai.yaml` → `output/mybai/cover.mp3`.

Nếu file nguồn còn dính nhạc/guitar thì chạy `python scripts/separate.py --config config.mybai.yaml` trước thay cho bước ffmpeg ở trên; Demucs sẽ tạo luôn cả `vocals.wav` lẫn `instrumental.wav`.

## Chỉnh chất lượng

| Triệu chứng | Tham số cần sửa |
|---|---|
| Giọng sai quãng, chênh tông | `convert.f0up_key` (±12 khi đổi giới tính giọng) |
| Nghe máy móc, mất hơi thở | giảm `index_rate` (0.3-0.5), tăng `protect` (0.4) |
| Không giống giọng mình | tăng `index_rate`, train thêm epoch, thêm dữ liệu hát |
| Pitch lỗi ở nốt cao | dùng `f0method: rmvpe` (mặc định), tránh `pm` |
| Vocal chìm dưới nhạc | `mix.vocal_gain_db: +2`, `mix.inst_gain_db: -2` |
| Tách vocal còn dính nhạc | `separate.model: htdemucs_ft` |

Vocal gốc còn sạch sẽ cho kết quả tốt hơn nhiều — nếu có bản instrumental/karaoke chính thức của bài, dùng nó thay cho stem Demucs.

## Lưu ý pháp lý

Cover dùng bản ghi gốc chạm vào quyền tác giả và quyền liên quan của bản ghi. Dùng cho mục đích cá nhân/học tập; muốn phát hành cần xin phép hoặc qua dịch vụ cấp phép cover. Chỉ train model trên giọng của chính bạn hoặc giọng bạn được phép sử dụng.
