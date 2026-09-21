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

### 2. Train model trên Colab GPU

Mở `notebooks/RVC_train_colab.ipynb` trên Colab (T4 GPU), upload `dataset.zip`, làm theo các cell. Kết quả là `my-voice.pth` + `added_*.index` — copy cả hai vào `models/`.

Train cần GPU; máy CPU-only thì chỉ chạy được bước infer ở dưới.

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
