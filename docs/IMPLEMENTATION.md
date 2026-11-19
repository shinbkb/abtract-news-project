# Vietnamese News NLP — chạy web và triển khai

## 1. Tải kết quả từ Kaggle
Chạy cell lưu checkpoint trong notebook. Sau đó chép nội dung `scripts/export_kaggle.py` vào cell cuối và chạy.
Tải `news_project_output.zip` qua link hoặc danh sách output. Nếu phiên Kaggle đã mất dữ liệu, cần chạy lại notebook.

## 2. Cài và nhập model
Chạy trong thư mục project, Python 3.11 hoặc 3.12:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
python -m scripts.import_kaggle /duong/dan/news_project_output.zip
streamlit run app.py
```
Lệnh import chỉ đọc checkpoint với `weights_only=True`; dùng output do chính bạn tạo.
Nó kiểm tra kích thước model/index và nếu có train.csv, tính lại toàn bộ vector train để xác minh cùng vocabulary/IDF.
Không cần train lại để dùng model notebook. Notebook chỉ xuất mô hình thắng, không thể tái tạo checkpoint mô hình thua từ output đó.
`best_linear.pt`/`best_mlp.pt` chỉ có khi chạy huấn luyện bằng source.

## 3. Cấu trúc
- `src/prepare_data.py`: lấy mẫu từ split gốc, dedup content/title chuẩn hóa trước chia.
- `src/preprocessing.py`, `tfidf.py`: tokenizer tương thích notebook, vocabulary/IDF train-only.
- `src/dataset.py`: đọc split và chặn trùng content/title giữa các tập.
- `src/models.py`, `train.py`: Linear, MLP, early stopping và chọn bằng validation.
- `src/metrics.py`, `evaluate.py`: metrics PyTorch, confusion CSV, lỗi dự đoán.
- `src/retrieval.py`, `inference.py`: inference CPU, Top 3 cosine dương, loại bài nguồn.
- `data/`: CSV; `artifacts/`: checkpoint và index (được bỏ qua bởi Git).
- `notebooks/kaggle_experiments.ipynb`: bản sao notebook gốc đã xóa output.
- `app.py`: Streamlit; `Dockerfile`: deploy cổng 7860.

## 4. Huấn luyện lại (tùy chọn)
```bash
pip install -r requirements-train.txt
python -m src.prepare_data
python -m src.train
# Chỉ chạy test sau khi đã chốt cấu hình:
python -m src.evaluate
```
Chỉnh `configs/default.json`. Chuẩn bị dữ liệu mới dùng split test gốc khác notebook cũ,
vì vậy không kỳ vọng tái tạo chính xác điểm cũ. Dùng revision SHA trong dataset_summary.json để cố định nguồn.
Near-duplicate chưa được bảo đảm loại hết. Không đổi mô hình dựa trên test rồi báo cùng test như đánh giá độc lập.

## 5. Docker / Hugging Face Spaces
```bash
python -m scripts.package_deploy /tmp/news-deploy
cd /tmp/news-deploy
docker build -t vietnamese-news .
docker run --rm -p 7860:7860 vietnamese-news
```
Mở http://localhost:7860. Để deploy Spaces, tạo Space SDK Docker, upload nội dung thư mục deploy.
README trong gói đã có `sdk: docker` và `app_port: 7860`.
Tài liệu: https://huggingface.co/docs/hub/spaces-sdks-docker
Gói deploy chỉ có model/index và source web; không chứa CSV test/val hay thông tin đăng nhập.
Index chứa tiêu đề/trích đoạn dữ liệu nguồn: kiểm tra quyền phân phối trước khi public.
Chưa tạo tài nguyên cloud hay triển khai URL cho đến khi có model và tài khoản đích.

## 6. Kiểm thử
```bash
python -m unittest discover -s tests -v
```
Dữ liệu giả trong tests chỉ để kiểm tra kỹ thuật, không dùng làm kết quả demo hoặc metrics thực nghiệm.
