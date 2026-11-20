# Vietnamese News NLP

Ứng dụng phân loại chủ đề tin tức tiếng Việt và tìm ba bài có nội dung gần nhất theo TF-IDF. Web demo đang chạy tại [Bản Tin — Vietnamese News NLP](https://abtract-news-project-3crruicceabgyojs66xe4l.streamlit.app/).

## Tính năng

- Dự đoán một trong 13 chuyên mục của bài báo.
- Hiển thị các điểm softmax để tham khảo. Đây là điểm ước lượng chưa được hiệu chuẩn, không nên hiểu là xác suất chắc chắn.
- Tìm tối đa ba bài gần nhất từ 4.550 bài train bằng cosine similarity trên TF-IDF. Đây là tương đồng từ vựng, không phải tìm kiếm ngữ nghĩa.
- Giao diện Streamlit; inference chạy bằng PyTorch trên CPU.

## Chạy ứng dụng

Cần Python 3.11 hoặc 3.12. Tại thư mục project:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
streamlit run app.py
```

Ứng dụng cần hai file runtime `artifacts/model.pt` và `artifacts/retrieval_index.pt`. Các file này được lưu trong repository để Streamlit Community Cloud có thể nạp model. Nếu bạn có ZIP output từ Kaggle, có thể nhập bằng:

```bash
python -m scripts.import_kaggle /path/to/results.zip
```

Sau đó chạy `streamlit run app.py`.

## Mô hình và dữ liệu

- Dataset dự kiến: [VLUS06/VietOnlineNews](https://huggingface.co/datasets/VLUS06/VietOnlineNews).
- Notebook gốc: `summary-text.ipynb`; bản notebook dùng để chạy thí nghiệm đặt ở `notebooks/kaggle_experiments.ipynb`.
- Pipeline dùng tokenizer regex tiếng Việt đơn giản, custom TF-IDF và hai classifier PyTorch: Linear và MLP.
- Tập train/validation/test có lần lượt 4.550 / 975 / 975 bài. Từ checkpoint Kaggle đã chọn, model là MLP; kết quả test ghi trong notebook: accuracy 86,97%, macro-F1 86,99%.
- Notebook tạo các split từ dữ liệu thu thập; điểm số này không phải đánh giá trên test split nguyên bản của dataset.
- Tham khảo `docs/IMPLEMENTATION.md` để biết cấu trúc module, lệnh huấn luyện lại và giới hạn đánh giá.

Dataset Hub hiện ghi license là `other`. Hãy kiểm tra điều khoản của nguồn trước khi phân phối lại dữ liệu. Repository chỉ chứa model và retrieval index, không kèm các CSV train/validation/test.

## Cấu trúc chính

```text
app.py                  # Giao diện Streamlit
src/                    # preprocessing, TF-IDF, model, inference, retrieval
configs/default.json    # Cấu hình cho pipeline huấn luyện lại
scripts/                # nhập artifact Kaggle và đóng gói deploy
docs/                   # hướng dẫn triển khai
notebooks/              # notebook thí nghiệm
artifacts/              # checkpoint và retrieval index đã chọn
```

## Huấn luyện và đánh giá lại

Cài thêm thư viện huấn luyện rồi chạy:

```bash
pip install -r requirements-train.txt
python -m src.prepare_data
python -m src.train
python -m src.evaluate
```

Chỉ chạy đánh giá test sau khi đã chọn mô hình bằng validation. Lệnh chuẩn bị dữ liệu dùng split train/test từ nguồn và lưu CSV vào `data/`. Xem `docs/IMPLEMENTATION.md` trước khi thay đổi cấu hình hoặc báo cáo metric.

## Deploy

Ứng dụng được deploy bằng Streamlit Community Cloud từ repository này, branch `main`, file `app.py`. Mỗi lần cập nhật source hoặc model, push commit lên `main`; Streamlit sẽ đồng bộ thay đổi.
