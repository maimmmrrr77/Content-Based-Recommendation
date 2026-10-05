# 🎥 YouTube Video Recommendation System (Content-Based)

Mô hình Gợi ý Video dựa trên Nội dung (Content-Based Filtering) xây dựng trên tập dữ liệu **YouTube Trending Videos (US)**. Hệ thống trích xuất đặc trưng văn bản từ *Tiêu đề (Title)*, *Thẻ (Tags)*, và *Mô tả (Description)* bằng kỹ thuật **TF-IDF Vectorization** và đo độ tương đồng ngữ nghĩa bằng **Cosine Similarity**.

---

## Tính năng chính

- **Xử lý Dữ liệu Văn bản (Text Preprocessing):** Chuẩn hóa chữ thường, lọc ký tự đặc biệt và loại bỏ từ dừng (*stop words*).
- **Trích xuất Đặc trưng (Feature Extraction):** Mã hóa thông tin văn bản tổng hợp thành Vector không gian bằng **TF-IDF Vectorizer** (giới hạn 5,000 đặc trưng nổi bật).
- **So khớp Tương đồng (Cosine Similarity Matrix):** Đánh giá khoảng cách ngữ nghĩa giữa các vector trong không gian vector đa chiều.
- **Đánh giá Hiệu năng Đầu ra (Evaluation Framework):**
  - Tách tập dữ liệu theo tỷ lệ **Train/Test (80/20)**.
  - Đo lường chất lượng hệ thống bằng bộ chỉ số **Precision@K**, **Recall@K**, **F1-Score@K** và **Accuracy@K** theo độ đồng nhất danh mục (*Category Consistency*).
  - Trực quan hóa biến thiên của chỉ số qua các ngưỡng $K \in [1, 10]$.

---

## Công nghệ Sử dụng

- **Ngôn ngữ:** Python 3.x
- **Thư viện chính:**
  - `pandas`, `numpy`: Thao tác và biến đổi dữ liệu dạng bảng.
  - `scikit-learn`: `TfidfVectorizer`, `cosine_similarity`, `train_test_split`.
  - `matplotlib`: Trực quan hóa đồ thị hiệu năng.

---

## Kết quả Đánh giá Mô hình (Evaluation Metrics)

Hệ thống được kiểm thử dựa trên khả năng gợi ý đúng các video có cùng `category_id` trong tập dữ liệu thử nghiệm:

| Top-K | Precision@K | Recall@K | F1-Score@K | Accuracy@K |
| :---: | :---: | :---: | :---: | :---: |
| **Top-1** | High | Low | Balanced | High |
| **Top-5** | Standard | High | Optimal | Standard |
| **Top-10** | Broad | Maximum | Broad | Broad |


---

## Cấu trúc Dự án

```text
├── USvideos.csv              # Tập dữ liệu YouTube Trending Videos
├── recommender system.py     # Script chính xử lý, huấn luyện và đánh giá mô hình
└── README.md                 # Tài liệu hướng dẫn dự án