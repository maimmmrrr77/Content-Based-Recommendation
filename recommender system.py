import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split


# =========================================================
# 1. Đọc & xử lý dữ liệu
# =========================================================
df = pd.read_csv("USvideos.csv")
df = df.dropna(subset=['title', 'tags', 'description'])

# Lấy 100 video để chạy nhanh
df = df.head(500)

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9 ]', " ", text)
    return text

df['clean_title'] = df['title'].apply(clean_text)
df['clean_tags'] = df['tags'].apply(clean_text)
df['clean_desc'] = df['description'].apply(clean_text)
df['combined_text'] = df['clean_title'] + " " + df['clean_tags'] + " " + df['clean_desc']


# =========================================================
# 2. Chia dữ liệu TRAIN / TEST
# =========================================================
train_df, test_df = train_test_split(
    df,
    test_size=0.2,
    random_state=42,
    shuffle=True
)

print("Số lượng TRAIN:", len(train_df))
print("Số lượng TEST:", len(test_df))


# =========================================================
# 3. TF-IDF TRAIN & TEST
# =========================================================
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)

tfidf_train = tfidf.fit_transform(train_df['combined_text'])
tfidf_test = tfidf.transform(test_df['combined_text'])

print("\nTF-IDF Train:", tfidf_train.shape)
print("TF-IDF Test:", tfidf_test.shape)


# =========================================================
# 4. Cosine similarity TEST với TRAIN
# =========================================================
cosine_sim_test_train = cosine_similarity(tfidf_test, tfidf_train)


# =========================================================
# 5. Hàm gợi ý video (từ TEST → TRAIN)
# =========================================================
def recommend_from_test(test_index, top_k=10):
    sim_scores = cosine_sim_test_train[test_index]
    best_indices = sim_scores.argsort()[::-1][:top_k]
    return train_df[['title', 'channel_title', 'views']].iloc[best_indices]


# Test thử gợi ý
print("\nVí dụ gợi ý video:")
sample_test_title = test_df['title'].iloc[0]
print("Video test:", sample_test_title)
print(recommend_from_test(0, top_k=5))


# =========================================================
# 6. Đánh giá mô hình TRAIN / TEST
# =========================================================
def evaluate_train_test(max_k=20):
    avg_scores = []
    for k in range(1, max_k + 1):
        scores = []
        for i in range(tfidf_test.shape[0]):
            sim_row = cosine_sim_test_train[i]
            topk = sim_row.argsort()[::-1][:k]
            scores.extend(sim_row[topk])
        avg_scores.append(np.mean(scores))
    return avg_scores

max_k = 20
avg_test_scores = evaluate_train_test(max_k)


# =========================================================
# 7. ĐÁNH GIÁ FULL DATASET
# =========================================================
tfidf_full = TfidfVectorizer(stop_words='english', max_features=5000)
tfidf_full_matrix = tfidf_full.fit_transform(df['combined_text'])

cosine_sim_full = cosine_similarity(tfidf_full_matrix, tfidf_full_matrix)

def evaluate_full_data(max_k=20):
    avg_scores = []
    for k in range(1, max_k + 1):
        scores = []
        for i in range(tfidf_full_matrix.shape[0]):
            sim_row = cosine_sim_full[i]
            topk = sim_row.argsort()[::-1][1:k+1]  # bỏ chính nó
            scores.extend(sim_row[topk])
        avg_scores.append(np.mean(scores))
    return avg_scores

avg_full_scores = evaluate_full_data(max_k)


# =========================================================
# 8A. Biểu đồ 1: FULL DATASET
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), avg_full_scores, marker='o')
plt.title("Average Cosine Similarity – FULL DATASET")
plt.xlabel("K")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.show()


# =========================================================
# 8B. Biểu đồ 2: TRAIN → TEST
# =========================================================
plt.figure(figsize=(8, 5))
plt.plot(range(1, max_k + 1), avg_test_scores, marker='s')
plt.title("Average Cosine Similarity – TRAIN → TEST")
plt.xlabel("K")
plt.ylabel("Cosine Similarity")
plt.grid(True)
plt.show()


# =========================================================
# 8C. Biểu đồ 3: SO SÁNH hai mô hình
# =========================================================
plt.figure(figsize=(10, 5))
plt.plot(range(1, max_k + 1), avg_full_scores, marker='o', label="Full Dataset (Trước chia)")
plt.plot(range(1, max_k + 1), avg_test_scores, marker='s', label="Train → Test (Sau chia)")

plt.title("SO SÁNH: Full Dataset vs Train/Test")
plt.xlabel("K")
plt.ylabel("Average Cosine Similarity")
plt.legend()
plt.grid(True)
plt.show()
# =========================================================
# 9. TÍNH CHỈ SỐ PRECISION@K, RECALL@K, F1@K VÀ ACCURACY@K
# =========================================================
def evaluate_recommendation_metrics(max_k=10):
    precisions = []
    recalls = []
    f1_scores = []
    accuracies = []

    test_categories = test_df['category_id'].values
    train_categories = train_df['category_id'].values

    for k in range(1, max_k + 1):
        p_list, r_list, f1_list, acc_list = [], [], [], []

        for i in range(len(test_df)):
            target_cat = test_categories[i]
            
            # Tổng số video trong tập Train thuộc cùng category
            total_relevant_in_train = np.sum(train_categories == target_cat)
            
            if total_relevant_in_train == 0:
                continue

            # Lấy Top-K chỉ số gợi ý tốt nhất
            sim_row = cosine_sim_test_train[i]
            topk_indices = sim_row.argsort()[::-1][:k]
            recommended_cats = train_categories[topk_indices]

            # Đếm số video gợi ý đúng Category (True Positives)
            hits = np.sum(recommended_cats == target_cat)

            # 1. Precision@K = Số video đúng trong Top K / K
            precision = hits / k
            
            # 2. Recall@K = Số video đúng / Tổng video cùng category trong Train
            recall = hits / total_relevant_in_train
            
            # 3. F1-Score@K
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            
            # 4. Accuracy@K = Tỷ lệ phần trăm đoán đúng
            accuracy = hits / k

            p_list.append(precision)
            r_list.append(recall)
            f1_list.append(f1)
            acc_list.append(accuracy)

        precisions.append(np.mean(p_list))
        recalls.append(np.mean(r_list))
        f1_scores.append(np.mean(f1_list))
        accuracies.append(np.mean(acc_list))

    return precisions, recalls, f1_scores, accuracies

# Chạy đánh giá Top 1 -> Top 10
max_k_eval = 10
p, r, f1, acc = evaluate_recommendation_metrics(max_k=max_k_eval)

print("\n================ BÁO CÁO KẾT QUẢ ĐÁNH GIÁ METRICS ================")
print(f"Top-K | Precision |  Recall  | F1-Score | Accuracy")
print("-" * 55)
for k in range(1, max_k_eval + 1):
    print(f"Top-{k:<2}|   {p[k-1]*100:.2f}%   |  {r[k-1]*100:.2f}%  |  {f1[k-1]*100:.2f}%   |  {acc[k-1]*100:.2f}%")

# =========================================================
# 10. BIỂU ĐỒ TRỰC QUAN HÓA PRECISION, RECALL & F1-SCORE
# =========================================================
plt.figure(figsize=(10, 6))
plt.plot(range(1, max_k_eval + 1), p, marker='o', label='Precision@K')
plt.plot(range(1, max_k_eval + 1), r, marker='s', label='Recall@K')
plt.plot(range(1, max_k_eval + 1), f1, marker='^', label='F1-Score@K')

plt.title("Recommendation System Evaluation Metrics across Top-K")
plt.xlabel("Top-K Value")
plt.ylabel("Score")
plt.xticks(range(1, max_k_eval + 1))
plt.legend()
plt.grid(True)
plt.show()
