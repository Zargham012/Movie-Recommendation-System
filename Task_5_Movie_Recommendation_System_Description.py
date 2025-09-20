import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD

# ------------- Load MovieLens data (local) -------------
def load_movielens(path="ml-100k"):
    ratings = pd.read_csv(
        os.path.join(path, "u.data"),
        sep="\t", names=["user_id", "item_id", "rating", "timestamp"]
    )
    items = pd.read_csv(
        os.path.join(path, "u.item"),
        sep="|", encoding="latin-1",
        names=["item_id", "title"], usecols=[0, 1]
    )
    return ratings, items

# ------------- Train-test split (leave-one-out) ---------------
def train_test_split(ratings):
    test = ratings.groupby("user_id").tail(1)
    train = ratings.drop(test.index)
    return train, test

# ---------------- Evaluation Metrics ----------------
def precision_at_k(recommended, relevant, k=5):
    recommended_at_k = recommended[:k]
    return len(set(recommended_at_k) & set(relevant)) / float(k)

def recall_at_k(recommended, relevant, k=5):
    recommended_at_k = recommended[:k]
    return len(set(recommended_at_k) & set(relevant)) / float(len(relevant))

def f1_at_k(recommended, relevant, k=5):
    p = precision_at_k(recommended, relevant, k)
    r = recall_at_k(recommended, relevant, k)
    if (p + r) == 0:
        return 0.0
    return 2 * (p * r) / (p + r)

# --------------- User-Item Matrix ----------------
def create_user_item_matrix(ratings, n_users, n_items):
    mat = np.zeros((n_users, n_items))
    for row in ratings.itertuples():
        mat[row.user_id-1, row.item_id-1] = row.rating
    return mat

# --------------- User-based Collaborative Filtering ----------------
def user_based_cf(user_item_matrix, metric="cosine"):
    if metric == "cosine":
        sim = cosine_similarity(user_item_matrix)
    elif metric == "pearson":
        sim = np.corrcoef(user_item_matrix)
        sim = np.nan_to_num(sim)
    return np.dot(sim, user_item_matrix)   # prediction scores

# ------------ Item-based Collaborative Filtering -----------------
def item_based_cf(user_item_matrix):
    sim = cosine_similarity(user_item_matrix.T)
    return np.dot(user_item_matrix, sim)   # prediction scores

# -------------- Matrix Factorization (SVD) ----------------
def svd_recommendations(user_item_matrix, n_factors=20):
    svd = TruncatedSVD(n_components=n_factors)
    latent_matrix = svd.fit_transform(user_item_matrix)
    return np.dot(latent_matrix, svd.components_)

# --------------- Recommendation helper ---------------
def recommend_for_user(user_id, scores, train, top_n=10):
    seen_items = train[train.user_id == user_id].item_id.values
    scores_user = scores[user_id-1]
    ranking = np.argsort(-scores_user)
    recommendations = [i+1 for i in ranking if i+1 not in seen_items][:top_n]
    return recommendations

# ------------ Evaluate all metrics ---------------
def evaluate_at_k(scores, train, test, k=5):
    precisions, recalls, f1s = [], [], []
    for user_id in test.user_id.unique():
        relevant = test[test.user_id == user_id].item_id.values
        recs = recommend_for_user(user_id, scores, train, top_n=k)
        precisions.append(precision_at_k(recs, relevant, k))
        recalls.append(recall_at_k(recs, relevant, k))
        f1s.append(f1_at_k(recs, relevant, k))
    return np.mean(precisions), np.mean(recalls), np.mean(f1s)

# -------------- Main ----------------
if __name__ == "__main__":
    ratings, items = load_movielens("ml-100k")
    train, test = train_test_split(ratings)

    n_users = ratings.user_id.nunique()
    n_items = ratings.item_id.nunique()
    user_item_matrix = create_user_item_matrix(train, n_users, n_items)

    # --- User-based CF ---
    user_scores = user_based_cf(user_item_matrix, "cosine")
    # --- Item-based CF ---
    item_scores = item_based_cf(user_item_matrix)
    # --- SVD ---
    svd_scores = svd_recommendations(user_item_matrix, n_factors=50)

    # ---- Evaluation ----
    for method_name, scores in {
        "User-based CF": user_scores,
        "Item-based CF": item_scores,
        "SVD": svd_scores
    }.items():
        print(f"\n{method_name} Results:")
        for k in [1, 5, 10]:
            p, r, f1 = evaluate_at_k(scores, train, test, k=k)
            print(f"@{k} -> Precision: {p:.4f}, Recall: {r:.4f}, F1: {f1:.4f}")

    # ---- Example Recommendations ----
    user_id = 1
    recs = recommend_for_user(user_id, svd_scores, train, top_n=10)
    rec_titles = items[items.item_id.isin(recs)]["title"].tolist()
    print(f"\nTop-10 Recommendations for User {user_id} (SVD):")
    for i, movie in enumerate(rec_titles, 1):
        print(f"{i}. {movie}")

# ------------- Collect results dynamically ---------------
results = {}

for method_name, scores in {
    "User-based CF": user_scores,
    "Item-based CF": item_scores,
    "SVD": svd_scores
}.items():
    results[method_name] = {}
    for k in [1, 5, 10]:
        p, r, f1 = evaluate_at_k(scores, train, test, k=k)
        results[method_name][f"@{k}"] = {"Precision": p, "Recall": r, "F1": f1}

# --------------- Plot comparison graphs ---------------
metrics = ["Precision", "Recall", "F1"]
ks = ["@1", "@5", "@10"]
colors = {"Precision": "blue", "Recall": "green", "F1": "red"}

for k in ks:
    plt.figure(figsize=(8,5))
    for metric in metrics:
        values = [results[method][k][metric] for method in results.keys()]
        plt.plot(list(results.keys()), values, marker="o", label=metric, color=colors[metric])
    
    plt.title(f"Comparison of Metrics at {k}")
    plt.xlabel("Method")
    plt.ylabel("Score")
    plt.ylim(0, 0.35)
    plt.legend()
    plt.grid(True)
    plt.show()