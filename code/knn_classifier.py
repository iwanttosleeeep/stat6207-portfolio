import torch
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ---- Load saved features ----
data = torch.load("train_features.pt", weights_only=False)
features = data["features"].numpy()   # (400, 512)
labels = data["labels"].numpy()       # (400,)
filenames = data["filenames"]

# L2-normalize features so dot product = cosine similarity
features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)

K = 5
N = len(labels)

def knn_predict(train_X, train_y, test_X, k):
    """Predict labels for test set using KNN with cosine similarity."""
    # Cosine similarity: (test @ train.T)
    sim = test_X @ train_X.T            # (n_test, n_train)
    # Top-k indices (highest similarity)
    topk_idx = np.argpartition(-sim, k, axis=1)[:, :k]
    # Gather actual similarity values for tie-breaking
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
    # Sort within top-k
    sort_order = np.argsort(-topk_sim, axis=1)
    topk_idx_sorted = np.take_along_axis(topk_idx, sort_order, axis=1)
    # Gather labels of top-k neighbors
    topk_labels = train_y[topk_idx_sorted]   # (n_test, k)
    # Majority vote
    from scipy import stats
    predictions = stats.mode(topk_labels, axis=1).mode.flatten()
    return predictions

# ---- 5-Fold Cross-Validation ----
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accuracies = []

print(f"KNN Classifier: k={K}, distance=cosine similarity")
print(f"Dataset: {N} images ({int((labels==0).sum())} cat, {int((labels==1).sum())} dog)")
print(f"{'='*50}")

for fold, (train_idx, val_idx) in enumerate(skf.split(features_norm, labels), 1):
    train_X, train_y = features_norm[train_idx], labels[train_idx]
    val_X, val_y = features_norm[val_idx], labels[val_idx]

    preds = knn_predict(train_X, train_y, val_X, K)
    acc = accuracy_score(val_y, preds)
    fold_accuracies.append(acc)
    print(f"Fold {fold}: {len(train_idx)} train / {len(val_idx)} val  |  Accuracy = {acc:.4f}")

mean_acc = np.mean(fold_accuracies)
std_acc = np.std(fold_accuracies)
print(f"{'='*50}")
print(f"Mean Accuracy: {mean_acc:.4f} (+/- {std_acc:.4f})")

# ---- Full evaluation (train on all, predict on all — for sanity check) ----
print(f"\n{'='*50}")
print("Full-data evaluation (train = predict = all 400):")
preds_all = knn_predict(features_norm, labels, features_norm, K)
full_acc = accuracy_score(labels, preds_all)
print(f"Accuracy: {full_acc:.4f}")
print()
print(classification_report(labels, preds_all, target_names=["cat", "dog"]))

# ---- Save model artifacts for later use ----
torch.save({
    "features_norm": torch.tensor(features_norm),
    "labels": torch.tensor(labels),
    "filenames": filenames,
    "k": K,
}, "knn_model.pt")
print("Saved KNN model artifacts to knn_model.pt")
