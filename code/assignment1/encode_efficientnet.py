import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report

# ---- Config ----
TRAIN_DIR = "dogs_vs_cats_subset/train"
TEST_DIR = "dogs_vs_cats_subset/test"
K = 5

# ---- Load pretrained EfficientNet-B0 and remove classifier ----
model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])  # output: (batch, 1280, 1, 1)
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ============================================================
# PART 1: Encode training images
# ============================================================
cat_files = sorted(
    [f for f in os.listdir(TRAIN_DIR) if f.startswith("cat.") and f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[1]),
)
dog_files = sorted(
    [f for f in os.listdir(TRAIN_DIR) if f.startswith("dog.") and f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[1]),
)
train_files = cat_files + dog_files
train_labels_list = [0] * len(cat_files) + [1] * len(dog_files)

print(f"Encoding {len(train_files)} training images with EfficientNet-B0...")
train_features = []
with torch.no_grad():
    for i, fname in enumerate(train_files):
        img = Image.open(os.path.join(TRAIN_DIR, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        feat = model(tensor).squeeze().numpy()
        train_features.append(feat)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(train_files)} done")

train_features = np.array(train_features)
train_labels = np.array(train_labels_list)
print(f"Training features shape: {train_features.shape}")

# Save
torch.save({
    "features": torch.tensor(train_features),
    "labels": torch.tensor(train_labels),
    "filenames": train_files,
}, "train_features_efficientnet_b0.pt")
print("Saved to train_features_efficientnet_b0.pt")

# L2-normalize
train_features_norm = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)

# ============================================================
# PART 2: 5-Fold Cross-Validation on training set
# ============================================================
print(f"\n{'='*60}")
print(f"5-Fold Cross-Validation (k={K}, cosine)")
print(f"{'='*60}")

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
fold_accs = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(train_features_norm, train_labels), 1):
    tr_X, tr_y = train_features_norm[tr_idx], train_labels[tr_idx]
    va_X, va_y = train_features_norm[va_idx], train_labels[va_idx]

    sim = va_X @ tr_X.T
    topk_idx = np.argpartition(-sim, K, axis=1)[:, :K]
    topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
    sort_order = np.argsort(-topk_sim, axis=1)
    topk_idx_sorted = np.take_along_axis(topk_idx, sort_order, axis=1)
    topk_labels = tr_y[topk_idx_sorted]
    preds = stats.mode(topk_labels, axis=1).mode.flatten()

    acc = accuracy_score(va_y, preds)
    fold_accs.append(acc)
    print(f"Fold {fold}: {len(tr_idx)} train / {len(va_idx)} val  |  Accuracy = {acc:.4f}")

print(f"{'='*60}")
print(f"Mean Accuracy: {np.mean(fold_accs):.4f} (+/- {np.std(fold_accs):.4f})")

# ============================================================
# PART 3: Predict on 10 unseen test images
# ============================================================
test_files = sorted(
    [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[1].split(".")[0]),
)
test_true = np.array([0]*5 + [1]*5)

print(f"\n{'='*60}")
print(f"Encoding 10 test images with EfficientNet-B0...")
test_features = []
with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        feat = model(tensor).squeeze().numpy()
        test_features.append(feat)

test_features = np.array(test_features)
test_features_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)

# KNN predict
sim = test_features_norm @ train_features_norm.T
topk_idx = np.argpartition(-sim, K, axis=1)[:, :K]
topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
sort_order = np.argsort(-topk_sim, axis=1)
topk_idx_sorted = np.take_along_axis(topk_idx, sort_order, axis=1)
topk_labels = train_labels[topk_idx_sorted]
test_preds = stats.mode(topk_labels, axis=1).mode.flatten()

print(f"\n{'='*60}")
print(f"KNN Predictions on 10 Unseen Test Images (k={K})")
print(f"{'='*60}")
print(f"{'Image':<14} {'True':<8} {'Predicted':<10} {'Correct?'}")
print("-" * 60)

correct = 0
for i, fname in enumerate(test_files):
    true_l = "cat" if test_true[i] == 0 else "dog"
    pred_l = "cat" if test_preds[i] == 0 else "dog"
    ok = test_true[i] == test_preds[i]
    correct += int(ok)
    print(f"{fname:<14} {true_l:<8} {pred_l:<10} {'✓' if ok else '✗'}")

print("-" * 60)
test_acc = correct / len(test_files)
print(f"Accuracy: {correct}/{len(test_files)} = {test_acc:.2%}")
print()
print(classification_report(test_true, test_preds, target_names=["cat", "dog"]))
