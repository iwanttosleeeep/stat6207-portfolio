import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

# ---- Config ----
TEST_DIR = "dogs_vs_cats_subset/test"
K = 5

# ---- Load KNN model artifacts ----
knn_data = torch.load("knn_model.pt", weights_only=False)
train_features_norm = knn_data["features_norm"].numpy()  # (400, 512)
train_labels = knn_data["labels"].numpy()                 # (400,)
train_filenames = knn_data["filenames"]

# ---- Load ResNet-18 feature extractor ----
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])
model.eval()

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- Extract features for test images ----
test_files = sorted(
    [f for f in os.listdir(TEST_DIR) if f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[1].split(".")[0]),
)
# Ground truth: test.1-5 = cat(0), test.6-10 = dog(1)
test_true = [0]*5 + [1]*5

test_features = []
with torch.no_grad():
    for fname in test_files:
        img = Image.open(os.path.join(TEST_DIR, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        feat = model(tensor).squeeze().numpy()
        test_features.append(feat)

test_features = np.array(test_features)                    # (10, 512)
test_features_norm = test_features / np.linalg.norm(test_features, axis=1, keepdims=True)
test_true = np.array(test_true)

# ---- KNN prediction ----
sim = test_features_norm @ train_features_norm.T           # (10, 400)
topk_idx = np.argpartition(-sim, K, axis=1)[:, :K]
topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
sort_order = np.argsort(-topk_sim, axis=1)
topk_idx_sorted = np.take_along_axis(topk_idx, sort_order, axis=1)
topk_labels = train_labels[topk_idx_sorted]
test_preds = stats.mode(topk_labels, axis=1).mode.flatten()

# ---- Print results ----
print("=" * 60)
print(f"KNN Predictions on 10 Unseen Test Images (k={K})")
print("=" * 60)
print(f"{'Image':<14} {'True':<8} {'Predicted':<10} {'Correct?'}")
print("-" * 60)

correct = 0
failed = []
for i, fname in enumerate(test_files):
    true_label = "cat" if test_true[i] == 0 else "dog"
    pred_label = "cat" if test_preds[i] == 0 else "dog"
    is_correct = test_true[i] == test_preds[i]
    correct += int(is_correct)
    status = "✓" if is_correct else "✗"
    print(f"{fname:<14} {true_label:<8} {pred_label:<10} {status}")
    if not is_correct:
        failed.append(i)

acc = correct / len(test_files)
print("-" * 60)
print(f"Accuracy: {correct}/{len(test_files)} = {acc:.2%}")

# ---- Visualize all 10 predictions ----
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
fig.suptitle(f"KNN Test Predictions (k={K}, cosine) — Accuracy: {acc:.0%}",
             fontsize=14, fontweight="bold")

for i, ax in enumerate(axes.flat):
    img = Image.open(os.path.join(TEST_DIR, test_files[i]))
    true_label = "cat" if test_true[i] == 0 else "dog"
    pred_label = "cat" if test_preds[i] == 0 else "dog"
    is_correct = test_true[i] == test_preds[i]

    ax.imshow(img)
    color = "green" if is_correct else "red"
    symbol = "✓" if is_correct else "✗"
    ax.set_title(f"{test_files[i]}\nTrue: {true_label} | Pred: {pred_label} {symbol}",
                 fontsize=9, color=color, fontweight="bold")
    ax.axis("off")

    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig("test_predictions.png", dpi=150, bbox_inches="tight")
plt.close()
print("Visualization saved to test_predictions.png")

# ---- Failed cases ----
if failed:
    print(f"\n{'='*60}")
    print(f"FAILED CASES ({len(failed)}):")
    print(f"{'='*60}")
    for i in failed:
        true_label = "cat" if test_true[i] == 0 else "dog"
        pred_label = "cat" if test_preds[i] == 0 else "dog"
        print(f"  {test_files[i]}: true={true_label}, predicted={pred_label}")
        # Show top-5 neighbors
        neighbors_idx = topk_idx_sorted[i]
        neighbors_sim = topk_sim[i][sort_order[i]]
        print(f"    Top-{K} neighbors:")
        for rank, (nidx, nsim) in enumerate(zip(neighbors_idx, neighbors_sim), 1):
            nlabel = "cat" if train_labels[nidx] == 0 else "dog"
            print(f"      {rank}. {train_filenames[nidx]:20s} ({nlabel}) cos={nsim:.4f}")
else:
    print("\nNo incorrect predictions!")
