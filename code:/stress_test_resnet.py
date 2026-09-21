import os
import torch
import torch.nn as nn
import numpy as np
from torchvision import models, transforms
from PIL import Image
from scipy import stats
from sklearn.metrics import accuracy_score, classification_report

# ---- Config ----
STRESS_DIR = "stress_sample"
K = 5

CAT_BREEDS = {
    'Abyssinian','Bengal','Birman','Bombay','British_Shorthair',
    'Egyptian_Mau','Maine_Coon','Persian','Ragdoll','Russian_Blue',
    'Siamese','Sphynx'
}

# ---- Load saved ResNet-18 training features ----
data = torch.load("train_features.pt", weights_only=False)
train_features = data["features"].numpy()
train_labels = data["labels"].numpy()
train_features_norm = train_features / np.linalg.norm(train_features, axis=1, keepdims=True)

# ---- Load ResNet-18 for encoding stress images ----
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

# ---- Determine label from filename ----
def get_breed_and_label(filename):
    breed = filename.rsplit('_', 1)[0]
    label = 0 if breed in CAT_BREEDS else 1  # 0=cat, 1=dog
    return breed, label

# ---- Load stress sample ----
stress_files = sorted([f for f in os.listdir(STRESS_DIR) if f.endswith('.jpg')])
stress_true = []
breeds = []
for f in stress_files:
    breed, label = get_breed_and_label(f)
    stress_true.append(label)
    breeds.append(breed)
stress_true = np.array(stress_true)

print(f"Stress test: {len(stress_files)} images ({int((stress_true==0).sum())} cat, {int((stress_true==1).sum())} dog)")
print(f"KNN: k={K}, distance=cosine, encoder=ResNet-18")
print(f"{'='*60}")

# ---- Encode stress images ----
stress_features = []
with torch.no_grad():
    for fname in stress_files:
        img = Image.open(os.path.join(STRESS_DIR, fname)).convert("RGB")
        tensor = transform(img).unsqueeze(0)
        feat = model(tensor).squeeze().numpy()
        stress_features.append(feat)

stress_features = np.array(stress_features)
stress_features_norm = stress_features / np.linalg.norm(stress_features, axis=1, keepdims=True)

# ---- KNN predict ----
sim = stress_features_norm @ train_features_norm.T
topk_idx = np.argpartition(-sim, K, axis=1)[:, :K]
topk_sim = np.take_along_axis(sim, topk_idx, axis=1)
sort_order = np.argsort(-topk_sim, axis=1)
topk_idx_sorted = np.take_along_axis(topk_idx, sort_order, axis=1)
topk_labels = train_labels[topk_idx_sorted]
test_preds = stats.mode(topk_labels, axis=1).mode.flatten()

# ---- Results ----
correct = test_preds == stress_true
acc = accuracy_score(stress_true, test_preds)

print(f"\nOverall Accuracy: {int(correct.sum())}/{len(stress_files)} = {acc:.2%}")
print()
print(classification_report(stress_true, test_preds, target_names=["cat", "dog"]))

# ---- Failed cases ----
failed_idx = np.where(~correct)[0]
if len(failed_idx) > 0:
    print(f"{'='*60}")
    print(f"INCORRECT PREDICTIONS ({len(failed_idx)}/{len(stress_files)}):")
    print(f"{'='*60}")
    for i in failed_idx:
        true_l = "cat" if stress_true[i] == 0 else "dog"
        pred_l = "cat" if test_preds[i] == 0 else "dog"
        print(f"  {stress_files[i]:40s} breed={breeds[i]:22s} true={true_l} pred={pred_l}")
else:
    print("No incorrect predictions!")

# ---- Per-breed accuracy ----
print(f"\n{'='*60}")
print("PER-BREED ACCURACY:")
print(f"{'='*60}")
from collections import defaultdict
breed_results = defaultdict(lambda: {"correct": 0, "total": 0})
for i in range(len(stress_files)):
    b = breeds[i]
    breed_results[b]["total"] += 1
    if correct[i]:
        breed_results[b]["correct"] += 1

for breed in sorted(breed_results.keys()):
    r = breed_results[breed]
    c = "cat" if breed in CAT_BREEDS else "dog"
    print(f"  {breed:28s} ({c:3s}) {r['correct']}/{r['total']} = {r['correct']/r['total']:.0%}")
