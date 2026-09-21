import torch
import matplotlib.pyplot as plt
from PIL import Image

# ---- Load saved features ----
data = torch.load("train_features.pt", weights_only=False)
features = data["features"]      # (400, 512)
labels = data["labels"]          # (400,)
filenames = data["filenames"]    # list of 400 strings

# ---- Query image ----
query_idx = 0  # cat.1.jpg is the first entry
query_feat = features[query_idx].unsqueeze(0)  # (1, 512)

# ---- Compute cosine similarity against all others ----
# L2-normalize then dot product = cosine similarity
query_norm = torch.nn.functional.normalize(query_feat, p=2, dim=1)
all_norm = torch.nn.functional.normalize(features, p=2, dim=1)
similarities = (all_norm @ query_norm.T).squeeze()  # (400,)

# ---- Exclude the query itself ----
mask = torch.ones(len(filenames), dtype=bool)
mask[query_idx] = False
sims_masked = similarities[mask]
idxs_masked = torch.arange(len(filenames))[mask]

# ---- Top 5 most similar (highest score) and most dissimilar (lowest score) ----
top5_sim = sims_masked.topk(5)
top5_dis = sims_masked.topk(5, largest=False)

sim_indices = idxs_masked[top5_sim.indices]
sim_scores = top5_sim.values
dis_indices = idxs_masked[top5_dis.indices]
dis_scores = top5_dis.values

# ---- Print results ----
print("=" * 50)
print(f"Query: cat.1.jpg")
print("=" * 50)
print("\nTop 5 MOST SIMILAR:")
for rank, (idx, score) in enumerate(zip(sim_indices, sim_scores), 1):
    print(f"  {rank}. {filenames[idx]:20s}  cosine = {score:.4f}")

print("\nTop 5 MOST DISSIMILAR:")
for rank, (idx, score) in enumerate(zip(dis_indices, dis_scores), 1):
    print(f"  {rank}. {filenames[idx]:20s}  cosine = {score:.4f}")

# ---- Visualize ----
DATA_DIR = "dogs_vs_cats_subset/train"
all_indices = list(sim_indices) + list(dis_indices)
all_scores = list(sim_scores) + list(dis_scores)
titles = ["Similar" if i < 5 else "Dissimilar" for i in range(10)]

fig, axes = plt.subplots(2, 5, figsize=(18, 7))

# Add query image
query_img = Image.open(f"{DATA_DIR}/cat.1.jpg")
fig.suptitle("Query: cat.1.jpg", fontsize=16, fontweight="bold", y=1.02)

for i, ax in enumerate(axes.flat):
    idx = all_indices[i]
    score = all_scores[i]
    img = Image.open(f"{DATA_DIR}/{filenames[idx]}")
    ax.imshow(img)
    label = "cat" if labels[idx] == 0 else "dog"
    ax.set_title(f"{filenames[idx]}\n{label} | cos={score:.4f}", fontsize=9)
    ax.axis("off")

    # Color border: green for similar, red for dissimilar
    color = "#2ecc71" if i < 5 else "#e74c3c"
    for spine in ax.spines.values():
        spine.set_edgecolor(color)
        spine.set_linewidth(3)
        spine.set_visible(True)

plt.tight_layout()
plt.savefig("similarity_results.png", dpi=150, bbox_inches="tight")
plt.show()
print("\nVisualization saved to similarity_results.png")
