import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image

# ---- Config ----
DATA_DIR = "dogs_vs_cats_subset/train"
OUTPUT_FILE = "train_features.pt"

# ---- Load pretrained ResNet-18 and remove the final FC layer ----
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model = nn.Sequential(*list(model.children())[:-1])  # output: (batch, 512, 1, 1)
model.eval()

# ---- Preprocessing (ImageNet standard) ----
transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---- Collect files sorted by numeric ID ----
cat_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.startswith("cat.") and f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[1]),
)
dog_files = sorted(
    [f for f in os.listdir(DATA_DIR) if f.startswith("dog.") and f.endswith(".jpg")],
    key=lambda x: int(x.split(".")[1]),
)
all_files = cat_files + dog_files
labels = [0] * len(cat_files) + [1] * len(dog_files)  # 0=cat, 1=dog

print(f"Found {len(cat_files)} cat + {len(dog_files)} dog = {len(all_files)} images")

# ---- Extract features ----
features_list = []

with torch.no_grad():
    for i, fname in enumerate(all_files):
        img_path = os.path.join(DATA_DIR, fname)
        img = Image.open(img_path).convert("RGB")
        tensor = transform(img).unsqueeze(0)  # (1, 3, 224, 224)
        feat = model(tensor).squeeze()         # (512,)
        features_list.append(feat)

        if (i + 1) % 50 == 0 or (i + 1) == len(all_files):
            print(f"  Processed {i + 1}/{len(all_files)}: {fname}")

# ---- Stack and save ----
features = torch.stack(features_list)   # (400, 512)
labels_tensor = torch.tensor(labels, dtype=torch.long)

torch.save({
    "features": features,
    "labels": labels_tensor,
    "filenames": all_files,
}, OUTPUT_FILE)

print(f"\nSaved features {features.shape} and labels {labels_tensor.shape} to {OUTPUT_FILE}")
