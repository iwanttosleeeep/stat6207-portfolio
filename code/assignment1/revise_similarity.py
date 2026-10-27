import os
from PIL import Image, ImageDraw, ImageFont

BASE = "/Users/gongxiyue/Downloads/STAT6207_assignments/Assignment 1"
TRAIN_DIR = os.path.join(BASE, "dogs_vs_cats_subset/train")

# Data from previous results
similar_files = ["cat.197.jpg", "cat.103.jpg", "cat.82.jpg", "cat.104.jpg", "cat.43.jpg"]
similar_scores = [0.9055, 0.9048, 0.8935, 0.8893, 0.8879]
dissimilar_files = ["dog.172.jpg", "dog.162.jpg", "dog.31.jpg", "dog.130.jpg", "dog.176.jpg"]
dissimilar_scores = [0.3509, 0.3930, 0.3933, 0.4106, 0.4190]

# Load fonts
try:
    font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    font_label = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 11)
    font_score = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 10)
except:
    font_title = font_label = font_score = ImageFont.load_default()

# Load and resize images
def load_img(path, size=(140, 140)):
    img = Image.open(path).convert("RGB")
    img.thumbnail(size, Image.LANCZOS)
    return img

query_img = load_img(os.path.join(TRAIN_DIR, "cat.1.jpg"), (160, 160))
sim_imgs = [load_img(os.path.join(TRAIN_DIR, f)) for f in similar_files]
dis_imgs = [load_img(os.path.join(TRAIN_DIR, f)) for f in dissimilar_files]

# Layout
MARGIN = 20
GAP = 15
IMG_W = 140
THUMB_H = 140
QUERY_SIZE = 160

# Calculate canvas size
total_w = MARGIN * 2 + 5 * IMG_W + 4 * GAP
query_section_h = QUERY_SIZE + 60  # query image + label
row_section_h = THUMB_H + 55  # image + labels
section_gap = 30

total_h = MARGIN + 30 + query_section_h + 20 + 15 + row_section_h + section_gap + 15 + row_section_h + MARGIN + 10

# Create canvas
canvas = Image.new("RGB", (total_w, total_h), "white")
draw = ImageDraw.Draw(canvas)

# Title
y = MARGIN
draw.text((total_w // 2, y), "Similarity Retrieval: cat.1.jpg Query", fill="#1a1a1a", font=font_title, anchor="mt")
y += 30

# Query image section
draw.text((total_w // 2, y), "Query Image", fill="#666", font=font_label, anchor="mt")
y += 18
qx = (total_w - QUERY_SIZE) // 2
# Border around query
draw.rectangle([qx - 3, y - 3, qx + QUERY_SIZE + 3, y + QUERY_SIZE + 3], outline="#4285F4", width=3)
canvas.paste(query_img, (qx, y))
draw.text((total_w // 2, y + QUERY_SIZE + 6), "cat.1.jpg (cat)", fill="#1a1a1a", font=font_label, anchor="mt")
y += QUERY_SIZE + 25

# Divider
draw.line([(MARGIN, y), (total_w - MARGIN, y)], fill="#ccc", width=1)
y += 12

# Top-5 Similar
draw.text((total_w // 2, y), "Top 5 Most Similar Images", fill="#27ae60", font=font_label, anchor="mt")
y += 15

start_x = MARGIN
for i, (img, score) in enumerate(zip(sim_imgs, similar_scores)):
    x = start_x + i * (IMG_W + GAP)
    # Center image in cell
    ix = x + (IMG_W - img.width) // 2
    canvas.paste(img, (ix, y))
    # Label
    label = similar_files[i].replace(".jpg", "")
    draw.text((x + IMG_W // 2, y + img.height + 4), label, fill="#1a1a1a", font=font_label, anchor="mt")
    draw.text((x + IMG_W // 2, y + img.height + 17), f"cat | cos={score:.4f}", fill="#27ae60", font=font_score, anchor="mt")

y += THUMB_H + 40

# Divider
draw.line([(MARGIN, y), (total_w - MARGIN, y)], fill="#ccc", width=1)
y += 12

# Top-5 Dissimilar
draw.text((total_w // 2, y), "Top 5 Most Dissimilar Images", fill="#c0392b", font=font_label, anchor="mt")
y += 15

for i, (img, score) in enumerate(zip(dis_imgs, dissimilar_scores)):
    x = start_x + i * (IMG_W + GAP)
    ix = x + (IMG_W - img.width) // 2
    canvas.paste(img, (ix, y))
    label = dissimilar_files[i].replace(".jpg", "")
    draw.text((x + IMG_W // 2, y + img.height + 4), label, fill="#1a1a1a", font=font_label, anchor="mt")
    draw.text((x + IMG_W // 2, y + img.height + 17), f"dog | cos={score:.4f}", fill="#c0392b", font=font_score, anchor="mt")

# Save
out_path = os.path.join(BASE, "website/assets/similarity_results.png")
os.makedirs(os.path.dirname(out_path), exist_ok=True)
canvas.save(out_path, quality=95)
print(f"Saved revised similarity figure to {out_path}")
print(f"Canvas size: {canvas.size}")
