import os
from PIL import Image, ImageDraw, ImageFont
import textwrap

BASE = "/Users/gongxiyue/Downloads/STAT6207_assignments/Assignment 1"

# ---- Figure 1: Workflow Diagram ----
def create_workflow_diagram():
    W, H = 900, 200
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 16)
        font_small = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 13)
    except:
        font = ImageFont.load_default()
        font_small = font

    boxes = [
        ("Training\nImages", 30),
        ("Pretrained\nEncoder (fixed)", 210),
        ("Feature\nVectors", 400),
        ("KNN\n(k=5)", 570),
        ("Prediction", 740),
    ]

    for text, x in boxes:
        draw.rounded_rectangle([x, 40, x+140, 120], radius=8, fill="#E8F0FE", outline="#4285F4", width=2)
        bbox = draw.textbbox((0, 0), text, font=font, anchor="mm")
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text((x + 70, 80), text, fill="#1a1a1a", font=font, anchor="mm")

    for i in range(len(boxes) - 1):
        x1 = boxes[i][1] + 140
        x2 = boxes[i+1][1]
        draw.line([(x1, 80), (x2, 80)], fill="#4285F4", width=2)
        draw.polygon([(x2-8, 72), (x2, 80), (x2-8, 88)], fill="#4285F4")

    draw.text((W//2, 160), "Figure 1: Overall classification workflow", fill="#555", font=font_small, anchor="mm")
    img.save(os.path.join(BASE, "fig1_workflow.png"))
    return os.path.join(BASE, "fig1_workflow.png")

# ---- Figure 4: Keeshond Analysis ----
def create_keeshond_figure():
    # Load the keeshond image
    keeshond_img = Image.open(os.path.join(BASE, "stress_sample", "keeshond_117.jpg"))
    keeshond_img.thumbnail((220, 220))

    W, H = 900, 520
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)

    try:
        font_title = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 14)
        font_bold = ImageFont.truetype("/System/Library/Fonts/Helvetica-Bold.ttc", 13)
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except:
        font_title = font_bold = font = ImageFont.load_default()

    # Title
    draw.text((W//2, 18), "Figure 4: Difficult-Case Analysis — keeshond_117.jpg (dog)", fill="#1a1a1a", font=font_bold, anchor="mm")

    # Place keeshond image
    ix, iy = 30, 45
    img.paste(keeshond_img, (ix, iy))
    draw.text((ix + keeshond_img.width//2, iy + keeshond_img.height + 8), "keeshond_117.jpg", fill="#555", font=font, anchor="mm")

    # 3x3 Prediction Table
    tx, ty = 300, 45
    draw.text((tx, ty), "3 × 3 Prediction Table", fill="#1a1a1a", font=font_bold)
    ty += 22

    headers = ["Encoder", "Cosine", "Euclidean", "Manhattan"]
    col_w = [110, 80, 85, 85]
    row_h = 22

    # Header row
    x = tx
    for i, h in enumerate(headers):
        draw.rectangle([x, ty, x+col_w[i], ty+row_h], fill="#D2E3FC", outline="#999")
        draw.text((x+4, ty+4), h, fill="#1a1a1a", font=font_bold)
        x += col_w[i]

    # Data rows
    data = [
        ["ResNet-18",      "cat ✗",  "cat ✗",  "dog ✓"],
        ["EfficientNet-B0","cat ✗",  "cat ✗",  "dog ✓"],
        ["ViT-B/16",       "dog ✓",  "dog ✓",  "dog ✓"],
    ]
    for row in data:
        ty += row_h
        x = tx
        for i, cell in enumerate(row):
            draw.rectangle([x, ty, x+col_w[i], ty+row_h], outline="#ccc")
            color = "#1a1a1a"
            if "✗" in cell:
                color = "#c0392b"
            elif "✓" in cell:
                color = "#27ae60"
            draw.text((x+4, ty+4), cell, fill=color, font=font)
            x += col_w[i]

    # Top-5 Neighbours Table
    ty += row_h + 25
    draw.text((tx, ty), "Top-5 Nearest Neighbours (ResNet-18 + Cosine)", fill="#1a1a1a", font=font_bold)
    ty += 22

    headers2 = ["Rank", "Filename", "Label", "Cosine"]
    col_w2 = [40, 120, 60, 80]
    x = tx
    for i, h in enumerate(headers2):
        draw.rectangle([x, ty, x+col_w2[i], ty+row_h], fill="#D2E3FC", outline="#999")
        draw.text((x+4, ty+4), h, fill="#1a1a1a", font=font_bold)
        x += col_w2[i]

    neighbours = [
        ["1", "dog.39.jpg",  "dog", "0.7503"],
        ["2", "cat.139.jpg", "cat", "0.7441"],
        ["3", "cat.131.jpg", "cat", "0.7392"],
        ["4", "cat.36.jpg",  "cat", "0.7378"],
        ["5", "dog.188.jpg", "dog", "0.7374"],
    ]
    for row in neighbours:
        ty += row_h
        x = tx
        for i, cell in enumerate(row):
            draw.rectangle([x, ty, x+col_w2[i], ty+row_h], outline="#ccc")
            color = "#27ae60" if cell == "dog" else "#c0392b" if cell == "cat" else "#1a1a1a"
            draw.text((x+4, ty+4), cell, fill=color, font=font)
            x += col_w2[i]

    img.save(os.path.join(BASE, "fig4_keeshond.png"))
    return os.path.join(BASE, "fig4_keeshond.png")

# Run
f1 = create_workflow_diagram()
f4 = create_keeshond_figure()
print(f"Created: {f1}")
print(f"Created: {f4}")
