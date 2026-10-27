from fpdf import FPDF
import os

BASE = "/Users/gongxiyue/Downloads/STAT6207_assignments/Assignment 1"

class ReportPDF(FPDF):
    def header(self):
        if self.page_no() > 1:
            self.set_font("Helvetica", "I", 8)
            self.set_text_color(120, 120, 120)
            self.cell(0, 8, "STAT6207 Assignment - Image Classification with Pretrained Encoders and KNN", align="C")
            self.ln(4)
            self.set_draw_color(200, 200, 200)
            self.line(10, self.get_y(), 200, self.get_y())
            self.ln(4)

    def footer(self):
        self.set_y(-15)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", align="C")

    def section_title(self, num, title):
        self.set_font("Helvetica", "B", 14)
        self.set_text_color(20, 60, 120)
        self.ln(4)
        self.cell(0, 10, f"{num}. {title}")
        self.ln(8)
        self.set_draw_color(20, 60, 120)
        self.line(10, self.get_y(), 200, self.get_y())
        self.ln(4)

    def sub_title(self, num, title):
        self.set_font("Helvetica", "B", 11)
        self.set_text_color(40, 40, 40)
        self.ln(2)
        self.cell(0, 8, f"{num} {title}")
        self.ln(8)

    def body_text(self, text):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def bold_text(self, text):
        self.set_font("Helvetica", "B", 10)
        self.set_text_color(30, 30, 30)
        self.multi_cell(0, 5.5, text)
        self.ln(1)

    def italic_text(self, text):
        self.set_font("Helvetica", "I", 10)
        self.set_text_color(60, 60, 60)
        self.multi_cell(0, 5.5, text)
        self.ln(2)

    def add_figure(self, img_path, caption, w=170):
        if not os.path.exists(img_path):
            self.body_text(f"[Image not found: {img_path}]")
            return
        img = Image.open(img_path)
        aspect = img.height / img.width
        h = w * aspect
        if h > 120:
            h = 120
            w = h / aspect
        x = (210 - w) / 2
        self.image(img_path, x=x, w=w)
        self.ln(2)
        self.set_font("Helvetica", "I", 9)
        self.set_text_color(80, 80, 80)
        self.multi_cell(0, 5, caption, align="C")
        self.ln(4)

    def add_table(self, headers, data, col_widths=None, caption=None):
        if col_widths is None:
            col_widths = [190 / len(headers)] * len(headers)

        # Caption
        if caption:
            self.set_font("Helvetica", "I", 9)
            self.set_text_color(80, 80, 80)
            self.multi_cell(0, 5, caption, align="C")
            self.ln(2)

        # Header
        self.set_font("Helvetica", "B", 9)
        self.set_fill_color(210, 227, 252)
        self.set_text_color(20, 20, 20)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, fill=True, align="C")
        self.ln()

        # Data
        self.set_font("Helvetica", "", 9)
        for row in data:
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 7, str(cell), border=1, align="C")
            self.ln()
        self.ln(4)

    def add_list(self, items):
        self.set_font("Helvetica", "", 10)
        self.set_text_color(30, 30, 30)
        for item in items:
            self.cell(5, 5.5, "")
            self.cell(5, 5.5, "-")
            self.multi_cell(0, 5.5, item)
            self.ln(1)
        self.ln(2)


from PIL import Image

pdf = ReportPDF()
pdf.alias_nb_pages()
pdf.set_auto_page_break(auto=True, margin=20)

# ============================================================
# TITLE PAGE
# ============================================================
pdf.add_page()
pdf.ln(50)
pdf.set_font("Helvetica", "B", 22)
pdf.set_text_color(20, 60, 120)
pdf.multi_cell(0, 12, "Image Classification Using\nPretrained Encoders and KNN:\nA Transfer Learning Approach", align="C")
pdf.ln(10)
pdf.set_font("Helvetica", "", 12)
pdf.set_text_color(80, 80, 80)
pdf.cell(0, 8, "STAT6207 Assignment", align="C")
pdf.ln(30)
pdf.set_font("Helvetica", "I", 10)
pdf.set_text_color(120, 120, 120)
pdf.cell(0, 8, "Report generated with OpenCode", align="C")

# ============================================================
# 1. INTRODUCTION
# ============================================================
pdf.add_page()
pdf.section_title("1", "Introduction / Objective")
pdf.body_text(
    "This assignment investigates image classification using pretrained convolutional neural networks "
    "and vision transformers as fixed feature extractors, combined with a K-nearest neighbours (KNN) classifier. "
    "Two questions are addressed:"
)
pdf.body_text(
    "1. Can pretrained models effectively encode cat and dog images for KNN-based classification?\n"
    "2. How do different encoder architectures and distance measures affect classification performance?"
)
pdf.body_text(
    "A companion coursework website presenting the Assignment 1 results is available at:\n"
    "https://iwanttosleeeep.github.io/stat6207-portfolio/"
)

# ============================================================
# 2. DATASET AND METHODOLOGY
# ============================================================
pdf.section_title("2", "Dataset and Methodology")

pdf.sub_title("2.1", "Main Assignment Dataset")
pdf.body_text(
    "The cat and dog images were sourced from the Dogs vs Cats subset on Zenodo "
    "(zenodo.org/records/10997322). From this source, a smaller dataset was prepared for the assignment:"
)
pdf.add_list([
    "Training set: 400 images - 200 cats (cat.1.jpg to cat.200.jpg) and 200 dogs (dog.1.jpg to dog.200.jpg).",
    "Test set: 10 unseen images - test.1.jpg to test.5.jpg (cats) and test.6.jpg to test.10.jpg (dogs).",
])

pdf.sub_title("2.2", "Additional Stress-Test Dataset")
pdf.body_text(
    "An additional 100-image stress-test set was created from the Oxford-IIIT Pet Dataset "
    "(robots.ox.ac.uk/~vgg/data/pets/). This dataset contains images from 37 distinct breeds "
    "(12 cat breeds, 25 dog breeds), providing a more challenging evaluation outside the original training "
    "distribution. The 100 images comprised 50 cats (4-5 images from each of 12 breeds) and 50 dogs "
    "(2 images from each of 25 breeds). Labels were determined from the breed names in the filenames."
)

pdf.sub_title("2.3", "Methodology")
pdf.body_text(
    "The classification pipeline followed the Pre-trained model -> encode -> classifier framework described "
    "in the STAT6207 lecture."
)

pdf.add_figure(os.path.join(BASE, "fig1_workflow.png"), "Figure 1: Overall classification workflow", w=160)

pdf.bold_text("Pretrained encoders as fixed feature extractors.")
pdf.body_text(
    "Each image was passed through a pretrained model with the final classification layer removed. "
    "The remaining layers acted as a fixed feature extractor, converting each image into a high-dimensional "
    "feature vector. No weights were updated during this process. As the lecture describes, the pre-trained "
    "layers act as a fixed feature map, and this representation-based transfer learning approach is optimal "
    "for very small target datasets (< 1,000 samples) - applicable here given the 400-image training set."
)

pdf.bold_text("KNN classifier.")
pdf.body_text(
    "KNN is a lazy learner: it builds no mathematical model during training; it simply memorizes the "
    "dataset and defers all computation to the moment a query is made. The stored training feature vectors "
    "and their labels were compared against each new query image's feature vector to find the k = 5 nearest "
    "neighbours, with the prediction determined by majority vote."
)

pdf.body_text("Three pretrained encoders were used:")
pdf.add_table(
    ["Encoder", "Architecture", "Feature Dimension"],
    [["ResNet-18", "CNN", "512"], ["EfficientNet-B0", "CNN", "1280"], ["ViT-B/16", "Vision Transformer", "768"]],
    col_widths=[60, 60, 70],
    caption="Table: Encoder architectures and feature dimensions"
)

pdf.body_text("Three distance measures were compared:")
pdf.add_table(
    ["Measure", "Definition (from lecture)"],
    [
        ["Cosine Similarity", "Measures directional alignment, ignoring magnitude"],
        ["Euclidean Distance", "Straight-line geometric distance (L2 norm)"],
        ["Manhattan Distance", "Sum of absolute coordinate differences (L1 norm)"],
    ],
    col_widths=[60, 130],
    caption="Table: Distance measures used in the comparison"
)

pdf.body_text(
    "All feature vectors were L2-normalized before applying Cosine and Euclidean distances. "
    "Preprocessing followed ImageNet standards: resize to 256, centre crop to 224, and normalize "
    "with mean [0.485, 0.456, 0.406] and standard deviation [0.229, 0.224, 0.225]."
)

# ============================================================
# 3. BASIC TASK RESULTS
# ============================================================
pdf.add_page()
pdf.section_title("3", "Basic Task Results")

pdf.sub_title("3.1", "ResNet-18 Feature Extraction")
pdf.body_text(
    "The 400 training images were encoded using ResNet-18 pretrained on ImageNet, producing a feature "
    "matrix of shape (400, 512). The features and labels were saved for subsequent use."
)

pdf.sub_title("3.2", "Similarity Retrieval")
pdf.body_text(
    "Using cat.1.jpg as the query, cosine similarity was computed against all 399 other training images. "
    "The results are shown in Figure 2."
)
pdf.add_figure(os.path.join(BASE, "website/assets/similarity_results.png"),
    "Figure 2: Similarity retrieval for cat.1.jpg. Top: the query image. "
    "Middle: top-5 most similar images (all cats, scores 0.8879-0.9055). "
    "Bottom: top-5 most dissimilar images (all dogs, scores 0.3509-0.4190).",
    w=170)

pdf.sub_title("3.3", "KNN Classification")
pdf.body_text(
    "A KNN classifier with k = 5 and cosine similarity was trained on the 400 encoded training images. "
    "Five-fold cross-validation yielded a mean accuracy of 98.25% (\u00b11.00%)."
)

pdf.sub_title("3.4", "Predictions on 10 Unseen Test Images")
pdf.body_text(
    "The classifier was applied to the 10 unseen test images. The results are shown in Figure 3."
)
pdf.add_figure(os.path.join(BASE, "test_predictions.png"),
    "Figure 3: Predictions on the 10 unseen test images (ResNet-18, cosine similarity, k=5). "
    "All 10 images were correctly classified: test.1-test.5 as cats, test.6-test.10 as dogs (100% accuracy).",
    w=170)

# ============================================================
# 4. ADVANCED COMPARISON
# ============================================================
pdf.add_page()
pdf.section_title("4", "Advanced Comparison")

pdf.sub_title("4.1", "Three Encoders x Three Distance Measures")
pdf.body_text(
    "All nine combinations of encoder and distance measure were evaluated using 5-fold cross-validation "
    "on the 400 training images and tested on the 10 unseen images. The results are summarised in Table 1."
)

pdf.add_table(
    ["Encoder", "Cosine", "Euclidean", "Manhattan"],
    [
        ["ResNet-18", "98.25", "98.25", "98.50"],
        ["EfficientNet-B0", "98.25", "98.25", "98.75"],
        ["ViT-B/16", "99.25", "99.25", "99.25"],
    ],
    col_widths=[55, 40, 45, 45],
    caption="Table 1: Mean 5-fold CV accuracy (%) across encoders and distance measures. All nine combinations achieved 100% on the 10 unseen test images."
)

pdf.sub_title("4.2", "Observations")
pdf.bold_text("Cosine and Euclidean produced identical results.")
pdf.body_text(
    "Wherever both were compared, Cosine similarity and Euclidean distance produced identical nearest-neighbour "
    "rankings. This is because, for L2-normalized vectors, Euclidean distance is a monotonic transformation "
    "of cosine similarity (d^2 = 2 - 2cos(theta)), so the nearest-neighbour rankings are the same."
)

pdf.bold_text("Manhattan distance produced different nearest-neighbour rankings.")
pdf.body_text(
    "For ResNet-18, CV accuracy increased from 98.25% to 98.50%; for EfficientNet-B0, from 98.25% to "
    "98.75%. For ViT-B/16, accuracy remained at 99.25%. These differences were small and within the "
    "standard deviation, so they should be interpreted with caution."
)

pdf.bold_text("ViT-B/16 achieved the highest CV accuracy (99.25%)")
pdf.body_text(
    "across all three distance measures, with the lowest standard deviation (\u00b10.61%)."
)

# ============================================================
# 5. DIFFICULT CASE ANALYSIS
# ============================================================
pdf.add_page()
pdf.section_title("5", "Difficult Case Analysis")

pdf.sub_title("5.1", "The keeshond_117.jpg Case")
pdf.body_text(
    "The keeshond_117.jpg image (a Keeshond dog) was the only misclassification in the stress test when "
    "using ResNet-18 with cosine similarity. To investigate, the image was tested with all nine "
    "encoder-distance combinations."
)

pdf.add_figure(os.path.join(BASE, "fig4_keeshond.png"),
    "Figure 4: Difficult-case analysis. Left: the Keeshond image. Right top: 3 x 3 prediction table across all "
    "encoder-distance combinations. Right bottom: top-5 nearest neighbours for ResNet-18 + Cosine.",
    w=170)

pdf.body_text(
    "With cosine similarity, three of the five nearest neighbours were cats, resulting in a "
    "misclassification. The lecture states: a KNN algorithm is dependent on how closeness is defined. "
    "Changing the mathematical rules for measuring space fundamentally alters which neighbors are "
    "considered closest. When Manhattan distance was applied to the same ResNet-18 features, the nearest "
    "neighbours shifted, and the prediction changed to dog. Similarly, when ViT-B/16 features were used "
    "(regardless of distance measure), the prediction was correct."
)
pdf.body_text(
    "This single case illustrates that both the choice of encoder (feature space) and the choice of "
    "distance measure can affect KNN predictions on individual images. However, this observation is based "
    "on one image and should not be generalised."
)

# ============================================================
# 6. ADDITIONAL STRESS TEST
# ============================================================
pdf.section_title("6", "Additional Stress Test")
pdf.body_text(
    "The 100-image stress-test set (from the Oxford-IIIT Pet Dataset, 37 breeds) was evaluated using "
    "ResNet-18 with cosine similarity and k = 5 only. This combination was chosen as the baseline "
    "configuration from the basic task."
)
pdf.bold_text("Overall accuracy: 99/100 = 99.00%")
pdf.add_list([
    "Cat images: 50/50 correctly classified (100%)",
    "Dog images: 49/50 correctly classified (98%)",
    "Keeshond: 50% (1/2 correct) - all other breeds: 100%",
])
pdf.body_text(
    "The single misclassification was keeshond_117.jpg, classified as cat. This is discussed in Section 5."
)
pdf.italic_text(
    "Note: The stress test was run with only one encoder-distance combination. The performance of "
    "EfficientNet-B0, ViT-B/16, and Manhattan distance on this stress-test set was not evaluated, "
    "and no comparisons can be drawn."
)

# ============================================================
# 7. DISCUSSION
# ============================================================
pdf.add_page()
pdf.section_title("7", "Discussion")

pdf.sub_title("7.1", "Connection to STAT6207 Lecture Concepts")

pdf.bold_text("Representation-based transfer learning.")
pdf.body_text(
    "The assignment applied the lecture framework of using pretrained models as fixed feature maps. "
    "Given the small dataset (400 images), training a model from scratch would have been less practical "
    "for this assignment. The pretrained encoders leveraged representations learned from ImageNet, and "
    "the 400 training images were used only to configure the KNN classifier."
)

pdf.bold_text("Feature space and nearest-neighbour structure.")
pdf.body_text(
    "The lecture describes the pipeline: Pre-trained model -> representation / fixed feature map -> new "
    "feature space -> downstream model. Each encoder produced a different feature space (512, 1280, or 768 "
    "dimensions). The downstream model (KNN) was identical in all cases; differences in results were "
    "attributable to the different feature spaces. This was observed both in the CV accuracy differences "
    "(Table 1) and in the keeshond_117.jpg case, where ViT-B/16's feature space placed the image closer "
    "to dogs while ResNet-18's did not."
)

pdf.bold_text("Distance measures and closeness.")
pdf.body_text(
    "The lecture states: changing the mathematical rules for measuring space fundamentally alters which "
    "neighbors are considered closest. Wherever both were compared on normalized vectors, Cosine and "
    "Euclidean produced identical rankings. Manhattan distance produced different rankings, which "
    "affected the prediction on keeshond_117.jpg for two of the three encoders."
)

pdf.sub_title("7.2", "Strengths and Weaknesses (Based on Measured Results)")
pdf.add_table(
    ["Model / Metric", "Evidence from our experiments"],
    [
        ["ResNet-18", "98.25% CV, 100% test. Smallest features (512). Misclassified keeshond_117.jpg with Cosine/Euclidean."],
        ["EfficientNet-B0", "98.25% CV, 100% test. Largest features (1280). Same misclassification pattern as ResNet-18."],
        ["ViT-B/16", "99.25% CV (highest), 100% test. Correct on keeshond_117.jpg with all 3 distances."],
        ["Cosine / Euclidean", "Identical results wherever both were compared. Interchangeable on normalized vectors."],
        ["Manhattan", "Different rankings. Fixed keeshond_117.jpg for 2/3 encoders. Small CV gains (within std)."],
    ],
    col_widths=[50, 140],
    caption="Table 2: Summary of strengths and weaknesses based on measured results"
)

pdf.sub_title("7.3", "Decisions Made While Using OpenCode")
pdf.body_text(
    "Before starting the implementation, I first used ChatGPT to break down the assignment requirements "
    "and identify which experimental choices needed to be made: "
    "https://chatgpt.com/share/6aaf8f57-437c-83ea-b39e-5d001bf8efa5"
)
pdf.body_text(
    "I then worked through the Basic Tasks step by step with OpenCode. For choices that were not specified "
    "by the assignment, such as the initial pretrained model and the value of k for KNN, I asked OpenCode "
    "for suggestions, reviewed them, and adopted ResNet-18 and k = 5 as the initial configuration."
)
pdf.body_text(
    "For the Advanced Tasks, I again asked OpenCode for suitable pretrained models to compare and used "
    "ResNet-18, EfficientNet-B0, and ViT-B/16. The three distance measures were selected based on the "
    "STAT6207 lecture content. Since all methods correctly classified the original 10 unseen test images, "
    "I decided to add an additional stress test using images from the Oxford-IIIT Pet Dataset to examine "
    "the method on a more varied set of images. OpenCode assisted with preparing the sample and running "
    "the experiment."
)
pdf.body_text(
    "After obtaining the experimental results, I first asked OpenCode to generate a draft report. I then "
    "reviewed the STAT6207 lecture notes myself to identify which course concepts could appropriately "
    "explain the methodology and findings, including representation-based transfer learning, fixed feature "
    "maps, KNN as a lazy learner, and the effect of different definitions of closeness. I provided these "
    "lecture concepts back to OpenCode and asked it to revise the report so that the discussion was "
    "grounded in the terminology and ideas covered in the course."
)
pdf.body_text(
    "Finally, I reviewed OpenCode's interpretation of the experimental results rather than using its "
    "conclusions directly. Some initial statements, such as claims about runtime, robustness, and "
    "stress-test performance for models that had not actually been evaluated, went beyond the available "
    "evidence. I asked OpenCode to review these claims again and revised the report so that the "
    "conclusions were limited to results that were actually measured."
)
pdf.body_text(
    "When creating the webpage, I chose a portfolio-style structure rather than a one-off assignment page, "
    "so that Assignment 1 could be presented as the first project and future STAT6207 coursework could be "
    "added later. OpenCode assisted with implementing the static HTML/CSS layout and presenting the "
    "existing results."
)

pdf.sub_title("7.4", "Limitations")
pdf.add_list([
    "The 10-image test set is too small to draw statistically meaningful conclusions. All nine combinations achieved 100%, providing no differentiation.",
    "The 100-image stress test was only run with one combination (ResNet-18 + Cosine). No comparisons across encoders or distances are possible on this set.",
    "keeshond_117.jpg is a single example. The observation that Manhattan distance and ViT-B/16 handled this case correctly cannot be generalised to claim broader robustness or superiority.",
    "CV accuracy differences between distance measures were small (within \u00b11-2 standard deviations) and should not be interpreted as definitive.",
    "No runtime or efficiency measurements were performed. Claims about speed based on feature dimensions alone are theoretical, not empirical.",
    "The dataset is relatively clean and balanced. Performance on noisier, larger, or more imbalanced datasets may differ.",
])

# ============================================================
# 8. CONCLUSION
# ============================================================
pdf.section_title("8", "Conclusion")
pdf.body_text(
    "This assignment demonstrated that pretrained models can effectively encode images for KNN-based "
    "classification, even with a small training set of 400 images. The Pre-trained model -> encode -> "
    "classifier pipeline, grounded in representation-based transfer learning, achieved strong results "
    "across all evaluated configurations."
)
pdf.body_text(
    "Among the three encoders, ViT-B/16 achieved the highest 5-fold CV accuracy (99.25%) and correctly "
    "handled the one edge case that tripped up the CNN-based encoders. Wherever both were compared, "
    "Cosine similarity and Euclidean distance produced identical results on normalized features, while "
    "Manhattan distance produced different nearest-neighbour rankings, which affected predictions on a "
    "single difficult image."
)
pdf.body_text(
    "The main limitations are the small test set size, the single-configuration stress test, and the "
    "single-image edge case. Stronger conclusions would require larger and more diverse evaluation sets, "
    "and the performance of all encoder-distance combinations on the stress-test set remains to be "
    "evaluated."
)

# Save
output_path = os.path.join(BASE, "report.pdf")
pdf.output(output_path)
print(f"PDF saved to: {output_path}")
