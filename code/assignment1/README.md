# Assignment 1 source guide

[View the results page](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment1.html)

These are the original Assignment 1 Python scripts, relocated from the repository's
top-level `code/` directory without modifying their contents or experimental logic.

| Stage | Scripts |
| --- | --- |
| Feature extraction | `encode_images.py`, `encode_efficientnet.py`, `encode_vit.py` |
| KNN evaluation and distance comparisons | `knn_classifier.py`, `knn_euclidean.py`, `knn_euclidean_effnet.py`, `knn_euclidean_vit.py`, `knn_manhattan_resnet.py`, `knn_manhattan_effnet.py`, `knn_manhattan_vit.py` |
| Test predictions and stress testing | `predict_test.py`, `stress_test_resnet.py` |
| Similarity and presentation figures | `visualize_similarity.py`, `revise_similarity.py`, `create_figures.py` |
| Report generation | `generate_report.py` |

## Running the historical experiments

The source snapshot is not a self-contained runnable package. The original image
datasets, generated feature files and Python dependencies are required. Inspect
each script's input/output paths before running: several retain absolute paths
to the original author's assignment directory, which must be adapted to your
local data location. This repository reorganisation does not change those paths
or regenerate any experiment results.

The already-published presentation images are stored in `assets/assignment1/` at
the repository root. The website itself is static HTML/CSS and does not execute
these Python scripts.
