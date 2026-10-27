# STAT6207 Coursework Portfolio

Static coursework website, extending the original Assignment 1 layout.

## Repository layout

```text
index.html                 Portfolio homepage
assignment1.html           Image encoders and KNN
assignment2.html           Time-series forecasting
assignment3.html           FaceNet and CNN comparison
style.css                  Shared page styles
assets/assignment1/        Assignment 1 presentation images
assets/assignment2/        Assignment 2 presentation figures
assets/assignment3/        Assignment 3 presentation figures
code/assignment1/          Assignment 1 original Python scripts
code/assignment2/          Sessions 1–3 and Transformer scripts
code/assignment3/          Dataset preparation and experiment scripts
results/                   Published aggregate result evidence
```

Assignment 1's former `code/*.py` and `assets/*` files have been moved into their
assignment-specific folders without changing their contents. All website image
references use the new locations; the public `assignment1.html` URL is unchanged.

## Assignment pages

- [Portfolio](https://iwanttosleeeep.github.io/stat6207-portfolio/)
- [Assignment 1](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment1.html): pretrained encoders and KNN.
- [Assignment 2](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment2.html): LSTM and Transformer forecasting; repeated-run and generated-code audits.
- [Assignment 3](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment3.html): FaceNet freezing strategies, CNN comparison and inference audit.

## Content and provenance

Assignment 1 retains the original published results and analysis, with updated presentation and resource paths. Assignment 2 follows the audited final assignment report and Session 3 result file. Assignment 3 follows the final report and saved validation/inference summaries. The website describes the original assignment experiments, not later environment-verification reruns. Assignment 3 external images have no ground-truth labels; prediction distributions must not be interpreted as accuracy.

`code/assignment1/`, `code/assignment2/` and `code/assignment3/` contain original Python source snapshots. Running them requires their original data, directory layout and Python dependencies; model weights and full datasets are not distributed here. Some Assignment 1 scripts retain historical machine-specific paths; see [the Assignment 1 code guide](code/assignment1/README.md) before running them. The webpages themselves need no Python environment or build step. `results/` contains compact aggregate evidence, and `assets/` contains presentation figures and selected examples.

## Local preview

From this directory:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/. The same HTML/CSS and assets are served on GitHub Pages.
