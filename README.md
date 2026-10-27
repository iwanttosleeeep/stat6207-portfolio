# STAT6207 Coursework Portfolio

Static coursework website, extending the original Assignment 1 layout.

- [Portfolio](https://iwanttosleeeep.github.io/stat6207-portfolio/)
- [Assignment 1](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment1.html): pretrained encoders and KNN.
- [Assignment 2](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment2.html): LSTM and Transformer forecasting; repeated-run and generated-code audits.
- [Assignment 3](https://iwanttosleeeep.github.io/stat6207-portfolio/assignment3.html): FaceNet freezing strategies, CNN comparison and inference audit.

## Content and provenance

Assignment 2 follows the audited final assignment report and Session 3 result file. Assignment 3 follows the final report and saved validation/inference summaries. The website describes the original assignment experiments, not later environment-verification reruns. Assignment 3 external images have no ground-truth labels; prediction distributions must not be interpreted as accuracy.

`code/assignment2/` and `code/assignment3/` contain original Python source snapshots. Running them requires their original data, directory layout and Python dependencies; model weights and datasets are not distributed here. The webpages themselves need no Python environment or build step. `results/` contains compact aggregate evidence, and `assets/` contains presentation figures.

## Local preview

From this directory:

```bash
python3 -m http.server 8765 --bind 127.0.0.1
```

Open http://127.0.0.1:8765/. The same HTML/CSS and assets are served on GitHub Pages.
