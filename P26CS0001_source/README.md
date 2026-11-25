# Machine Learning Assignment 3 — P26CS0001

This project uses an independent stratified CIFAR-10 split of 45,000 training
images and 5,000 validation images, with 4,500 and 500 per class. The split
seed is 260001. See `EXPERIMENT_PLAN.md` for the Q1–Q4 design and selection rules.

## Run in Google Colab

Open `ML_Assignment_3_P26CS0001_Colab.ipynb`, select a GPU runtime, and upload
`P26CS0001_source.zip` to `/content/P26CS0001_source.zip`. Run the cells in
order. They install the environment, verify the split, train Q1–Q3, perform Q4
selection, freeze the choices, evaluate on the official test set, and export
the figures, metrics, checkpoints, and logs.

## Run locally

From this directory:

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install --extra-index-url https://download.pytorch.org/whl/cu124 -r requirements-lock.txt
bash tools/run_pipeline.sh
```

Use this workspace for all checkpoints and results. Training can resume
from its own checkpoints. Preserve the validation freeze when resuming an
evaluation.

## CVPR report

After finishing the run, examine all three Q4 boundary figures and
`artifacts/results/test/pair_examples.csv`. Record the student’s image-specific
observations in `report/pair_observations.json`, keyed by class index pair such as
`"3-5"`. Use the actual three pairs selected by this run.

Commit the project code, completed results, logs, and figures to the student's
repository. Put the student's name in `report/student_name.txt`, the full
experiment commit SHA in `report/source_commit.txt` and the GitHub URL in
`report/repository_url.txt`. Generate and verify the report with:

```bash
.venv/bin/python src/write_report.py
tectonic -k report/ML_Assignment_3_P26CS0001.tex
.venv/bin/python src/verify.py
```

Review the report's conclusions, identity and page limit before submission.
Codex assisted with code and report generation.
