# Bearing Fault Detection

This project classifies CWRU bearing vibration data into four classes:
`Normal`, `Ball`, `InnerRace`, and `OuterRace`.

The data is downloaded with KaggleHub from:

```text
brjapon/cwru-bearing-datasets
```

The raw data files do not need to be submitted with the code.

## Files

- `run_experiment.py` runs the full experiment and recreates the results.
- `model.py` contains loading, feature extraction, model training, and evaluation.
- `data/download_data.py` downloads the Kaggle dataset.
- `requirements.txt` lists the Python packages used.
- `results_summary.csv` is written after the experiment finishes.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Download the data:

```bash
python data/download_data.py
```

Run the experiment:

```bash
python run_experiment.py
```

The experiment also calls the downloader automatically if no local raw data is
found. If the download fails, make sure KaggleHub can access your Kaggle account
or API token.

## Data Location

By default, KaggleHub stores the dataset in its local cache and prints the path.
The model also supports local data in `data/raw/` if it already exists.

You can force a specific dataset folder with:

```bash
$env:CWRU_DATA_DIR="path_to_dataset_folder"
python run_experiment.py
```

## Main Settings

The main settings are at the top of `model.py`:

- random seed: `42`
- sample rate: `48000`
- window length: `4096`
- step size: `2048`
- cross-defect holdout size: `021`

The strongest leakage-safe result is from the cross-defect split using the
`Centroid` model.

## Held-out fault size results

The experiment compares a time-blocked split with a cross-defect test that holds
out faults measuring 0.021 inches. On the held-out size, nearest centroid reached
95.94% accuracy and 96.68% macro F1. Logistic regression, linear SVM, and Extra Trees
scored about 69% accuracy on that split despite perfect scores on the easier
time-blocked split.

See [the experiment report](final_report.md) and [full results](results_summary.csv).
