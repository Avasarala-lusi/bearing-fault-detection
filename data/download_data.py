from pathlib import Path

import kagglehub


DATASET_HANDLE = "brjapon/cwru-bearing-datasets"


def download_dataset():
    path = Path(kagglehub.dataset_download(DATASET_HANDLE))
    print("Path to dataset files:", path)
    return path


def ensure_dataset():
    return download_dataset()


if __name__ == "__main__":
    download_dataset()
