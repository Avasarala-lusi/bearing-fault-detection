import os
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat
from scipy.signal import butter, hilbert, sosfiltfilt

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.neighbors import NearestCentroid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import LinearSVC

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings(
    "ignore",
    message="self.within_class_std_dev_ has at least 1 zero standard deviation.*",
    category=UserWarning,
)


DATA_DIR = Path("data")

FS = 48000
WIN = 4096
STEP = 2048
SEED = 42
HOLDOUT_DEFECT = "021"
NORMAL_TRAIN_FRAC = 0.75

# SKF 6205-2RS JEM drive-end bearing characteristic frequency factors.
# Source: CWRU bearing specifications / benchmark literature.
DEFAULT_RPM = 1772.0
BPFI_FACTOR = 5.4152
BPFO_FACTOR = 3.5848
BSF_FACTOR = 2.3568
FTF_FACTOR = 0.3983

ENVELOPE_BANDS = [(500, 2000), (2000, 4000), (4000, 8000), (8000, 12000)]
ENVELOPE_FILTERS = [
    butter(4, band, btype="bandpass", fs=FS, output="sos")
    for band in ENVELOPE_BANDS
]


@dataclass
class SignalRecord:
    name: str
    label: str
    defect: str | None
    de: np.ndarray
    fe: np.ndarray | None
    rpm: float

    @property
    def shaft_hz(self) -> float:
        return self.rpm / 60.0


def parse_file(fname):
    b = os.path.basename(str(fname))
    if b.startswith("Time_Normal") or b.startswith("Normal"):
        return "Normal", None
    for prefix, cls in [("IR", "InnerRace"), ("OR", "OuterRace"), ("B", "Ball")]:
        if b.startswith(prefix):
            digits = b[len(prefix) : len(prefix) + 3]
            return cls, digits if digits.isdigit() else None
    return None, None


def _pick_key(mat, tag, suffix):
    keys = [k for k in mat if k.endswith(suffix)]
    if not keys:
        return None
    return next((k for k in keys if tag in k), keys[0])


def _read_rpm(mat, tag):
    keys = [k for k in mat if k.endswith("RPM")]
    key = next((k for k in keys if tag in k), keys[0] if keys else None)
    if key is None:
        return DEFAULT_RPM
    return float(np.asarray(mat[key]).squeeze())


def has_raw_files(root):
    root = Path(root)
    return any(parse_file(path.name)[0] is not None for path in root.rglob("*.mat"))


def get_data_root():
    override = os.environ.get("CWRU_DATA_DIR")
    if override:
        path = Path(override).expanduser()
        if not has_raw_files(path):
            raise FileNotFoundError(f"CWRU_DATA_DIR has no expected .mat files: {path}")
        return path

    if has_raw_files(DATA_DIR):
        return DATA_DIR

    from data.download_data import ensure_dataset

    return ensure_dataset()


def load_records(folder):
    records = []
    for path in sorted(Path(folder).rglob("*.mat")):
        label, defect = parse_file(path)
        if label is None:
            continue

        mat = loadmat(path)
        tag = path.stem.split("_")[-1]
        de_key = _pick_key(mat, tag, "_DE_time")
        fe_key = _pick_key(mat, tag, "_FE_time")
        if de_key is None:
            continue

        de = np.asarray(mat[de_key]).squeeze().astype(np.float64)
        fe = (
            np.asarray(mat[fe_key]).squeeze().astype(np.float64)
            if fe_key is not None
            else None
        )
        n = len(de) if fe is None else min(len(de), len(fe))
        records.append(
            SignalRecord(
                name=path.name,
                label=label,
                defect=defect,
                de=de[:n],
                fe=fe[:n] if fe is not None else None,
                rpm=_read_rpm(mat, tag),
            )
        )
    return records


def load_data():
    data_root = get_data_root()
    print(f"  data path: {data_root}")
    records = load_records(data_root)
    print(f"  {len(records)} files loaded")
    return records


def clone_segment(record, name_suffix, start, stop):
    fe = record.fe[start:stop] if record.fe is not None else None
    return SignalRecord(
        name=f"{record.name}{name_suffix}",
        label=record.label,
        defect=record.defect,
        de=record.de[start:stop],
        fe=fe,
        rpm=record.rpm,
    )


def split_cross_defect(records, holdout=HOLDOUT_DEFECT):
    train, test = [], []
    for record in records:
        if record.label == "Normal":
            cut = int(len(record.de) * NORMAL_TRAIN_FRAC)
            train.append(clone_segment(record, "_train", 0, max(cut - WIN, 0)))
            test.append(
                clone_segment(
                    record,
                    "_test",
                    min(cut + WIN, len(record.de)),
                    len(record.de),
                )
            )
        elif record.defect == holdout:
            test.append(record)
        else:
            train.append(record)
    return train, test


def split_time_blocked(records):
    train, test = [], []
    for record in records:
        cut = int(len(record.de) * NORMAL_TRAIN_FRAC)
        train.append(clone_segment(record, "_train", 0, max(cut - WIN, 0)))
        test.append(
            clone_segment(
                record,
                "_test",
                min(cut + WIN, len(record.de)),
                len(record.de),
            )
        )
    return train, test


def iter_windows(record):
    n = (len(record.de) - WIN) // STEP + 1
    if n <= 0:
        return
    for i in range(n):
        start = i * STEP
        stop = start + WIN
        de_w = record.de[start:stop]
        fe_w = record.fe[start:stop] if record.fe is not None else None
        yield de_w, fe_w


def band_sum(freq, spectrum, lo, hi):
    mask = (freq >= lo) & (freq < hi)
    return float(spectrum[mask].sum())


def _tol(freq):
    if len(freq) < 2:
        return 6.0
    return max(6.0, 0.51 * (freq[1] - freq[0]))


def peak_ratio(freq, spectrum, center):
    if center <= 0 or center >= freq[-1]:
        return 0.0
    tol = _tol(freq)
    mask = (freq >= center - tol) & (freq <= center + tol)
    if not mask.any():
        return 0.0
    return float(spectrum[mask].max() / (spectrum.sum() + 1e-12))


def energy_ratio(freq, spectrum, center):
    if center <= 0 or center >= freq[-1]:
        return 0.0
    tol = _tol(freq)
    mask = (freq >= center - tol) & (freq <= center + tol)
    if not mask.any():
        return 0.0
    return float(spectrum[mask].sum() / (spectrum.sum() + 1e-12))


def harmonic_features(freq, spectrum, base_freq, sideband_freq):
    out, peaks, energies = [], [], []
    for h in range(1, 6):
        center = h * base_freq
        peaks.append(peak_ratio(freq, spectrum, center))
        energies.append(energy_ratio(freq, spectrum, center))
        out.append(peak_ratio(freq, spectrum, center - sideband_freq))
        out.append(peak_ratio(freq, spectrum, center + sideband_freq))
    out.extend(peaks)
    out.extend(energies)
    out.append(float(sum(peaks)))
    out.append(float(max(peaks)))
    return out


def autocorr_features(x, shaft_hz):
    bpfi = BPFI_FACTOR * shaft_hz
    bpfo = BPFO_FACTOR * shaft_hz
    bsf = BSF_FACTOR * shaft_hz
    ftf = FTF_FACTOR * shaft_hz
    targets = [bpfo, bpfi, bsf, 2 * bsf, ftf, shaft_hz]

    x = x - x.mean()
    denom = float(np.dot(x, x) + 1e-12)
    max_lag = min(1200, len(x) - 1)
    ac = np.correlate(x, x, mode="full")[len(x) - 1 : len(x) + max_lag] / denom

    out = []
    for freq in targets:
        lag = FS / freq
        lo = max(1, int(lag * 0.9))
        hi = min(max_lag, int(lag * 1.1) + 1)
        out.append(float(np.max(ac[lo:hi])) if hi > lo else 0.0)
    return out


def channel_features(x, shaft_hz):
    x = np.asarray(x, dtype=np.float64)
    std = x.std() + 1e-12
    rms = np.sqrt(np.mean(x * x))
    z = (x - x.mean()) / std
    aw = np.abs(x)
    zaw = np.abs(z)

    out = [
        float(x.mean()),
        float(x.std()),
        float(rms),
        float(aw.max()),
        float(np.ptp(x)),
        float(stats.kurtosis(x)),
        float(stats.skew(x)),
        float(aw.max() / (rms + 1e-12)),
        float(stats.kurtosis(z)),
        float(stats.skew(z)),
        float(zaw.max()),
        float(zaw.mean()),
        float(np.quantile(z, 0.95) - np.quantile(z, 0.05)),
    ]

    freq = np.fft.rfftfreq(len(z), 1 / FS)
    spectrum = np.abs(np.fft.rfft(z))
    spectrum_norm = spectrum / (spectrum.sum() + 1e-12)
    out.extend(
        [
            float((freq * spectrum_norm).sum()),
            float(-(spectrum_norm * np.log2(spectrum_norm + 1e-12)).sum()),
            float(freq[np.argmax(spectrum)]),
        ]
    )
    for lo in range(0, 12000, 500):
        out.append(band_sum(freq, spectrum_norm, lo, lo + 500))
    for lo in range(0, 6000, 250):
        out.append(band_sum(freq, spectrum_norm, lo, lo + 250))

    bpfi = BPFI_FACTOR * shaft_hz
    bpfo = BPFO_FACTOR * shaft_hz
    bsf = BSF_FACTOR * shaft_hz
    ftf = FTF_FACTOR * shaft_hz

    for sos in ENVELOPE_FILTERS:
        filtered = sosfiltfilt(sos, z)
        env = np.abs(hilbert(filtered))
        env -= env.mean()
        efreq = np.fft.rfftfreq(len(env), 1 / FS)
        env_spectrum = np.abs(np.fft.rfft(env))
        env_norm = env_spectrum / (env_spectrum.sum() + 1e-12)

        out.extend(
            [
                float((efreq * env_norm).sum()),
                float(-(env_norm * np.log2(env_norm + 1e-12)).sum()),
                float(efreq[np.argmax(env_spectrum)]),
            ]
        )
        out.extend(harmonic_features(efreq, env_spectrum, bpfo, shaft_hz))
        out.extend(harmonic_features(efreq, env_spectrum, bpfi, shaft_hz))
        out.extend(harmonic_features(efreq, env_spectrum, bsf, ftf))
        out.extend(harmonic_features(efreq, env_spectrum, 2 * bsf, ftf))
        for lo in range(0, 700, 25):
            out.append(band_sum(efreq, env_norm, lo, lo + 25))

    out.extend(autocorr_features(z, shaft_hz))
    env = np.abs(hilbert(sosfiltfilt(ENVELOPE_FILTERS[1], z)))
    env = (env - env.mean()) / (env.std() + 1e-12)
    out.extend(autocorr_features(env, shaft_hz))

    return np.nan_to_num(np.asarray(out, dtype=np.float64), copy=False).tolist()


def pair_features(de_w, fe_w, shaft_hz):
    de_features = channel_features(de_w, shaft_hz)
    if fe_w is None:
        fe_features = [0.0] * len(de_features)
    else:
        fe_features = channel_features(fe_w, shaft_hz)

    cross = [
        de_features[1] / (fe_features[1] + 1e-12),
        de_features[5] - fe_features[5],
        de_features[2] / (fe_features[2] + 1e-12),
    ]
    return de_features + fe_features + cross


def make_table(records):
    X, y = [], []
    for record in records:
        for de_w, fe_w in iter_windows(record):
            X.append(pair_features(de_w, fe_w, record.shaft_hz))
            y.append(record.label)
    cols = [f"f_{i:04d}" for i in range(len(X[0]))]
    return pd.DataFrame(X, columns=cols), np.asarray(y)


def evaluate(name, model, Xt, yt, order):
    pred = model.predict(Xt)
    acc = accuracy_score(yt, pred)
    prec = precision_score(yt, pred, average="macro", zero_division=0)
    rec = recall_score(yt, pred, average="macro", zero_division=0)
    f1 = f1_score(yt, pred, average="macro", zero_division=0)
    print(f"\n[{name}]  acc={acc:.4f}  P={prec:.4f}  R={rec:.4f}  F1={f1:.4f}")
    print(classification_report(yt, pred, zero_division=0))
    return {
        "name": name,
        "acc": acc,
        "prec": prec,
        "rec": rec,
        "f1": f1,
        "cm": confusion_matrix(yt, pred, labels=order),
    }


def plot_cm(result, order, path):
    cm = result["cm"]
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(order)))
    ax.set_yticks(range(len(order)))
    ax.set_xticklabels(order, rotation=30, ha="right")
    ax.set_yticklabels(order)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"{result['name']} confusion matrix")
    threshold = cm.max() / 2 if cm.max() else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def model_specs():
    return [
        ("Centroid", make_pipeline(MinMaxScaler(), NearestCentroid())),
        (
            "LogReg",
            make_pipeline(
                MinMaxScaler(),
                LogisticRegression(
                    max_iter=4000,
                    C=0.1,
                    class_weight="balanced",
                    random_state=SEED,
                ),
            ),
        ),
        (
            "LinearSVM",
            make_pipeline(
                MinMaxScaler(),
                LinearSVC(
                    C=0.03,
                    class_weight="balanced",
                    max_iter=20000,
                    random_state=SEED,
                    dual="auto",
                ),
            ),
        ),
        (
            "ExtraTrees",
            ExtraTreesClassifier(
                n_estimators=300,
                max_features="sqrt",
                min_samples_leaf=1,
                random_state=SEED,
                n_jobs=1,
                class_weight="balanced",
            ),
        ),
    ]


def train_and_eval(train_records, test_records, split_tag, plot_prefix):
    print(f"\n[{split_tag}] extracting features...")
    Xtr, ytr = make_table(train_records)
    Xte, yte = make_table(test_records)
    print(f"  train rows: {len(Xtr)}  |  test rows: {len(Xte)}")
    train_counts = {k: int(v) for k, v in pd.Series(ytr).value_counts().items()}
    test_counts = {k: int(v) for k, v in pd.Series(yte).value_counts().items()}
    print(f"  train classes: {train_counts}")
    print(f"  test  classes: {test_counts}")

    order = sorted(set(ytr) | set(yte))
    results = []
    for model_name, model in model_specs():
        print(f"\nTraining {model_name}...")
        model.fit(Xtr, ytr)
        result = evaluate(model_name, model, Xte, yte, order)
        plot_cm(result, order, f"confusion_{plot_prefix}_{model_name.lower()}.png")
        results.append(result)

    print(f"\n[{split_tag}] summary:")
    print(f"{'model':<12}{'acc':>8}{'prec':>8}{'rec':>8}{'f1':>8}")
    for result in results:
        print(
            f"{result['name']:<12}"
            f"{result['acc']:>8.4f}"
            f"{result['prec']:>8.4f}"
            f"{result['rec']:>8.4f}"
            f"{result['f1']:>8.4f}"
        )
    return results


def main():
    np.random.seed(SEED)

    print("Loading data...")
    records = load_data()
    print(f"  classes: {sorted({r.label for r in records})}")
    print(f"  defect sizes: {sorted({r.defect for r in records if r.defect})}")
    print("  using DE + FE channels")
    print(f"  window={WIN}, step={STEP}")

    print("\nSplit A: time-blocked with a full-window gap")
    blocked_train, blocked_test = split_time_blocked(records)
    print(
        f"  train files/segments: {len(blocked_train)}"
        f"  |  test files/segments: {len(blocked_test)}"
    )
    res_blocked = train_and_eval(
        blocked_train,
        blocked_test,
        split_tag="time-blocked gap split",
        plot_prefix="blocked",
    )

    print(f"\nSplit B: cross-defect holdout ({HOLDOUT_DEFECT})")
    xdef_train, xdef_test = split_cross_defect(records, holdout=HOLDOUT_DEFECT)
    print(
        f"  train files/segments: {len(xdef_train)}"
        f"  |  test files/segments: {len(xdef_test)}"
    )
    res_xdef = train_and_eval(
        xdef_train,
        xdef_test,
        split_tag=f"cross-defect holdout {HOLDOUT_DEFECT}",
        plot_prefix="xdef",
    )

    rows = []
    for split_tag, results in [
        ("time_blocked_gap", res_blocked),
        (f"xdefect_{HOLDOUT_DEFECT}", res_xdef),
    ]:
        for result in results:
            rows.append(
                {
                    "split": split_tag,
                    "model": result["name"],
                    "accuracy": result["acc"],
                    "precision": result["prec"],
                    "recall": result["rec"],
                    "f1_macro": result["f1"],
                }
            )
    pd.DataFrame(rows).to_csv("results_summary.csv", index=False)
    print("\nsaved results_summary.csv + confusion plots")

    return res_blocked, res_xdef


if __name__ == "__main__":
    main()
