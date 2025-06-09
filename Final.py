# %%
import os
import random
import time
import glob
import math
from collections import Counter, defaultdict

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import torch
import intel_extension_for_pytorch as ipex
import torch.nn as nn
from torch import amp
from torch.optim.lr_scheduler import OneCycleLR

from sklearnex import patch_sklearn

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.preprocessing import (
    LabelEncoder,
    LabelBinarizer,
    StandardScaler,
)
from sklearn.impute import SimpleImputer
from sklearn.semi_supervised import SelfTrainingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import TomekLinks, RandomUnderSampler

from skorch import NeuralNetClassifier
from skorch.callbacks import LRScheduler, EarlyStopping
import skorch.utils as sk_utils

# For inline plotting in Jupyter notebooks
#%matplotlib inline
# adding a testing comment to test git
# Thread allocation for BLAS backends
env_threads = {
    "OMP_NUM_THREADS": "28",
    "MKL_NUM_THREADS": "28",
    "OPENBLAS_NUM_THREADS": "28",
    "DAAL_NUM_THREADS": "28",
}
for var, val in env_threads.items():
    os.environ[var] = val

# Global seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Patch sklearn to use Intel optimizations
patch_sklearn()

# %%
# -------------------------------
# Safe tensor-to-NumPy conversion
# -------------------------------
_old_to_numpy = sk_utils.to_numpy

def _to_numpy_safe(x):
    """
    Ensure bfloat16 tensors are cast to float32 before converting to numpy.
    """
    if isinstance(x, torch.Tensor) and x.dtype is torch.bfloat16:
        x = x.float()
    return _old_to_numpy(x)

sk_utils.to_numpy = _to_numpy_safe

# --------------------------------------
# Override predict_proba for NeuralNetClassifier
# --------------------------------------

def _predict_proba_safe(self, X):
    """
    Compute predict_proba on NumPy inputs, ensuring outputs
    are cast to float32 before conversion and concatenated correctly.
    """
    nonlin = self._get_predict_nonlinearity()
    outputs = []
    for yp in self.forward_iter(X, training=False):
        tensor = yp[0] if isinstance(yp, tuple) else yp
        prob = nonlin(tensor).float()
        outputs.append(prob)
    return torch.cat(outputs, dim=0).cpu().numpy()

NeuralNetClassifier.predict_proba = _predict_proba_safe



# %%
# --------------------------------------
# Plotting utilities
# --------------------------------------

def plot_class_distribution(data, label=None, title=None, ax=None):
    """
    Plot a bar chart of class counts.

    Parameters
    ----------
    data   : pd.DataFrame | pd.Series | np.ndarray | list
        Either a full dataframe that contains the label column,
        or the label vector itself.
    label  : str | int | None
        Column name (or index) of the label if `data` is a dataframe.
        Ignored when `data` is already a 1-D label vector.
    title  : str
        Optional title for the plot.
    ax     : matplotlib Axes
        Plot onto an existing axis (useful for subplots).
    """
    if isinstance(data, pd.DataFrame):
        if label is None:
            raise ValueError("Please specify `label=` when passing a DataFrame.")
        y = data[label]
    else:
        y = pd.Series(data)

    counts = y.value_counts().sort_index()
    total = counts.sum()

    if ax is None:
        _, ax = plt.subplots(figsize=(12, 6))

    sns.barplot(
        x=counts.index.astype(str),
        y=counts.values,
        hue=counts.index.astype(str),
        palette='viridis',
        dodge=False,
        legend=False,
        ax=ax,
    )
    ax.set_ylabel("count")
    ax.set_xlabel(label if label else "label")
    ax.set_title(title or "Class distribution")

    # Annotate bars with count and percentage
    for bar in ax.patches:
        h = bar.get_height()
        x = bar.get_x() + bar.get_width() / 2
        ax.text(
            x,
            h + max(counts.values) * 0.01,
            f"{int(h):,}\n({h/total*100:.1f}%)",
            ha='center',
            va='bottom',
            fontsize=9,
        )

    plt.tight_layout()
    return ax


def tidy_axis(ax, rotation=45, fontsize=6):
    """
    Apply consistent formatting to an axis: rotate x-ticks and set label size.
    """
    ax.tick_params(axis='x', rotation=rotation, labelsize=fontsize)
    plt.tight_layout()
    return ax

# %%
# --------------------------------------
# Resampling pipelines
# --------------------------------------

def make_realistic_resampler(y_train, under_frac=0.8, smote_frac=0.8, singleton_target=15):
    """
    Build a sampling-plus-preprocess pipeline that is *leak-safe*:
    - mean impute → scale → RUS → ROS(singletons) → SMOTE → TomekLinks
    All resampling ratios are computed from y_train only.
    """
    ctr = Counter(y_train)

    # 1) Controlled undersampling of the majority class
    maj_label, maj_count = ctr.most_common(1)[0]
    target_maj = int(under_frac * maj_count)
    rus = RandomUnderSampler(
        sampling_strategy={maj_label: target_maj},
        random_state=SEED
    )

    # 2) Give every singleton at least `singleton_target` examples
    ros = RandomOverSampler(
        sampling_strategy={lbl: singleton_target for lbl, c in ctr.items() if c == 1},
        random_state=SEED
    )

    # Peek at post-ROS counts so SMOTE knows its targets
    _, y_tmp = ros.fit_resample(
        *rus.fit_resample(np.zeros_like(y_train).reshape(-1, 1), y_train)
    )
    post_ctr = Counter(y_tmp)
    _, post_maj_count = post_ctr.most_common(1)[0]

    # 3) SMOTE the still-minor classes up to ≤ smote_frac of majority
    smote_strat = {}
    for cls, cnt in post_ctr.items():
        if cls == maj_label:
            continue
        raw_target = min(int(smote_frac * post_maj_count), cnt * 3)
        if raw_target > cnt:
            smote_strat[cls] = raw_target

    k = max(1, min(5, min(post_ctr.values()) - 1))
    sm = SMOTE(
        sampling_strategy=smote_strat,
        k_neighbors=k,
        random_state=SEED
    )

    # 4) Full leakage-safe pipeline
    return ImbPipeline([
        ('impute', SimpleImputer(strategy='mean')),
        ('scale', StandardScaler()),
        ('under', rus),
        ('ros1', ros),
        ('smote', sm),
        ('clean', TomekLinks())
    ])


def make_kdd_sampler(y_train):
    """
    KDD-specific sampler: custom under/SMOTE fractions.
    """
    return make_realistic_resampler(y_train, under_frac=0.3, smote_frac=0.4)

# %%
# --------------------------------------
# Probability alignment utility
# --------------------------------------

def align_probabilities(y_test, probas, model_classes):
    """
    Align modeled probabilities to the ground-truth label order.
    Returns binarized y and aligned probability matrix.
    """
    lb = LabelBinarizer().fit(model_classes)
    y_bin = lb.transform(y_test)

    # Handle binary case
    if y_bin.ndim == 1:
        y_bin = np.vstack([1 - y_bin, y_bin]).T
        probas = np.vstack([1 - probas, probas]).T

    # Re-order/drop probability columns so they match classes
    aligned = np.zeros_like(y_bin, dtype=float)
    class_to_idx = {cls: idx for idx, cls in enumerate(model_classes)}
    for col_idx, cls in enumerate(model_classes):
        aligned[:, class_to_idx[cls]] = probas[:, col_idx]

    return y_bin, aligned


# %%
# --------------------------------------
# Training & evaluation utilities
# --------------------------------------

def train_and_evaluate(models, X_train, y_train_semi, X_test, y_test, dataset_name):
    """
    Train and evaluate semi-supervised and supervised classifiers.
    Returns a dict of performance metrics per model.
    """
    print(f"\n=== {dataset_name} ===\nSemi-Supervised Classification\n{'-'*40}")
    results = {}

    # Build a purely-labeled subset for fully-supervised baselines
    sup_mask = (y_train_semi != -1)
    X_sup, y_sup = X_train[sup_mask], y_train_semi[sup_mask]

    for name, clf in models.items():
        t0 = time.time()

        # Fit on the right target
        if isinstance(clf, SelfTrainingClassifier):
            clf.fit(X_train, y_train_semi)
        else:
            clf.fit(X_sup, y_sup)

        duration = time.time() - t0

        # Predictions and class checks
        y_pred = clf.predict(X_test)
        print("Classes in y_test:", np.unique(y_test))
        print("Classes predicted:", np.unique(y_pred))
        missing = set(np.unique(y_test)) - set(np.unique(y_pred))
        print("Classes with no predictions:", missing)

        # Classification report
        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

        print(f"\n--- {name} ---")
        print(classification_report(y_test, y_pred, zero_division=0))

        df_rpt = pd.DataFrame(report).T
        fname = f"{dataset_name}_{name.replace(' ', '_')}_class_report.csv"
        df_rpt.to_csv(os.path.join("/home/cwtgl", fname), index=True)
        print(f"Saved full report to {fname}")

        # PR-AUC & ROC-AUC
        classes = np.unique(y_test)
        probas = clf.predict_proba(X_test)
        if len(classes) == 2:
            positive = probas[:, 1]
            print("Binary PR-AUC:", average_precision_score(y_test, positive))
            print("Binary ROC-AUC:",  roc_auc_score(y_test, positive))
        else:
            model_classes = clf.classes_
            y_bin, aligned_probas = align_probabilities(y_test, probas, model_classes)
            print("Macro PR-AUC:", average_precision_score(y_bin, aligned_probas, average='macro'))
            print("Macro ROC-AUC:",  roc_auc_score(y_bin, aligned_probas, average='macro', multi_class='ovo'))

        print(f"{name} trained in {duration:.2f}s.")

        results[name] = {
            "accuracy": report["accuracy"],
            "precision": report["macro avg"]["precision"],
            "recall": report["macro avg"]["recall"],
            "f1_score": report["macro avg"]["f1-score"],
            "training_time_seconds": duration
        }

    return results


# %%
# --------------------------------------
# Label detection & loading
# --------------------------------------

def find_label_column(columns, preferred=("label","class","target")):
    """Return first matching lower-case name or last column."""
    for col in columns[::-1]:
        if col.lower() in preferred: return col
    return columns[-1]

# Data Loading & Preprocessing
def load_and_preprocess_cicids(filepath):
    df=pd.read_csv(filepath, low_memory=False)
    target=find_label_column(df.columns)
    plot_class_distribution(df,label=target,title="Raw CICIDS2017")
    df.replace('-',np.nan,inplace=True)
    features=df.columns.drop(target)
    df[features]=df[features].apply(pd.to_numeric,errors='coerce')
    df[target] = df[target].fillna(df[target].mode(dropna=True)[0])
    X=df[features].to_numpy(np.float32); y=df[target].to_numpy()
    le=LabelEncoder().fit(y); return X,le.transform(y),le.classes_

def load_and_preprocess_kdd(filepath):
    df=pd.read_csv(filepath, low_memory=False)
    target=find_label_column(df.columns)
    plot_class_distribution(df,label=target,title="Raw KDD")
    df.replace('-',np.nan,inplace=True)
    features=df.columns.drop(target)
    df[features]=df[features].apply(pd.to_numeric,errors='coerce')
    df[target] = df[target].fillna(df[target].mode(dropna=True)[0])
    X=df[features].to_numpy(np.float32); y=df[target].to_numpy()
    le=LabelEncoder().fit(y); return X,le.transform(y),le.classes_

def load_and_preprocess_unsw(filepath):
    df=pd.read_csv(filepath,low_memory=False)
    target=find_label_column(df.columns)
    plot_class_distribution(df,label=target,title="Raw UNSW")
    df.replace('-',np.nan,inplace=True)
    features=df.columns.drop(target)
    df[features]=df[features].apply(pd.to_numeric,errors='coerce')
    df[features] = df[features].fillna(0)
    df[target] = df[target].fillna(df[target].mode(dropna=True)[0])
    X=df[features].to_numpy(np.float32); y=df[target].to_numpy()
    le=LabelEncoder().fit(y); return X,le.transform(y),le.classes_


# %%
# --------------------------------------
# Neural network model definitions
# --------------------------------------

# b) Slightly deeper fully connected network
class SimpleMLP(nn.Module):
    def __init__(self, n_in, n_out, hidden=(256, 128), p=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, hidden[0]), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Dropout(p),
            nn.Linear(hidden[1], n_out)
        )
    def forward(self, x):
        return self.net(x)

class IpexNetClassifier(NeuralNetClassifier):
    """NeuralNetClassifier + Intel® Extension for PyTorch speed-ups."""

    def initialize_module(self, *args, **kwargs):
        super().initialize_module(*args, **kwargs)
        # Uncomment to compile with TorchDynamo + IPEX
        # self.module_ = torch.compile(self.module_, backend="ipex")
        return self

    def initialize_optimizer(self, *args, **kwargs):
        super().initialize_optimizer(*args, **kwargs)
        self.module_, self.optimizer_ = ipex.optimize(
            self.module_,
            optimizer=self.optimizer_,
            dtype=torch.bfloat16,
            level="O1",
            inplace=True,
            auto_kernel_selection=True
        )
        return self

    def train_step_single(self, batch, **fit_params):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return super().train_step_single(batch, **fit_params)

    def validation_step(self, batch, **fit_params):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return super().validation_step(batch, **fit_params)

    def evaluation_step(self, batch, training=False):
        with torch.autocast("cpu", dtype=torch.bfloat16):
            return super().evaluation_step(batch, training=training)

# %%
# --------------------------------------
# Model factory utilities
# --------------------------------------

def make_ipex_mlp(
    n_features: int,
    n_classes:  int,
    *,
    device: str      = "cpu",
    criterion       = None,
    max_epochs: int  = 40,
    lr: float        = 5e-3,
    batch_size: int  = 4096,
    num_workers: int = 6,
    extra_callbacks = None,
):
    """
    Build a skorch-based MLP that is:
      • Intel IPEX-optimised (BF16 + fused AdamW)
      • compiled once with torch.compile(..., backend='ipex')
      • fed by DataLoaders that keep workers alive between epochs
    """
    def _steps_per_epoch(n):
        return math.ceil(n / batch_size)

    return IpexNetClassifier(
        module                  = SimpleMLP,
        module__n_in            = n_features,
        module__n_out           = n_classes,
        max_epochs              = max_epochs,
        lr                      = lr,
        batch_size              = batch_size,
        criterion               = criterion or nn.CrossEntropyLoss(),
        optimizer               = torch.optim.AdamW,
        iterator_train__shuffle = True,
        iterator_train__num_workers = num_workers,
        device                  = device,
        callbacks               = (extra_callbacks or []) + [
            ("early_stop", EarlyStopping(patience=8, monitor="valid_loss")),
        ],
    )


# %%
class FixedClassWrapper(BaseEstimator, ClassifierMixin):
    """
    Wraps an arbitrary classifier so that:

      1. `classes_` is always the full range [0 … n_total-1].
      2. `predict_proba` returns a matrix with exactly n_total columns,
         ordered from class 0 up to class n_total-1.
    """

    def __init__(self, base_estimator, n_total: int):
        self.base_estimator = base_estimator
        self.n_total = n_total

    def fit(self, X, y):
        """Fit the underlying estimator and build a full `classes_` array."""
        self.base_estimator.fit(X, y)
        seen = list(self.base_estimator.classes_)
        missing = [c for c in range(self.n_total) if c not in seen]
        # classes_ must be in ascending order
        self.classes_ = np.array(seen + missing, dtype=int)
        return self

    def predict(self, X):
        """Delegate prediction to the wrapped estimator."""
        return self.base_estimator.predict(X)

    def predict_proba(self, X):
        """
        Get probabilities for the existing classes, then pad
        with zeros so the result has shape (n_samples, n_total).
        """
        probs = self.base_estimator.predict_proba(X)
        n_samples = probs.shape[0]
        full_probs = np.zeros((n_samples, self.n_total), dtype=probs.dtype)

        for idx, cls in enumerate(self.base_estimator.classes_):
            full_probs[:, cls] = probs[:, idx]

        return full_probs

    def get_params(self, deep=True):
        """
        Return parameters for this estimator.
        `deep` is ignored since we only expose top-level params.
        """
        return {"base_estimator": self.base_estimator, "n_total": self.n_total}

    def set_params(self, **params):
        """
        Set parameters on this wrapper or pass them down to
        the wrapped estimator.
        """
        if "base_estimator" in params:
            self.base_estimator = params.pop("base_estimator")
        if "n_total" in params:
            self.n_total = params.pop("n_total")
        # anything left is for the base estimator
        if params:
            self.base_estimator.set_params(**params)
        return self

# %%
def cross_validated_workflow(
    X,
    y,
    dataset_name: str,
    sampler_fn,
    n_splits: int = 5,
    sample_frac: float = 1.0,
) -> pd.DataFrame:
    """
    Perform a 5-fold CV that is leak-safe even with label-driven samplers.
    Returns a DataFrame of mean/std metrics aggregated across folds.
    """
    # — optionally downsample once up front —
    if sample_frac < 1.0:
        rng = np.random.default_rng(SEED)
        sample_size = int(len(X) * sample_frac)
        indices = rng.choice(len(X), size=sample_size, replace=False)
        X, y = X[indices], y[indices]
        print(f"[{dataset_name}] downsampled to {sample_size} rows.")

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    fold_metrics = defaultdict(list)

    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y), start=1):
        print(f"\n=== {dataset_name} · Fold {fold_idx}/{n_splits} ===")
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # visualize before/after only on first & last fold
        if fold_idx in {1, n_splits}:
            plot_class_distribution(
                pd.Series(y_train, name="label"),
                title=f"{dataset_name} · Fold{fold_idx} TRAIN (pre-resample)"
            )

        # build and apply sampler
        pipeline = sampler_fn(y_train)
        X_train_rs, y_train_rs = pipeline.fit_resample(X_train, y_train)
        X_train = X_train_rs.astype(np.float32)

        if fold_idx in {1, n_splits}:
            plot_class_distribution(
                pd.Series(y_train_rs, name="label"),
                title=f"{dataset_name} · Fold{fold_idx} TRAIN (post-resample)"
            )

        # prepare callbacks for MLP
        steps_per_epoch = math.ceil(len(X_train) / 4096)
        onecycle_cb = (
            "one_cycle",
            LRScheduler(
                policy=OneCycleLR,
                max_lr=5e-3,
                epochs=40,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.05,
                div_factor=25.0,
                step_every="batch",
            )
        )

        # preprocess test set with same imputer/scaler
        X_test = pipeline.named_steps['impute'].transform(X_test)
        X_test = pipeline.named_steps['scale'].transform(X_test)
        X_test = X_test.astype(np.float32)

        # mask half the labels for semi-supervised learning
        mask = np.random.RandomState(SEED).rand(len(y_train_rs)) < 0.5
        y_train_semi = y_train_rs.copy()
        y_train_semi[mask] = -1

        n_features = X_train.shape[1]
        n_classes = len(np.unique(y_train_rs))

        # define semi-supervised models
        def make_semi(base_clf):
            wrapped = FixedClassWrapper(base_clf, n_total=n_classes)
            return SelfTrainingClassifier(wrapped)

        semi_rf = make_semi(RandomForestClassifier(n_estimators=1000,class_weight="balanced_subsample",n_jobs=-1,random_state=SEED))
        semi_lr = make_semi(LogisticRegression(max_iter=1000,class_weight="balanced",n_jobs=-1,random_state=SEED))
        base_mlp = (make_ipex_mlp(n_features,n_classes,extra_callbacks=[onecycle_cb]).initialize())
        base_mlp.module_ = torch.compile(base_mlp.module_, backend="ipex")
        semi_mlp = SelfTrainingClassifier(estimator=base_mlp,max_iter=3,threshold=0.75,verbose=False,)

        models = {
            "Random Forest": semi_rf,
            "LogReg":        semi_lr,
            "MLP":           semi_mlp,
        }

        # add supervised baselines (no resampling, no SSL)
        supervised = {
            "Dummy Majority (Supervised)": DummyClassifier(strategy='most_frequent', random_state=SEED),
            "Gaussian NB (Supervised)": GaussianNB(),
            "RF (Supervised)": RandomForestClassifier(n_estimators=1000,class_weight='balanced_subsample',n_jobs=-1,random_state=SEED),
            "LogReg (Supervised)": LogisticRegression(max_iter=1000,class_weight='balanced',n_jobs=-1,random_state=SEED),}
        for name, clf in supervised.items():
            models[name] = clone(clf)

        # evaluate all models and collect metrics
        fold_results = train_and_evaluate(models,X_train,y_train_semi,X_test,y_test,f"{dataset_name}-fold{fold_idx}")
        for (model_name, metric), value in [
            ((m, k), v) for m, metrics in fold_results.items() for k, v in metrics.items()
        ]:
            fold_metrics[(model_name, metric)].append(value)

    # aggregate across folds
    summary = [
        {
            "Model":  model_name,
            "Metric": metric,
            "Mean":   np.mean(vals),
            "Std":    np.std(vals),
        }
        for (model_name, metric), vals in fold_metrics.items()
    ]

    return pd.DataFrame(summary)

# %%
# Base directory for data and outputs
DATA_DIR = '/home/cwtgl'

# File paths
CICIDS_PATH = os.path.join(DATA_DIR, 'cleaned_and_reduced_CICIDS2017-02.csv')
KDD_PATH = os.path.join(DATA_DIR, 'combined_cleaned_KDD.csv')
UNSW_PATH = os.path.join(DATA_DIR, 'cleaned_UNSW-NB15-02.csv')


def slugify(name: str) -> str:
    """Convert spaces/exotic spaces to underscores."""
    return name.replace(' ', '_').replace('\xa0', '_')


def save_class_reports(results_dict, data_dir):
    """
    Gather class report CSVs into a single Excel workbook.
    """
    output_path = os.path.join(data_dir, 'all_class_reports.xlsx')
    with pd.ExcelWriter(output_path) as writer:
        wrote_any = False
        for ds_name, ds_results in results_dict.items():
            for model_name in ds_results:
                pattern = os.path.join(
                    data_dir,
                    f"{ds_name}-fold*_{slugify(model_name)}_class_report.csv"
                )
                for csv_file in glob.glob(pattern):
                    fold = os.path.basename(csv_file).split('_')[0].split('-')[-1]
                    sheet_name = f"{ds_name[:4]}_{fold}_{slugify(model_name)[:10]}"
                    pd.read_csv(csv_file, index_col=0).to_excel(
                        writer, sheet_name=sheet_name
                    )
                    wrote_any = True

        if not wrote_any:
            pd.DataFrame({'msg': ['no reports found']}).to_excel(
                writer, sheet_name='EMPTY'
            )

    print(f"All available class reports saved to {output_path}")


def save_combined_results(df, data_dir):
    """
    Save the combined results DataFrame to a CSV file.
    """
    out_csv = os.path.join(data_dir, 'combined_results.csv')
    df.to_csv(out_csv, index=False)
    print(f"Saved results successfully to {out_csv}")



def plot_metrics(df_all, metrics=None, palette='tab20'):
    """
    Generate bar plots for given metrics and training time.
    """
    if metrics is None:
        metrics = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    # Plot each accuracy/precision/recall/F1 metric
    for metric in metrics:
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(
            x='Dataset', y=metric, hue='Model', data=df_all, palette=palette)
        ax.set(ylim=(0, 1), title=f'Comparison of {metric} Across Datasets & Models')
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f', padding=3)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.show()
    # Plot training times
    plt.figure(figsize=(12, 6))
    ax = sns.barplot(
        x='Dataset', y='training_time_seconds', hue='Model', data=df_all,
        palette='mako'
    )
    ax.set(title='Training Time (Seconds) across Datasets & Models', ylabel='Time (seconds)')
    for container in ax.containers:
        ax.bar_label(container, fmt='%.1f', padding=3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()
    # Plot average model performance
    avg_perf = df_all.groupby('Model')[['Accuracy', 'Precision', 'Recall', 'F1 Score']].mean()
    avg_perf.plot.bar(figsize=(10, 6))
    plt.ylim(0, 1)
    plt.title('Average Model Performance Across All Datasets')
    plt.tight_layout()
    plt.show()



"""
Load datasets, run cross-validated workflows, combine results,
plot metrics, and save reports and combined outputs.
"""
# Load and preprocess each dataset
datasets = {
    'CICIDS2017': load_and_preprocess_cicids(CICIDS_PATH),
    'KDD': load_and_preprocess_kdd(KDD_PATH),
    'UNSW-NB15': load_and_preprocess_unsw(UNSW_PATH),
}
# Run workflows for each dataset
results = {}
for name, (X, y, _) in datasets.items():
    sampler = make_kdd_sampler if name == 'KDD' else make_realistic_resampler
    results[name] = cross_validated_workflow(
        X, y, dataset_name=name, sampler_fn=sampler
    )
# Convert DataFrames to dict-of-dicts for easy lookup
def df_to_dict(df):
    return df.pivot(index='Model', columns='Metric', values='Mean').to_dict(orient='index')
results_dict = {name: df_to_dict(df) for name, df in results.items()}
# Combine all results into a single DataFrame
records = []
for ds_name, ds_results in results_dict.items():
    for model, metrics in ds_results.items():
        records.append({
            'Dataset': ds_name,
            'Model': model,
            'Accuracy': metrics['accuracy'],
            'Precision': metrics['precision'],
            'Recall': metrics['recall'],
            'F1 Score': metrics['f1_score'],
            'training_time_seconds': metrics['training_time_seconds'],
        })
df_all = pd.DataFrame(records)
print('\nCombined Results across all Datasets:')
print(df_all)
# Plot performance metrics and training times
plot_metrics(df_all)
# Save detailed class reports and combined results CSV
save_class_reports(results_dict, DATA_DIR)
save_combined_results(df_all, DATA_DIR)





