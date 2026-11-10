"""Train and apply a risk model from whatever dataset a case type names."""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from .case_types import CaseType, load_case_types
from .settings import PROTECTED_COLUMNS, Settings

SAMPLE_FLAG = "sample_only"
FALLBACK_ROWS = 300


def dataset_path(case_type: CaseType, settings: Settings) -> Path:
    """Path of the dataset this case type names."""
    return settings.root / case_type.risk_model["dataset"]


def _fallback_rows(case_type: CaseType, settings: Settings) -> pd.DataFrame:
    """Plausible random rows with the dataset's column names, for when the file is missing."""
    scenario = yaml.safe_load(
        (settings.root / case_type.sample_content / "scenario.yaml").read_text(encoding="utf-8")
    )
    rng = random.Random(settings.seed)
    label = case_type.risk_model["label_column"]
    columns: dict[str, list[Any]] = scenario["fallback_columns"]
    rows = []
    for _ in range(FALLBACK_ROWS):
        row = {name: rng.choice(values) for name, values in columns.items()}
        row[label] = int(rng.random() < 0.2)
        row[SAMPLE_FLAG] = True
        rows.append(row)
    return pd.DataFrame(rows)


def load_dataset(case_type: CaseType, settings: Settings) -> tuple[pd.DataFrame, bool]:
    """Return (rows, is_real). Falls back to generated rows if the file is missing."""
    path = dataset_path(case_type, settings)
    if path.exists():
        return pd.read_csv(path, encoding="utf-8-sig"), True
    return _fallback_rows(case_type, settings), False


def split_rows(
    rows: pd.DataFrame, case_type: CaseType, settings: Settings, n_open: int, n_past: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified split into rows for the risk model, open cases and past cases."""
    from sklearn.model_selection import train_test_split

    label = case_type.risk_model["label_column"]

    def take(frame: pd.DataFrame, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if n <= 0:
            return frame, frame.iloc[:0]
        classes = frame[label].nunique()
        # Stratifying needs at least one row of each class on both sides.
        stratify = frame[label] if min(n, len(frame) - n) >= classes else None
        return train_test_split(frame, test_size=n, stratify=stratify, random_state=settings.seed)

    rest, open_rows = take(rows, n_open)
    risk_rows, past_rows = take(rest, n_past)
    return risk_rows, open_rows, past_rows


def excluded_columns(case_type: CaseType) -> set[str]:
    """Columns that never become model features."""
    risk = case_type.risk_model
    return {risk["label_column"], SAMPLE_FLAG, *risk.get("drop_columns", []), *PROTECTED_COLUMNS}


def prepare_features(rows: pd.DataFrame, case_type: CaseType) -> tuple[pd.DataFrame, pd.Series]:
    """Split rows into the feature table and the label."""
    label = case_type.risk_model["label_column"]
    features = rows.drop(columns=[c for c in excluded_columns(case_type) if c in rows.columns])
    return features, rows[label].astype(int)


def _pipeline(features: pd.DataFrame, classifier: Any) -> Any:
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline, make_pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric = [c for c in features.columns if pd.api.types.is_numeric_dtype(features[c])]
    categorical = [c for c in features.columns if c not in numeric]
    encoder = ColumnTransformer(
        [("numeric", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), numeric),
         ("categorical", OneHotEncoder(handle_unknown="ignore"), categorical)]
    )
    return Pipeline([("encode", encoder), ("classify", classifier)])


def _classifiers(seed: int) -> dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    return {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=150, min_samples_leaf=5, class_weight="balanced_subsample", n_jobs=2, random_state=seed
        ),
    }


def recall_at_top(labels: Any, scores: Any, share: float = 0.1) -> float:
    """Share of all positives found among the `share` highest-scored rows."""
    frame = pd.DataFrame({"label": list(labels), "score": list(scores)})
    top = frame.sort_values("score", ascending=False, kind="mergesort").head(max(1, round(len(frame) * share)))
    positives = frame["label"].sum()
    return float(top["label"].sum() / positives) if positives else 0.0


def cross_validate(features: pd.DataFrame, labels: pd.Series, seed: int) -> dict[str, dict[str, float]]:
    """5-fold metrics for both models, averaged over folds."""
    from sklearn.base import clone
    from sklearn.metrics import average_precision_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=seed).split(features, labels))
    results: dict[str, dict[str, float]] = {}
    for name, classifier in _classifiers(seed).items():
        scores: dict[str, list[float]] = {"roc_auc": [], "pr_auc": [], "recall_top_10pct": []}
        for train_idx, test_idx in folds:
            model = _pipeline(features, clone(classifier))
            model.fit(features.iloc[train_idx], labels.iloc[train_idx])
            probabilities = model.predict_proba(features.iloc[test_idx])[:, 1]
            truth = labels.iloc[test_idx]
            scores["roc_auc"].append(float(roc_auc_score(truth, probabilities)))
            scores["pr_auc"].append(float(average_precision_score(truth, probabilities)))
            scores["recall_top_10pct"].append(recall_at_top(truth, probabilities))
        results[name] = {key: round(sum(v) / len(v), 4) for key, v in scores.items()}
    return results


def train_case_type(case_type: CaseType, settings: Settings) -> dict[str, Any]:
    """Cross-validate both models and save a logistic regression fitted on the risk-model rows."""
    import joblib

    rows, is_real = load_dataset(case_type, settings)
    # Hold back exactly the rows the sample cases use, so the model never trains on them.
    risk_rows, _, _ = split_rows(rows, case_type, settings, *sample_counts(settings).get(case_type.name, (0, 0)))
    features, labels = prepare_features(risk_rows, case_type)
    metrics = cross_validate(features, labels, settings.seed)
    model = _pipeline(features, _classifiers(settings.seed)["logistic_regression"])
    model.fit(features, labels)
    settings.models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "columns": list(features.columns)}, _model_path(case_type, settings))
    return {
        "data": "real" if is_real else "sample",
        "rows": int(len(risk_rows)),
        "positives": int(labels.sum()),
        "features": sorted(features.columns),
        "metrics": metrics,
    }


def _split_total(total: int, weights: list[float]) -> list[int]:
    """Share a whole number out by weight, so the parts always add up to the total."""
    exact = [total * w / sum(weights) for w in weights]
    counts = [int(value) for value in exact]
    by_remainder = sorted(range(len(weights)), key=lambda i: exact[i] - counts[i], reverse=True)
    for i in by_remainder[: total - sum(counts)]:
        counts[i] += 1
    return counts


def sample_counts(settings: Settings) -> dict[str, tuple[int, int]]:
    """(open cases, past cases) for each case type with sample content.

    `case_type_shares` in settings.yaml weights the split; a type it leaves out gets weight 1,
    and with no shares the types split evenly.
    """
    names = [name for name, case_type in load_case_types(settings).items() if case_type.sample_content]
    if not names:
        return {}
    weights = [float(settings.case_type_shares.get(name, 1)) for name in names]
    opens = _split_total(settings.open_cases, weights)
    pasts = _split_total(settings.past_cases, weights)
    return {name: (opens[i], pasts[i]) for i, name in enumerate(names)}


def train_all(settings: Settings) -> str:
    """Train every case type that names a dataset and write results/risk_model.json."""
    results = {
        name: train_case_type(case_type, settings)
        for name, case_type in load_case_types(settings).items()
        if case_type.risk_model
    }
    settings.results_dir.mkdir(parents=True, exist_ok=True)
    path = settings.results_dir / "risk_model.json"
    path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return f"Risk model results saved to {path}: " + json.dumps(
        {k: v["metrics"] for k, v in results.items()}
    )


def _model_path(case_type: CaseType, settings: Settings) -> Path:
    return settings.models_dir / f"{case_type.name}.joblib"


_loaded: dict[str, dict[str, Any]] = {}


def predict_probability(
    case_type: CaseType, features: dict[str, Any], settings: Settings
) -> tuple[float, list[str]] | None:
    """Return (probability, top factors), or None if this case type has no trained model."""
    if not case_type.risk_model or not features:
        return None
    path = _model_path(case_type, settings)
    if not path.exists():
        return None
    import joblib

    key = str(path)
    if key not in _loaded:
        _loaded[key] = joblib.load(path)
    bundle = _loaded[key]
    frame = pd.DataFrame([{column: features.get(column) for column in bundle["columns"]}])
    model = bundle["model"]
    probability = float(model.predict_proba(frame)[0, 1])
    return probability, _top_factors(model, frame)


def _top_factors(model: Any, frame: pd.DataFrame, n: int = 3) -> list[str]:
    """The dataset fields pushing this probability up the most."""
    encoded = model.named_steps["encode"].transform(frame)
    values = encoded.toarray()[0] if hasattr(encoded, "toarray") else encoded[0]
    names = model.named_steps["encode"].get_feature_names_out()
    contributions = values * model.named_steps["classify"].coef_[0]
    order = sorted(range(len(names)), key=lambda i: contributions[i], reverse=True)
    factors = []
    for i in order[:n]:
        if contributions[i] > 0:
            factors.append(names[i].split("__", 1)[-1].replace("_", " "))
    return factors
