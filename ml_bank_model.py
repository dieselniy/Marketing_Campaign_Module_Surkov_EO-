import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_VERSION = "bank_deposit_normal_dialogue_v5"

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_PATH = PROJECT_DIR / "Kaggle Database" / "bank.csv"

TARGET_COLUMN = "deposit"

MODEL_DROP_COLUMNS = [
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "day",
]

NORMAL_DIALOGUE_MIN_DURATION = 180
NORMAL_DIALOGUE_MAX_DURATION = 720
CONTACT_PROFILE = "normal_dialogue"

RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.25


def fix_seed(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)


def resolve_data_path(fp=None):
    if fp is None:
        return DEFAULT_DATA_PATH

    path = Path(fp)
    if not path.is_absolute():
        path = PROJECT_DIR / path

    return path


def mk_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dt(fp=None):
    path = resolve_data_path(fp)
    return pd.read_csv(path, usecols=range(17))


def apply_contact_profile(df, contact_profile=CONTACT_PROFILE):
    if contact_profile != "normal_dialogue" or "duration" not in df.columns:
        return df.copy(), {
            "contact_profile": "all_contacts",
            "source_rows": len(df),
            "training_rows": len(df),
            "normal_duration_min": None,
            "normal_duration_max": None,
        }

    duration = pd.to_numeric(df["duration"], errors="coerce")
    normal_dialogue_mask = duration.between(
        NORMAL_DIALOGUE_MIN_DURATION,
        NORMAL_DIALOGUE_MAX_DURATION,
        inclusive="both",
    )
    filtered = df[normal_dialogue_mask].copy()

    return filtered, {
        "contact_profile": contact_profile,
        "source_rows": len(df),
        "training_rows": len(filtered),
        "normal_duration_min": NORMAL_DIALOGUE_MIN_DURATION,
        "normal_duration_max": NORMAL_DIALOGUE_MAX_DURATION,
    }


def get_drop_columns(drop_columns=None, dd=False):
    if drop_columns is None:
        columns = list(MODEL_DROP_COLUMNS)
    else:
        columns = list(drop_columns)

    if dd and "duration" not in columns:
        columns.append("duration")

    return columns


def prep_xy(df, tgt=TARGET_COLUMN, dd=False, drop_columns=None):
    work = df.copy()
    columns_to_drop = [
        column
        for column in get_drop_columns(drop_columns=drop_columns, dd=dd)
        if column in work.columns and column != tgt
    ]

    if columns_to_drop:
        work = work.drop(columns=columns_to_drop)

    y = work[tgt].map({"no": 0, "yes": 1})
    X = work.drop(columns=[tgt])

    cat_cols = X.select_dtypes(include="object").columns.tolist()
    num_cols = X.select_dtypes(exclude="object").columns.tolist()

    return X, y, cat_cols, num_cols


def split_dt(X, y, test_size=TEST_SIZE, seed=RANDOM_STATE):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )


def mk_prep(cat_cols, num_cols, scale_numeric=False):
    numeric_transformer = StandardScaler() if scale_numeric else "passthrough"

    return ColumnTransformer(
        transformers=[
            ("cat", mk_ohe(), cat_cols),
            ("num", numeric_transformer, num_cols),
        ]
    )


def mk_gb(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=RANDOM_STATE,
        )),
    ])


def mk_model(model_key, cat_cols, num_cols):
    if model_key == "dummy":
        return Pipeline([
            ("prep", mk_prep(cat_cols, num_cols)),
            ("clf", DummyClassifier(strategy="most_frequent")),
        ])

    if model_key == "logistic_regression":
        return Pipeline([
            ("prep", mk_prep(cat_cols, num_cols, scale_numeric=True)),
            ("clf", LogisticRegression(
                max_iter=3000,
                class_weight="balanced",
                random_state=RANDOM_STATE,
            )),
        ])

    if model_key == "random_forest":
        return Pipeline([
            ("prep", mk_prep(cat_cols, num_cols)),
            ("clf", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=5,
                class_weight="balanced",
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ])

    if model_key == "gradient_boosting":
        return mk_gb(mk_prep(cat_cols, num_cols))

    raise ValueError(f"Unknown model_key: {model_key}")


def threshold_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }


def optimize_threshold(model, X_valid, y_valid):
    y_proba = model.predict_proba(X_valid)[:, 1]
    thresholds = np.linspace(0.35, 0.65, 61)

    scores = [
        threshold_metrics(y_valid, y_proba, threshold)
        for threshold in thresholds
    ]

    return max(
        scores,
        key=lambda item: (item["accuracy"], item["f1"]),
    )


def eval_mdl(model, X_test, y_test, name="Gradient Boosting", threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    report = classification_report(y_test, y_pred)
    matrix = confusion_matrix(y_test, y_pred)

    return metrics, report, matrix, y_pred, y_proba


def feat_imp(model, n=20):
    preprocessor = model.named_steps["prep"]
    classifier = model.named_steps["clf"]

    if not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = preprocessor.get_feature_names_out()
    importances = classifier.feature_importances_

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        })
        .sort_values(by="importance", ascending=False)
        .head(n)
    )


def mk_artf(
    model,
    metrics,
    cat_cols,
    num_cols,
    input_cols,
    excluded_columns,
    threshold_info,
    contact_context,
):
    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": feat_imp(model, n=20),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "input_cols": input_cols,
        "excluded_columns": excluded_columns,
        "model_version": MODEL_VERSION,
        "contact_context": contact_context,
        "decision_threshold": float(threshold_info["threshold"]),
        "threshold_validation_metrics": threshold_info,
        "model_key": "gradient_boosting",
        "model_title": "Gradient Boosting",
        "target_mapping": {"no": 0, "yes": 1},
        "inverse_target_mapping": {0: "no", 1: "yes"},
    }


def build_training_data(
    fp=None,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    source_df = load_dt(fp)

    filtered_df, contact_context = apply_contact_profile(
        source_df,
        contact_profile=contact_profile,
    )

    excluded_columns = [
        column
        for column in get_drop_columns(drop_columns=drop_columns, dd=dd)
        if column in source_df.columns
    ]

    X, y, cat_cols, num_cols = prep_xy(
        filtered_df,
        dd=dd,
        drop_columns=drop_columns,
    )

    return {
        "source_df": source_df,
        "filtered_df": filtered_df,
        "contact_context": contact_context,
        "excluded_columns": excluded_columns,
        "X": X,
        "y": y,
        "cat_cols": cat_cols,
        "num_cols": num_cols,
    }


def train_gb_artifact(
    fp=None,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    fix_seed(RANDOM_STATE)

    data = build_training_data(
        fp=fp,
        dd=dd,
        drop_columns=drop_columns,
        contact_profile=contact_profile,
    )

    X = data["X"]
    y = data["y"]
    cat_cols = data["cat_cols"]
    num_cols = data["num_cols"]

    X_train, X_test, y_train, y_test = split_dt(X, y)

    X_fit, X_valid, y_fit, y_valid = split_dt(
        X_train,
        y_train,
        test_size=VALIDATION_SIZE,
        seed=43,
    )

    threshold_probe = mk_model(
        "gradient_boosting",
        cat_cols,
        num_cols,
    )
    threshold_probe.fit(X_fit, y_fit)

    threshold_info = optimize_threshold(
        threshold_probe,
        X_valid,
        y_valid,
    )

    model = mk_model(
        "gradient_boosting",
        cat_cols,
        num_cols,
    )
    model.fit(X_train, y_train)

    metrics, report, matrix, y_pred, y_proba = eval_mdl(
        model,
        X_test,
        y_test,
        name="Gradient Boosting",
        threshold=threshold_info["threshold"],
    )

    artifact = mk_artf(
        model=model,
        metrics=metrics,
        cat_cols=cat_cols,
        num_cols=num_cols,
        input_cols=X.columns.tolist(),
        excluded_columns=data["excluded_columns"],
        threshold_info=threshold_info,
        contact_context=data["contact_context"],
    )

    return artifact, X_train, X_test, y_test, y_pred, y_proba, report, matrix


def compare_models(
    fp=None,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    data = build_training_data(
        fp=fp,
        dd=dd,
        drop_columns=drop_columns,
        contact_profile=contact_profile,
    )

    X = data["X"]
    y = data["y"]
    cat_cols = data["cat_cols"]
    num_cols = data["num_cols"]

    X_train, X_test, y_train, y_test = split_dt(X, y)

    models = {
        "Dummy baseline": "dummy",
        "Logistic Regression": "logistic_regression",
        "Random Forest": "random_forest",
        "Gradient Boosting": "gradient_boosting",
    }

    rows = []

    for model_name, model_key in models.items():
        model = mk_model(model_key, cat_cols, num_cols)
        model.fit(X_train, y_train)

        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)

        rows.append({
            "model": model_name,
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, zero_division=0),
            "recall": recall_score(y_test, y_pred, zero_division=0),
            "f1": f1_score(y_test, y_pred, zero_division=0),
            "roc_auc": roc_auc_score(y_test, y_proba),
        })

    return (
        pd.DataFrame(rows)
        .sort_values(by="roc_auc", ascending=False)
        .reset_index(drop=True)
    )


def train_gb(
    fp=None,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    artifact, X_train, X_test, y_test, y_pred, y_proba, report, matrix = (
        train_gb_artifact(
            fp=fp,
            dd=dd,
            drop_columns=drop_columns,
            contact_profile=contact_profile,
        )
    )

    return {
        "metrics": artifact["metrics"],
        "report": report,
        "confusion_matrix": matrix,
        "feature_importance": artifact["feature_importance"],
        "excluded_columns": artifact["excluded_columns"],
        "input_cols": artifact["input_cols"],
        "contact_context": artifact["contact_context"],
        "decision_threshold": artifact["decision_threshold"],
        "threshold_validation_metrics": artifact["threshold_validation_metrics"],
        "X_train_shape": X_train.shape,
        "X_test_shape": X_test.shape,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba,
    }


def train_mdl(
    fp=None,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    return train_gb(
        fp=fp,
        dd=dd,
        drop_columns=drop_columns,
        contact_profile=contact_profile,
    )


def load_mtr(model_artifact):
    return model_artifact.get("metrics", {})


def pred_client(client_data, model_artifact):
    model = model_artifact["model"]
    inverse_mapping = model_artifact["inverse_target_mapping"]
    cat_cols = model_artifact.get("cat_cols", [])
    num_cols = model_artifact.get("num_cols", [])
    input_cols = model_artifact.get("input_cols") or cat_cols + num_cols
    threshold = model_artifact.get("decision_threshold", 0.5)

    client_df = pd.DataFrame([client_data])

    for column in input_cols:
        if column not in client_df.columns:
            client_df[column] = "unknown" if column in cat_cols else 0

    for column in cat_cols:
        if column in client_df.columns:
            client_df[column] = (
                client_df[column]
                .where(client_df[column].notna(), "unknown")
                .astype(str)
            )

    for column in num_cols:
        if column in client_df.columns:
            client_df[column] = (
                pd.to_numeric(client_df[column], errors="coerce")
                .fillna(0)
            )

    client_df = client_df[input_cols]

    probability = model.predict_proba(client_df)[0, 1]
    prediction = int(probability >= threshold)

    return {
        "prediction": prediction,
        "prediction_label": inverse_mapping[prediction],
        "deposit_probability": float(probability),
    }


if __name__ == "__main__":
    result = train_gb()
    comparison = compare_models()

    print("MODEL_VERSION:", MODEL_VERSION)
    print("DATA_PATH:", DEFAULT_DATA_PATH)
    print("Контекст контакта:", result["contact_context"])
    print("Используемые входы:", result["input_cols"])
    print("Исключенные поля:", result["excluded_columns"])
    print("Размер обучающей выборки:", result["X_train_shape"])
    print("Размер тестовой выборки:", result["X_test_shape"])
    print("Порог решения:", result["decision_threshold"])

    print("\nСравнение моделей:")
    print(comparison.round(4))

    print("\nGradient Boosting:")
    print(result["metrics"])
    print(result["report"])
    print(result["confusion_matrix"])
    print(result["feature_importance"])
