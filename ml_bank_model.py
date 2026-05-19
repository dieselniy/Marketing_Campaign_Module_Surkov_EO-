import random
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
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
from sklearn.preprocessing import OneHotEncoder


MOD_VERS = "bank_dep_v5"

projdir = Path(__file__).resolve().parent
defpath = projdir / "Kaggle Database" / "bank.csv"

targcol = "deposit"

dropcols = [
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "day",
]

normdial_min = 180
normdial_max = 720
contactprof = "norm dial"

seednum = 42
testsize = 0.2
valsize = 0.25


def fix_seed(seed=seednum):
    random.seed(seed)
    np.random.seed(seed)


def resolve_data_path(fp=None):
    if fp is None:
        return defpath

    path = Path(fp)
    if not path.is_absolute():
        path = projdir / path

    return path


def mk_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dt(fp=None):
    path = resolve_data_path(fp)
    return pd.read_csv(path, usecols=range(17))


def apply_contactprof(df, contactprof_val=contactprof):
    if contactprof_val != "norm dial" or "duration" not in df.columns:
        return df.copy(), {
            "contactprof": "all contacts",
            "source_rows": len(df),
            "training_rows": len(df),
            "normdial_min": None,
            "normdial_max": None,
        }

    duration = pd.to_numeric(df["duration"], errors="coerce")
    normdial_mask = duration.between(
        normdial_min,
        normdial_max,
        inclusive="both",
    )
    filtered = df[normdial_mask].copy()

    return filtered, {
        "contactprof": contactprof_val,
        "source_rows": len(df),
        "training_rows": len(filtered),
        "normdial_min": normdial_min,
        "normdial_max": normdial_max,
    }


def get_drop_columns(drop_columns=None, dd=False):
    if drop_columns is None:
        columns = list(dropcols)
    else:
        columns = list(drop_columns)

    if dd and "duration" not in columns:
        columns.append("duration")

    return columns


def prep_xy(df, tgt=targcol, dd=False, drop_columns=None):
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


def split_dt(X, y, test_size=testsize, seed=seednum):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=seed,
    )


def mk_prep(cat_cols, num_cols):
    return ColumnTransformer(
        transformers=[
            ("cat", mk_ohe(), cat_cols),
            ("num", "passthrough", num_cols),
        ]
    )


def mk_gb(preprocessor):
    return Pipeline([
        ("prep", preprocessor),
        ("clf", GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=seednum,
        )),
    ])


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
    contactctx,
):
    return {
        "model": model,
        "metrics": metrics,
        "feature_importance": feat_imp(model, n=20),
        "cat_cols": cat_cols,
        "num_cols": num_cols,
        "input_cols": input_cols,
        "excluded_columns": excluded_columns,
        "modelver": modelver,
        "contactctx": contactctx,
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
    contactprof_val=contactprof,
):
    source_df = load_dt(fp)

    filtered_df, contactctx = apply_contactprof(
        source_df,
        contactprof_val=contactprof_val,
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
        "contactctx": contactctx,
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
    contactprof_val=contactprof,
):
    fix_seed(seednum)

    data = build_training_data(
        fp=fp,
        dd=dd,
        drop_columns=drop_columns,
        contactprof_val=contactprof_val,
    )

    X = data["X"]
    y = data["y"]
    cat_cols = data["cat_cols"]
    num_cols = data["num_cols"]

    X_train, X_test, y_train, y_test = split_dt(X, y)

    X_fit, X_valid, y_fit, y_valid = split_dt(
        X_train,
        y_train,
        test_size=valsize,
        seed=42,
    )

    threshold_probe = mk_gb(mk_prep(cat_cols, num_cols))
    threshold_probe.fit(X_fit, y_fit)

    threshold_info = optimize_threshold(
        threshold_probe,
        X_valid,
        y_valid,
    )

    model = mk_gb(mk_prep(cat_cols, num_cols))
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
        contactctx=data["contactctx"],
    )

    return artifact, X_train, X_test, y_test, y_pred, y_proba, report, matrix


def train_gb(
    fp=None,
    dd=False,
    drop_columns=None,
    contactprof_val=contactprof,
):
    artifact, X_train, X_test, y_test, y_pred, y_proba, report, matrix = (
        train_gb_artifact(
            fp=fp,
            dd=dd,
            drop_columns=drop_columns,
            contactprof_val=contactprof_val,
        )
    )

    return {
        "metrics": artifact["metrics"],
        "report": report,
        "confusion_matrix": matrix,
        "feature_importance": artifact["feature_importance"],
        "excluded_columns": artifact["excluded_columns"],
        "input_cols": artifact["input_cols"],
        "contactctx": artifact["contactctx"],
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
    contactprof_val=contactprof,
):
    return train_gb(
        fp=fp,
        dd=dd,
        drop_columns=drop_columns,
        contactprof_val=contactprof_val,
    )


def load_mtr(model_artifact):
    return model_artifact.get("metrics", {})


def pred_client(client_data=None, model_artifact=None, cd=None):
    if client_data is None:
        client_data = cd

    if client_data is None:
        raise ValueError("pred_client requires client_data or cd.")

    if model_artifact is None:
        raise ValueError("pred_client requires model_artifact.")

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

    print("modelver:", modelver)
    print("defpath:", defpath)
    print("contactctx:", result["contactctx"])
    print("Используемые входы:", result["input_cols"])
    print("Исключенные поля:", result["excluded_columns"])
    print("Размер обучающей выборки:", result["X_train_shape"])
    print("Размер тестовой выборки:", result["X_test_shape"])
    print("Порог решения:", result["decision_threshold"])

    print("\nGradient Boosting:")
    print(result["metrics"])
    print(result["report"])
    print(result["confusion_matrix"])
    print(result["feature_importance"])
