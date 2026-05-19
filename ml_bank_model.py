import random

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


MODEL_DROP_COLUMNS = [
    "balance",
    "duration",
    "campaign",
    "pdays",
    "previous",
    "poutcome",
    "day",
]

MODEL_VERSION = "bank_deposit_normal_dialogue_v4"
NORMAL_DIALOGUE_MIN_DURATION = 180
NORMAL_DIALOGUE_MAX_DURATION = 720
CONTACT_PROFILE = "normal_dialogue"


def fix_seed(s=42):
    random.seed(s)
    np.random.seed(s)


def mk_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dt(fp):
    return pd.read_csv(fp, usecols=range(17))


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
    mask = duration.between(
        NORMAL_DIALOGUE_MIN_DURATION,
        NORMAL_DIALOGUE_MAX_DURATION,
        inclusive="both",
    )
    filtered = df[mask].copy()

    return filtered, {
        "contact_profile": contact_profile,
        "source_rows": len(df),
        "training_rows": len(filtered),
        "normal_duration_min": NORMAL_DIALOGUE_MIN_DURATION,
        "normal_duration_max": NORMAL_DIALOGUE_MAX_DURATION,
    }


def get_drop_columns(drop_columns=None, dd=False):
    if drop_columns is None:
        cols = list(MODEL_DROP_COLUMNS)
    else:
        cols = list(drop_columns)

    if dd and "duration" not in cols:
        cols.append("duration")

    return cols


def prep_xy(df, tgt="deposit", dd=False, drop_columns=None):
    dt = df.copy()

    cols_to_drop = [
        col for col in get_drop_columns(drop_columns=drop_columns, dd=dd)
        if col in dt.columns and col != tgt
    ]
    if cols_to_drop:
        dt = dt.drop(columns=cols_to_drop)

    y = dt[tgt].map({"no": 0, "yes": 1})
    x = dt.drop(columns=[tgt])

    cat = x.select_dtypes(include="object").columns.tolist()
    num = x.select_dtypes(exclude="object").columns.tolist()

    return x, y, cat, num


def mk_prep(cat, num):
    return ColumnTransformer(
        transformers=[
            ("cat", mk_ohe(), cat),
            ("num", "passthrough", num),
        ]
    )


def mk_gb(pp):
    return Pipeline([
        ("prep", pp),
        ("clf", GradientBoostingClassifier(
            n_estimators=180,
            learning_rate=0.05,
            max_depth=3,
            random_state=42,
        )),
    ])


def split_dt(x, y, ts=0.2, s=42):
    return train_test_split(
        x,
        y,
        test_size=ts,
        stratify=y,
        random_state=s,
    )


def eval_mdl(mdl, xt, yt, nm="Gradient Boosting", threshold=0.5):
    pr = mdl.predict_proba(xt)[:, 1]
    yp = (pr >= threshold).astype(int)

    mtr = {
        "model": nm,
        "accuracy": accuracy_score(yt, yp),
        "precision": precision_score(yt, yp),
        "recall": recall_score(yt, yp),
        "f1": f1_score(yt, yp),
        "roc_auc": roc_auc_score(yt, pr),
    }

    rpt = classification_report(yt, yp)
    mx = confusion_matrix(yt, yp)

    return mtr, rpt, mx, yp, pr


def threshold_metrics(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": float(threshold),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
    }


def optimize_threshold(mdl, xv, yv):
    pr = mdl.predict_proba(xv)[:, 1]
    thresholds = np.linspace(0.35, 0.65, 61)

    scores = [
        threshold_metrics(yv, pr, threshold)
        for threshold in thresholds
    ]

    return max(
        scores,
        key=lambda item: (item["accuracy"], item["f1"]),
    )


def feat_imp(mdl, n=20):
    pp = mdl.named_steps["prep"]
    clf = mdl.named_steps["clf"]

    fn = pp.get_feature_names_out()
    im = clf.feature_importances_

    fi = pd.DataFrame({
        "feature": fn,
        "importance": im,
    }).sort_values(by="importance", ascending=False)

    return fi.head(n)


def mk_artf(
    mdl,
    mtr,
    cat,
    num,
    dd=False,
    input_cols=None,
    excluded_columns=None,
    threshold_info=None,
    contact_context=None,
):
    threshold_info = threshold_info or {"threshold": 0.5}
    contact_context = contact_context or {}

    return {
        "model": mdl,
        "metrics": mtr,
        "feature_importance": feat_imp(mdl, n=20),
        "cat_cols": cat,
        "num_cols": num,
        "input_cols": input_cols or cat + num,
        "excluded_columns": excluded_columns or [],
        "drop_duration": dd,
        "model_version": MODEL_VERSION,
        "contact_context": contact_context,
        "decision_threshold": float(threshold_info["threshold"]),
        "threshold_validation_metrics": threshold_info,
        "model_key": "gradient_boosting",
        "model_title": "Gradient Boosting",
        "target_mapping": {"no": 0, "yes": 1},
        "inverse_target_mapping": {0: "no", 1: "yes"},
    }


def train_gb_artifact(
    fp,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    fix_seed(42)

    source_df = load_dt(fp)
    df, contact_context = apply_contact_profile(
        source_df,
        contact_profile=contact_profile,
    )

    excluded_columns = [
        col for col in get_drop_columns(drop_columns=drop_columns, dd=dd)
        if col in source_df.columns
    ]

    x, y, cat, num = prep_xy(
        df,
        dd=dd,
        drop_columns=drop_columns,
    )

    xtr, xt, ytr, yt = split_dt(x, y)
    xfit, xv, yfit, yv = split_dt(xtr, ytr, ts=0.25, s=43)

    threshold_probe = mk_gb(mk_prep(cat, num))
    threshold_probe.fit(xfit, yfit)
    threshold_info = optimize_threshold(threshold_probe, xv, yv)

    pp = mk_prep(cat, num)
    mdl = mk_gb(pp)
    mdl.fit(xtr, ytr)

    mtr, rpt, mx, yp, pr = eval_mdl(
        mdl,
        xt,
        yt,
        nm="Gradient Boosting",
        threshold=threshold_info["threshold"],
    )

    art = mk_artf(
        mdl=mdl,
        mtr=mtr,
        cat=cat,
        num=num,
        dd=dd,
        input_cols=x.columns.tolist(),
        excluded_columns=excluded_columns,
        threshold_info=threshold_info,
        contact_context=contact_context,
    )

    return art, xtr, xt, yt, yp, pr, rpt, mx


def train_gb(
    fp,
    dd=False,
    drop_columns=None,
    contact_profile=CONTACT_PROFILE,
):
    art, xtr, xt, yt, yp, pr, rpt, mx = train_gb_artifact(
        fp,
        dd=dd,
        drop_columns=drop_columns,
        contact_profile=contact_profile,
    )
    mtr = art["metrics"]

    return {
        "metrics": mtr,
        "report": rpt,
        "confusion_matrix": mx,
        "feature_importance": art["feature_importance"],
        "excluded_columns": art["excluded_columns"],
        "input_cols": art["input_cols"],
        "contact_context": art["contact_context"],
        "decision_threshold": art["decision_threshold"],
        "threshold_validation_metrics": art["threshold_validation_metrics"],
        "X_train_shape": xtr.shape,
        "X_test_shape": xt.shape,
        "y_test": yt,
        "y_pred": yp,
        "y_proba": pr,
    }


def train_mdl(
    fp,
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
    art = model_artifact
    return art.get("metrics", {})


def pred_client(cd, model_artifact):
    art = model_artifact
    mdl = art["model"]
    inv = art["inverse_target_mapping"]
    cat_cols = art.get("cat_cols", [])
    num_cols = art.get("num_cols", [])
    input_cols = art.get("input_cols") or cat_cols + num_cols
    threshold = art.get("decision_threshold", 0.5)

    cdf = pd.DataFrame([cd])

    for col in input_cols:
        if col not in cdf.columns:
            cdf[col] = "unknown" if col in cat_cols else 0

    for col in cat_cols:
        if col in cdf.columns:
            cdf[col] = cdf[col].where(cdf[col].notna(), "unknown").astype(str)

    for col in num_cols:
        if col in cdf.columns:
            cdf[col] = pd.to_numeric(cdf[col], errors="coerce").fillna(0)

    cdf = cdf[input_cols]

    pr = mdl.predict_proba(cdf)[0, 1]
    yp = int(pr >= threshold)

    return {
        "prediction": int(yp),
        "prediction_label": inv[int(yp)],
        "deposit_probability": float(pr),
    }


if __name__ == "__main__":
    from pathlib import Path

    fp = Path(__file__).resolve().parent / "Kaggle Database" / "bank.csv"

    res = train_gb(
        fp=str(fp),
    )

    print("Размер обучающей выборки:", res["X_train_shape"])
    print("Размер тестовой выборки:", res["X_test_shape"])
    print("Используемые входы:", res["input_cols"])
    print("Исключенные поля:", res["excluded_columns"])
    print("Контекст контакта:", res["contact_context"])
    print("Порог решения:", res["decision_threshold"])
    print("\nGradient Boosting:")
    print(res["metrics"])
    print(res["report"])
    print(res["confusion_matrix"])
    print(res["feature_importance"])
