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


def prep_xy(df, tgt="deposit", dd=False):
    dt = df.copy()

    if dd and "duration" in dt.columns:
        dt = dt.drop(columns=["duration"])

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


def eval_mdl(mdl, xt, yt, nm="Gradient Boosting"):
    yp = mdl.predict(xt)
    pr = mdl.predict_proba(xt)[:, 1]

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


def mk_artf(mdl, mtr, cat, num, dd=False):
    return {
        "model": mdl,
        "metrics": mtr,
        "feature_importance": feat_imp(mdl, n=20),
        "cat_cols": cat,
        "num_cols": num,
        "drop_duration": dd,
        "model_key": "gradient_boosting",
        "model_title": "Gradient Boosting",
        "target_mapping": {"no": 0, "yes": 1},
        "inverse_target_mapping": {0: "no", 1: "yes"},
    }


def train_gb_artifact(fp, dd=False):
    fix_seed(42)

    df = load_dt(fp)
    x, y, cat, num = prep_xy(df, dd=dd)
    pp = mk_prep(cat, num)
    xtr, xt, ytr, yt = split_dt(x, y)

    mdl = mk_gb(pp)
    mdl.fit(xtr, ytr)

    mtr, rpt, mx, yp, pr = eval_mdl(
        mdl,
        xt,
        yt,
        nm="Gradient Boosting",
    )

    art = mk_artf(
        mdl=mdl,
        mtr=mtr,
        cat=cat,
        num=num,
        dd=dd,
    )

    return art, xtr, xt, yt, yp, pr, rpt, mx


def train_gb(fp, dd=False):
    art, xtr, xt, yt, yp, pr, rpt, mx = train_gb_artifact(fp, dd=dd)
    mtr = art["metrics"]

    return {
        "metrics": mtr,
        "report": rpt,
        "confusion_matrix": mx,
        "feature_importance": art["feature_importance"],
        "X_train_shape": xtr.shape,
        "X_test_shape": xt.shape,
        "y_test": yt,
        "y_pred": yp,
        "y_proba": pr,
    }


def train_mdl(fp, dd=False):
    return train_gb(
        fp=fp,
        dd=dd,
    )


def load_mtr(model_artifact):
    art = model_artifact
    return art.get("metrics", {})


def pred_client(cd, model_artifact):
    art = model_artifact
    mdl = art["model"]
    inv = art["inverse_target_mapping"]
    cdf = pd.DataFrame([cd])

    yp = mdl.predict(cdf)[0]
    pr = mdl.predict_proba(cdf)[0, 1]

    return {
        "prediction": int(yp),
        "prediction_label": inv[int(yp)],
        "deposit_probability": float(pr),
    }


if __name__ == "__main__":
    fp = r"C:\Users\egors\OneDrive\Рабочий стол\GRADUATE\Kaggle Database\bank.csv"

    res = train_gb(
        fp=fp,
        dd=False,
    )

    print("Размер обучающей выборки:", res["X_train_shape"])
    print("Размер тестовой выборки:", res["X_test_shape"])
    print("\nGradient Boosting:")
    print(res["metrics"])
    print(res["report"])
    print(res["confusion_matrix"])
    print(res["feature_importance"])