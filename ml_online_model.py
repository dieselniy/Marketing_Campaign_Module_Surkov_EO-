import random
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


FT = [
    "month",
    "day",
    "campaign_number",
    "user_engagement",
    "banner",
    "placement",
    "displays",
    "cost",
    "clicks",
    "revenue",
]

FORECAST_FT = [
    "month",
    "day",
    "banner",
    "placement",
    "cost",
]

FORECAST_TARGETS = [
    "displays",
    "clicks",
    "revenue",
]


def fix_seed(s=42):
    random.seed(s)
    np.random.seed(s)


def mk_ohe():
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_dt(fp):
    return pd.read_csv(fp, usecols=range(12))


def add_roas(df):
    dt = df.copy()
    dt["roas"] = np.where(
        dt["cost"] > 0,
        dt["post_click_sales_amount"] / dt["cost"],
        0,
    )
    dt["roas_log"] = np.log1p(dt["roas"])
    return dt


def prep_xy(df):
    dt = add_roas(df)
    x = dt[FT].copy()
    y = dt["roas_log"]

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


def mk_gbr(pp):
    return Pipeline([
        ("prep", pp),
        ("reg", GradientBoostingRegressor(
            n_estimators=220,
            learning_rate=0.05,
            max_depth=3,
            min_samples_leaf=8,
            random_state=42,
        )),
    ])


def split_dt(x, y, ts=0.2, s=42):
    return train_test_split(
        x,
        y,
        test_size=ts,
        random_state=s,
    )


def eval_reg(mdl, xt, yt):
    pr_log = mdl.predict(xt)
    yt_roas = np.expm1(yt)
    pr_roas = np.maximum(np.expm1(pr_log), 0)

    return {
        "target": "ROAS",
        "mae": mean_absolute_error(yt_roas, pr_roas),
        "rmse": mean_squared_error(yt_roas, pr_roas) ** 0.5,
        "r2": r2_score(yt_roas, pr_roas),
        "mae_log": mean_absolute_error(yt, pr_log),
        "r2_log": r2_score(yt, pr_log),
    }, yt_roas, pr_roas


def feat_imp(mdl, n=20):
    pp = mdl.named_steps["prep"]
    reg = mdl.named_steps["reg"]

    fn = pp.get_feature_names_out()
    im = reg.feature_importances_

    fi = pd.DataFrame({
        "feature": fn,
        "importance": im,
    }).sort_values(by="importance", ascending=False)

    return fi.head(n)


def mk_artf(mdl, mtr, cat, num):
    return {
        "model": mdl,
        "metrics": mtr,
        "feature_importance": feat_imp(mdl),
        "target": "roas",
        "target_formula": "post_click_sales_amount / cost",
        "features": FT,
        "cat_cols": cat,
        "num_cols": num,
    }


def train_roas(fp):
    fix_seed(42)

    df = load_dt(fp)
    x, y, cat, num = prep_xy(df)
    pp = mk_prep(cat, num)
    xtr, xt, ytr, yt = split_dt(x, y)

    mdl = mk_gbr(pp)
    mdl.fit(xtr, ytr)

    mtr, yt_roas, pr_roas = eval_reg(mdl, xt, yt)
    art = mk_artf(mdl, mtr, cat, num)

    return {
        "metrics": mtr,
        "feature_importance": art["feature_importance"],
        "artifact": art,
        "X_train_shape": xtr.shape,
        "X_test_shape": xt.shape,
        "y_test": yt_roas,
        "y_pred": pr_roas,
    }


def train_mdl(fp):
    return train_roas(fp=fp)


def load_mtr(model_artifact):
    art = model_artifact
    return art.get("metrics", {})


def pred_roas(cd, model_artifact):
    art = model_artifact
    mdl = art["model"]
    cdf = pd.DataFrame([cd])

    pr_log = mdl.predict(cdf)[0]
    roas = max(float(np.expm1(pr_log)), 0)

    return {
        "roas": roas,
        "sales_per_dollar": roas,
        "target": "ROAS",
    }


def safe_div(num, den, default=0.0):
    if den is None or den <= 0:
        return default
    return float(num / den)


def prep_forecast_xy(df):
    dt = add_roas(df)
    dt = dt[dt["cost"] > 0].copy()
    dt["displays_log"] = np.log1p(dt["displays"])
    dt["clicks_log"] = np.log1p(dt["clicks"])
    dt["sales_log"] = np.log1p(dt["post_click_sales_amount"])

    x = dt[FORECAST_FT].copy()
    y = dt[FORECAST_TARGETS].copy()

    cat = x.select_dtypes(include="object").columns.tolist()
    num = x.select_dtypes(exclude="object").columns.tolist()

    return x, y, cat, num


def mk_campaign_forecast_model(cat, num):
    pp = mk_prep(cat, num)
    base = GradientBoostingRegressor(
        n_estimators=240,
        learning_rate=0.04,
        max_depth=3,
        min_samples_leaf=10,
        random_state=42,
    )

    return Pipeline([
        ("prep", pp),
        ("reg", MultiOutputRegressor(base)),
    ])


def train_campaign_forecast(df):
    dt = df[df["cost"] > 0].copy()
    dt = dt.replace([np.inf, -np.inf], np.nan)

    global_stats = calc_campaign_rates(dt)

    return {
        "history": dt,
        "global_stats": global_stats,
        "features": ["month", "day", "banner", "placement", "budget"],
        "targets": FORECAST_TARGETS,
        "target_formula": "budget-limited funnel: CPM -> CTR -> revenue per click",
    }


def calc_campaign_rates(dt):
    displays = float(dt["displays"].sum())
    clicks = float(dt["clicks"].sum())
    cost = float(dt["cost"].sum())
    revenue = float(dt["revenue"].sum())

    cpm = safe_div(cost * 1000, displays)
    ctr = min(max(safe_div(clicks, displays), 0.0), 1.0)
    revenue_per_click = max(safe_div(revenue, clicks), 0.0)

    daily_cost_reference = float(dt["cost"].median()) if not dt.empty else 0.0
    if not np.isfinite(daily_cost_reference) or daily_cost_reference <= 0:
        daily_cost_reference = max(safe_div(cost, max(len(dt), 1)), 1.0)

    return {
        "cpm": max(cpm, 0.01),
        "ctr": ctr,
        "revenue_per_click": revenue_per_click,
        "daily_cost_reference": daily_cost_reference,
        "rows": int(len(dt)),
        "cost": cost,
        "displays": displays,
        "clicks": clicks,
        "revenue": revenue,
    }


def rate_confidence(rates):
    cost_score = min(safe_div(rates["cost"], 100.0), 1.0)
    display_score = min(safe_div(rates["displays"], 10000.0), 1.0)
    click_score = min(safe_div(rates["clicks"], 100.0), 1.0)
    row_score = min(safe_div(rates["rows"], 100.0), 1.0)
    return max((cost_score + display_score + click_score + row_score) / 4, 0.02)


def blend_campaign_rates(candidates, global_stats):
    weighted = {
        "cpm": 0.0,
        "ctr": 0.0,
        "revenue_per_click": 0.0,
        "daily_cost_reference": 0.0,
    }
    total_weight = 0.0
    total_rows = 0

    for rates, priority in candidates:
        confidence = rate_confidence(rates)
        weight = confidence * priority
        total_weight += weight
        total_rows += rates["rows"]
        for key in weighted:
            weighted[key] += rates[key] * weight

    if total_weight <= 0:
        blended = global_stats.copy()
    else:
        blended = {
            key: value / total_weight
            for key, value in weighted.items()
        }

    min_ctr = max(global_stats["ctr"] * 0.25, 0.001)
    min_revenue_per_click = max(global_stats["revenue_per_click"] * 0.25, 0.01)

    blended["cpm"] = max(blended["cpm"], 0.01)
    blended["ctr"] = min(max(blended["ctr"], min_ctr), 1.0)
    blended["revenue_per_click"] = max(
        blended["revenue_per_click"],
        min_revenue_per_click,
    )
    blended["daily_cost_reference"] = max(blended["daily_cost_reference"], 0.01)
    blended["rows"] = total_rows

    return blended


def select_campaign_rates(history, global_stats, day):
    candidates = [(global_stats, 1.0)]

    exact = history[
        (history["banner"] == day["banner"])
        & (history["placement"] == day["placement"])
        & (history["month"] == day["month"])
    ]
    if len(exact) >= 10 and exact["cost"].sum() > 0:
        candidates.append((calc_campaign_rates(exact), 4.0))

    broader = history[
        (history["banner"] == day["banner"])
        & (history["placement"] == day["placement"])
    ]
    if len(broader) >= 10 and broader["cost"].sum() > 0:
        candidates.append((calc_campaign_rates(broader), 2.5))

    placement = history[history["placement"] == day["placement"]]
    if len(placement) >= 10 and placement["cost"].sum() > 0:
        candidates.append((calc_campaign_rates(placement), 1.5))

    return blend_campaign_rates(candidates, global_stats)


def predict_campaign_forecast(model_artifact, campaign_days, budget):
    if budget <= 0:
        raise ValueError("Budget must be greater than zero.")

    if not campaign_days:
        raise ValueError("Campaign period must contain at least one day.")

    if "history" not in model_artifact or "global_stats" not in model_artifact:
        raise ValueError(
            "Campaign forecast artifact is outdated. Restart Streamlit or clear cache."
        )

    history = model_artifact["history"]
    global_stats = model_artifact["global_stats"]
    daily_budget = budget / len(campaign_days)

    expected_displays = 0.0
    expected_clicks = 0.0
    expected_sales = 0.0
    effective_budget = 0.0
    rate_rows = []

    for day in campaign_days:
        rates = select_campaign_rates(history, global_stats, day)
        pacing_factor = daily_budget / (daily_budget + rates["daily_cost_reference"])
        day_effective_budget = daily_budget * pacing_factor
        day_displays = day_effective_budget / rates["cpm"] * 1000
        day_clicks = min(day_displays * rates["ctr"], day_displays)
        day_sales = day_clicks * rates["revenue_per_click"]

        expected_displays += day_displays
        expected_clicks += day_clicks
        expected_sales += day_sales
        effective_budget += day_effective_budget
        rate_rows.append(rates["rows"])

    avg_roas = expected_sales / budget
    roi_percent = ((expected_sales - budget) / budget) * 100

    return {
        "expected_displays": expected_displays,
        "expected_clicks": expected_clicks,
        "roas": avg_roas,
        "expected_sales": expected_sales,
        "roi_percent": roi_percent,
        "duration_days": len(campaign_days),
        "daily_budget": daily_budget,
        "effective_budget": effective_budget,
        "avg_history_rows": float(np.mean(rate_rows)) if rate_rows else 0.0,
    }


if __name__ == "__main__":
    fp = kagglehub.dataset_download(
        "naniruddhan/online-advertising-digital-marketing-data",
        path="online_advertising_performance_data.csv",
    )
    res = train_roas(fp=fp)
    print("Основная метрика: ROAS = post_click_sales_amount / cost")
    print("Размер обучающей выборки:", res["X_train_shape"])
    print("Размер тестовой выборки:", res["X_test_shape"])
    print(res["metrics"])
    print(res["feature_importance"])
