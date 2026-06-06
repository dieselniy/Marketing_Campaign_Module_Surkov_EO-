import kagglehub
import numpy as npy
import pandas as pds
import plotly.express as px
import streamlit as slt
import kagglehub 
from kagglehub import KaggleDatasetAdapter
from sklearn.ensemble import RandomForestClassifier
from ml_bank_model import pred_client, train_gb_artifact
from src.auth import init_session_state, require_login, check_session


bank_daf = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "janiobachmann/bank-marketing-dataset",
    "bank.csv",
)
bank_source_columns = bank_daf.columns.tolist()

VAL_LABLS = {
    "job": {
        "admin.": "Администратор",
        "blue-collar": "Наемный рабочий",
        "entrepreneur": "Предприниматель",
        "housemaid": "Домработник",
        "management": "Менеджмент",
        "retired": "Пенсионер",
        "self-employed": "Самозанятый",
        "services": "Сфера услуг",
        "student": "Студент",
        "technician": "Технический специалист",
        "unemployed": "Безработный",
        "unknown": "Неизвестно"
    },
    "marital": {
        "divorced": "Разведён(а)",
        "married": "В браке",
        "single": "Не в браке"
    },
    "education": {
        "primary": "Начальное",
        "secondary": "Среднее",
        "tertiary": "Высшее",
        "unknown": "Неизвестно"
    },
    "default": {
        "no": "Нет",
        "yes": "Да"
    },
    "housing": {
        "no": "Нет",
        "yes": "Да"
    },
    "loan": {
        "no": "Нет",
        "yes": "Да"
    },
    "contact": {
        "cellular": "Сотовый телефон",
        "telephone": "Стационарный звонок",
        "unknown": "Неизвестно"
    },
    "month": {
        "jan": "Январь",
        "feb": "Февраль",
        "mar": "Март",
        "apr": "Апрель",
        "may": "Май",
        "jun": "Июнь",
        "jul": "Июль",
        "aug": "Август",
        "sep": "Сентябрь",
        "oct": "Октябрь",
        "nov": "Ноябрь",
        "dec": "Декабрь"
    },
    "poutcome": {
        "failure": "Неуспешно",
        "other": "Другое",
        "success": "Успешно",
        "unknown": "Неизвестно"
    },
    "deposit": {
        "no": "Нет",
        "yes": "Да"
    }
}

FEATURE_LABLS = {
    "age": "Возраст",
    "job": "Профессия",
    "marital": "Семейное положение",
    "education": "Образование",
    "default": "Дефолт по кредиту",
    "balance": "Баланс",
    "housing": "Ипотека",
    "loan": "Персональный заем",
    "contact": "Тип контакта",
    "day": "День контакта",
    "month": "Месяц",
    "duration": "Длительность звонка",
    "campaign": "Контактов в кампании",
    "pdays": "Дней после прошлого контакта",
    "previous": "Предыдущих контактов",
    "poutcome": "Результат прошлой кампании",
    "deposit": "Депозит",
}

COLUMN_LABLS = {
    "feature": "Признак",
    "value": "Значение",
    "importance": "Важность",
    "job": "Профессия",
    "marital": "Семейное положение",
    "education": "Образование",
    "deposit": "Депозит",
    "active_balance": "Активные средства",
    "avg_balance": "Средний баланс",
    "median_balance": "Медианный баланс",
    "clients": "Клиентов",
    "deposits": "Депозитов",
    "deposit_rate": "Конверсия в депозит",
    "low_balance_clients": "Клиентов с низким балансом",
    "low_balance_rate": "Риск низкого баланса",
    "low_balance_limit": "Порог низкого баланса",
}

bank_daf["deposit_flag"] = (
    bank_daf["deposit"].astype(str).str.strip().str.lower() == "yes"
).astype(int)

# --- Инициализация session_state ---
init_session_state()

# --- Восстановление session_id из query params ---
params = slt.query_params
if "session_id" in params:
    slt.session_state.session_id = params["session_id"][0]
    if check_session():
        slt.session_state.authenticated = True

# --- Проверка авторизации ---
# require_login()

# --- Контент страницы ---
slt.header("Аналитика рекламной кампании Банка-X")
slt.markdown(
    "Готовая статистика банковской кампании и факторов открытия депозита"
)

main_cont = slt.container(
    key="bank_main_cont",
    horizontal_alignment="center",
    vertical_alignment="center",
    border=False
)

with main_cont:
    overview_panel = slt.container(
        height=800,
        border=True
    )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    left_lane, right_lane = slt.columns([1, 1], gap="large")

    with left_lane:
        profile_panel = slt.container(
            border=True
        )

    with right_lane:
        segment_panel = slt.container(
            border=True
        )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    lower_left_lane, lower_right_lane = slt.columns([1, 1], gap="large")

    with lower_left_lane:
        conversion_panel = slt.container(
            border=True
        )

    with lower_right_lane:
        driver_panel = slt.container(
            border=True
        )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    balance_reason_panel = slt.container(
        height=760,
        border=True
    )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    predictor_panel = slt.container(
        border=True
    )



def loc_selctbox(label, field_name, options, key, index=0):
    labels = VAL_LABLS[field_name]
    return slt.selectbox(
        label,
        options,
        index=index,
        key=key,
        format_func=lambda value: labels.get(value, value)
    )


@slt.cache_resource
def bank_ml_artif(source_data):
    artifact, *_ = train_gb_artifact(df=source_data)
    return artifact


def rounded_table(data, digits=3):
    table = data.copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(digits)
    return table


def format_money_value(value):
    if pds.isna(value):
        return ""
    formatted_value = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{formatted_value} $"


def format_percent_value(value):
    if pds.isna(value):
        return ""
    formatted_value = f"{value * 100:,.2f}".rstrip("0").rstrip(".")
    return f"{formatted_value}%"


def format_integer_value(value):
    if pds.isna(value):
        return ""
    return f"{int(value):,}"


def format_decimal_value(value, digits=3):
    if pds.isna(value):
        return ""
    return f"{value:.{digits}f}".rstrip("0").rstrip(".")


def style_display_table(table):
    return table.style.set_properties(
        **{"text-align": "center"}
    ).set_table_styles([
        {"selector": "th", "props": [("text-align", "center")]},
        {"selector": "td", "props": [("text-align", "center")]},
    ])


def localize_value(field_name, value):
    if pds.isna(value):
        return ""

    labels = VAL_LABLS.get(str(field_name), {})
    return labels.get(str(value), value)


def localize_feature_name(feature_name):
    raw_feature = str(feature_name)
    if "__" in raw_feature:
        raw_feature = raw_feature.split("__", 1)[1]

    if raw_feature in FEATURE_LABLS:
        return FEATURE_LABLS[raw_feature]

    for field_name in sorted(VAL_LABLS.keys(), key=len, reverse=True):
        prefix = f"{field_name}_"
        if raw_feature.startswith(prefix):
            value = raw_feature[len(prefix):]
            field_label = FEATURE_LABLS.get(field_name, field_name)
            value_label = localize_value(field_name, value)
            return f"{field_label}: {value_label}"

    return raw_feature


def localize_categorical_columns(table, columns=None):
    localized = table.copy()
    selected_columns = columns or VAL_LABLS.keys()

    for column in selected_columns:
        if column in localized.columns:
            localized[column] = localized[column].map(
                lambda value: localize_value(column, value)
            )

    return localized


def rename_display_columns(table):
    return table.rename(columns={
        column: COLUMN_LABLS.get(column, column)
        for column in table.columns
    })


def format_money_columns(table, columns):
    formatted = table.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_money_value)
    return formatted


def format_percent_columns(table, columns):
    formatted = table.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_percent_value)
    return formatted


def format_integer_columns(table, columns):
    formatted = table.copy()
    for column in columns:
        if column in formatted.columns:
            formatted[column] = formatted[column].map(format_integer_value)
    return formatted


def factor_importance_display_table(data):
    table = rounded_table(data)

    if "feature" in table.columns:
        table["feature"] = table["feature"].map(localize_feature_name)
    if "importance" in table.columns:
        table["importance"] = table["importance"].map(format_decimal_value)

    return style_display_table(rename_display_columns(table))


def marital_deposit_display_table(data):
    table = rounded_table(data, digits=2)
    table = localize_categorical_columns(table, ["marital", "deposit"])
    table = format_money_columns(table, ["avg_balance", "median_balance"])
    table = format_integer_columns(table, ["clients"])
    return style_display_table(rename_display_columns(table))


def marital_conversion_display_table(data):
    table = rounded_table(data, digits=2)
    table = localize_categorical_columns(table, ["marital"])
    table = format_money_columns(table, ["avg_balance", "median_balance"])
    table = format_percent_columns(table, ["deposit_rate"])
    table = format_integer_columns(table, ["clients", "deposits"])
    return style_display_table(rename_display_columns(table))


def education_balance_display_table(data):
    table = rounded_table(data, digits=2)
    table = localize_categorical_columns(table, ["marital", "education"])
    table = format_money_columns(table, ["avg_balance", "median_balance"])
    table = format_integer_columns(table, ["clients"])
    return style_display_table(rename_display_columns(table))


def low_balance_segments_display_table(data):
    table = rounded_table(data, digits=2)

    if "feature" in table.columns and "value" in table.columns:
        raw_features = table["feature"].copy()
        table["value"] = [
            localize_value(feature, value)
            for feature, value in zip(raw_features, table["value"])
        ]
        table["feature"] = raw_features.map(localize_feature_name)

    table = format_money_columns(
        table,
        ["avg_balance", "median_balance", "low_balance_limit"],
    )
    table = format_percent_columns(table, ["low_balance_rate"])
    table = format_integer_columns(table, ["clients", "low_balance_clients"])
    return style_display_table(rename_display_columns(table))


def active_balance_display_table(data):
    table = rounded_table(data, digits=2)

    if "job" in table.columns:
        table["job"] = table["job"].map(
            lambda value: VAL_LABLS["job"].get(value, value)
        )

    for column in ["active_balance", "avg_balance", "median_balance"]:
        if column in table.columns:
            table[column] = table[column].map(format_money_value)

    if "deposit_rate" in table.columns:
        table["deposit_rate"] = table["deposit_rate"].map(format_percent_value)

    for column in ["clients", "deposits"]:
        if column in table.columns:
            table[column] = table[column].map(format_integer_value)

    table = rename_display_columns(table)

    return style_display_table(table)


@slt.cache_data
def build_bank_stats(daf, source_columns):
    work = daf[source_columns].copy()
    deposit_values = work["deposit"].astype(str).str.strip().str.lower()
    work["deposit_flag"] = deposit_values.map({
        "yes": 1,
        "no": 0,
    })

    if work["deposit_flag"].isna().any():
        encoded_values, _ = pds.factorize(work["deposit"])
        work["deposit_flag"] = encoded_values

    def numeric_from(column):
        return pds.to_numeric(
            work[column],
            errors="coerce",
        ).replace([npy.inf, -npy.inf], npy.nan)

    def fill_model_frame(frame):
        prepared = frame.replace([npy.inf, -npy.inf], npy.nan).copy()

        for column in prepared.columns:
            if pds.api.types.is_numeric_dtype(prepared[column]):
                median_value = prepared[column].median()
                fill_value = 0 if pds.isna(median_value) else median_value
                prepared[column] = prepared[column].fillna(fill_value)
            else:
                prepared[column] = prepared[column].fillna("")

        return pds.get_dummies(prepared)

    x_deposit = fill_model_frame(
        work.drop(columns=["deposit", "deposit_flag"], errors="ignore")
    )
    y_deposit = work["deposit_flag"]
    deposit_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    deposit_model.fit(x_deposit, y_deposit)
    deposit_factors = (
        pds.Series(
            deposit_model.feature_importances_,
            index=x_deposit.columns,
            name="importance",
        )
        .sort_values(ascending=False)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    balance_work = work.copy()
    balance_work["balance"] = numeric_from("balance").fillna(0)
    active_balance = (
        balance_work
        .groupby("job", as_index=False)
        .agg(
            active_balance=("balance", "sum"),
            avg_balance=("balance", "mean"),
            median_balance=("balance", "median"),
            clients=("job", "count"),
            deposits=("deposit_flag", "sum"),
        )
        .assign(
            deposit_rate=lambda data: npy.where(
                data["clients"] > 0,
                data["deposits"] / data["clients"],
                0,
            )
        )
        .sort_values(by="active_balance", ascending=False)
    )
    treemap_data = active_balance.copy()
    treemap_data["active_balance_for_plot"] = (
        treemap_data["active_balance"].clip(lower=0)
    )

    top_occupation = None
    if not active_balance.empty:
        top_occupation = active_balance.iloc[0]["job"]

    marital_balance_deposit = balance_work[
        ["job", "marital", "balance", "deposit", "deposit_flag"]
    ].copy()

    marital_deposit_summary = (
        marital_balance_deposit
        .groupby(["job", "marital", "deposit"], as_index=False)
        .agg(
            clients=("deposit_flag", "count"),
            avg_balance=("balance", "mean"),
            median_balance=("balance", "median"),
        )
        .sort_values(by=["job", "marital", "deposit"])
    )

    marital_conversion = (
        marital_balance_deposit
        .groupby("marital", as_index=False)
        .agg(
            clients=("deposit_flag", "count"),
            deposits=("deposit_flag", "sum"),
            avg_balance=("balance", "mean"),
            median_balance=("balance", "median"),
        )
        .assign(
            deposit_rate=lambda data: npy.where(
                data["clients"] > 0,
                data["deposits"] / data["clients"],
                0,
            )
        )
        .sort_values(by="deposit_rate", ascending=False)
    )

    education_marital_balance = (
        balance_work
        .groupby(["marital", "education"], as_index=False)
        .agg(
            median_balance=("balance", "median"),
            avg_balance=("balance", "mean"),
            clients=("balance", "count"),
        )
        .round(2)
        .sort_values(by=["marital", "education"])
    )

    low_balance_limit = balance_work["balance"].quantile(0.25)
    low_balance_work = work.copy()
    low_balance_work["balance"] = balance_work["balance"]
    low_balance_work["low_balance_flag"] = (
        low_balance_work["balance"] <= low_balance_limit
    ).astype(int)

    x_low_balance = fill_model_frame(
        low_balance_work.drop(
            columns=["balance", "low_balance_flag"],
            errors="ignore",
        )
    )
    y_low_balance = low_balance_work["low_balance_flag"]
    low_balance_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )
    low_balance_model.fit(x_low_balance, y_low_balance)
    low_balance_factors = (
        pds.Series(
            low_balance_model.feature_importances_,
            index=x_low_balance.columns,
            name="importance",
        )
        .sort_values(ascending=False)
        .head(10)
        .reset_index()
        .rename(columns={"index": "feature"})
    )

    low_balance_rows = []
    segment_columns = [
        column
        for column in [
            "job",
            "marital",
            "education",
            "default",
            "housing",
            "loan",
            "contact",
            "poutcome",
        ]
        if column in low_balance_work.columns
    ]

    for column in segment_columns:
        grouped = (
            low_balance_work
            .groupby(column, as_index=False)
            .agg(
                clients=("low_balance_flag", "count"),
                low_balance_clients=("low_balance_flag", "sum"),
                avg_balance=("balance", "mean"),
                median_balance=("balance", "median"),
            )
        )
        grouped = grouped[grouped["clients"] >= 20].copy()
        grouped["low_balance_rate"] = npy.where(
            grouped["clients"] > 0,
            grouped["low_balance_clients"] / grouped["clients"],
            0,
        )

        for _, row in grouped.iterrows():
            low_balance_rows.append({
                "feature": column,
                "value": row[column],
                "clients": int(row["clients"]),
                "low_balance_clients": int(row["low_balance_clients"]),
                "low_balance_rate": row["low_balance_rate"],
                "avg_balance": row["avg_balance"],
                "median_balance": row["median_balance"],
                "low_balance_limit": low_balance_limit,
            })

    low_balance_segments = pds.DataFrame(low_balance_rows)
    if not low_balance_segments.empty:
        low_balance_segments = low_balance_segments.sort_values(
            by=["low_balance_rate", "clients"],
            ascending=[False, False],
        )

    return {
        "deposit_factor_importance": deposit_factors,
        "occupation_active_balance": active_balance,
        "occupation_treemap_data": treemap_data,
        "top_active_balance_occupation": top_occupation,
        "marital_balance_deposit_data": marital_balance_deposit,
        "marital_deposit_summary": marital_deposit_summary,
        "marital_conversion": marital_conversion,
        "education_marital_balance": education_marital_balance,
        "low_balance_factor_importance": low_balance_factors,
        "low_balance_limit": low_balance_limit,
        "low_balance_segments": low_balance_segments,
    }


bank_stats = build_bank_stats(bank_daf, bank_source_columns)



with overview_panel:
    active_balance = bank_stats["occupation_active_balance"]

    slt.subheader("Активный баланс по профессиям")
    slt.dataframe(
        active_balance_display_table(active_balance),
        use_container_width=True,
        hide_index=True,
        height=300
    )

    if active_balance is not None and not active_balance.empty:
        treemap_data = active_balance.copy()
        treemap_data["Профессия"] = treemap_data["job"].map(
            lambda value: localize_value("job", value)
        )
        treemap_data["Активные средства"] = treemap_data["active_balance"]
        treemap_data["Активные средства для диаграммы"] = (
            treemap_data["active_balance"].clip(lower=0)
        )
        treemap_data["Активные средства, формат"] = treemap_data[
            "active_balance"
        ].map(format_money_value)
        treemap_data["Клиентов"] = treemap_data["clients"].map(
            format_integer_value
        )
        treemap_data["Депозитов"] = treemap_data["deposits"].map(
            format_integer_value
        )
        treemap_data["Конверсия"] = treemap_data["deposit_rate"].map(
            format_percent_value
        )

        fig = px.treemap(
            treemap_data,
            path=["Профессия"],
            values="Активные средства для диаграммы",
            custom_data=[
                "Активные средства, формат",
                "Клиентов",
                "Депозитов",
                "Конверсия",
            ],
            height=420,
        )
        fig.update_traces(
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Активные средства: %{customdata[0]}<br>"
                "Клиентов: %{customdata[1]}<br>"
                "Депозитов: %{customdata[2]}<br>"
                "Конверсия в депозит: %{customdata[3]}"
                "<extra></extra>"
            )
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        slt.plotly_chart(fig, use_container_width=True)


with profile_panel:
    slt.subheader("Все факторы открытия депозита с длительностью звонка")

    deposit_factors = bank_stats["deposit_factor_importance"]
    slt.dataframe(
        factor_importance_display_table(deposit_factors),
        use_container_width=True,
        hide_index=True,
        height=510
    )


with segment_panel:
    slt.subheader("Семейное положение, баланс и депозит")

    occupation_options = bank_stats["occupation_active_balance"]["job"].tolist()
    top_occupation = bank_stats["top_active_balance_occupation"]
    default_occupation_index = (
        occupation_options.index(top_occupation)
        if top_occupation in occupation_options
        else 0
    )

    selected_occupation = slt.selectbox(
        "Профессия",
        options=occupation_options,
        index=default_occupation_index,
        key="bank_marital_occupation",
        format_func=lambda value: VAL_LABLS["job"].get(value, value)
    )

    marital_balance_data = (
        bank_stats["marital_balance_deposit_data"]
        [bank_stats["marital_balance_deposit_data"]["job"] == selected_occupation]
        .copy()
    )

    marital_summary = (
        marital_balance_data
        .groupby(["marital", "deposit"], as_index=False)
        .agg(
            clients=("deposit_flag", "count"),
            avg_balance=("balance", "mean"),
            median_balance=("balance", "median"),
        )
        .sort_values(by=["marital", "deposit"])
    )

    slt.caption(
        "Выбранная профессия: "
        f"{VAL_LABLS['job'].get(selected_occupation, selected_occupation)}"
    )

    if marital_balance_data is not None and not marital_balance_data.empty:
        marital_plot_data = marital_balance_data.copy()
        marital_plot_data["Семейное положение"] = marital_plot_data[
            "marital"
        ].map(lambda value: localize_value("marital", value))
        marital_plot_data["Депозит"] = marital_plot_data["deposit"].map(
            lambda value: localize_value("deposit", value)
        )
        marital_plot_data["Баланс"] = marital_plot_data["balance"]
        marital_plot_data["Баланс, формат"] = marital_plot_data["balance"].map(
            format_money_value
        )

        fig = px.box(
            marital_plot_data,
            x="Семейное положение",
            y="Баланс",
            color="Депозит",
            points="all",
            custom_data=["Баланс, формат"],
            height=330,
            labels={
                "Семейное положение": "Семейное положение",
                "Баланс": "Баланс",
                "Депозит": "Депозит",
            },
        )
        fig.update_traces(
            hovertemplate=(
                "Семейное положение: %{x}<br>"
                "Баланс: %{customdata[0]}"
                "<extra>%{fullData.name}</extra>"
            )
        )
        fig.update_yaxes(
            title_text="Баланс",
            tickformat=",.0f",
            ticksuffix=" $",
            separatethousands=True,
        )
        fig.update_layout(legend_title_text="Депозит")
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        slt.plotly_chart(fig, use_container_width=True)

    slt.dataframe(
        marital_deposit_display_table(marital_summary),
        use_container_width=True,
        hide_index=True,
        height=190
    )


with conversion_panel:
    slt.subheader("Влияние на депозит семейного положения")

    marital_conversion = bank_stats["marital_conversion"]
    slt.dataframe(
        marital_conversion_display_table(marital_conversion),
        use_container_width=True,
        hide_index=True,
        height=210
    )

    if marital_conversion is not None and not marital_conversion.empty:
        conversion_plot_data = marital_conversion.copy()
        conversion_plot_data["Семейное положение"] = conversion_plot_data[
            "marital"
        ].map(lambda value: localize_value("marital", value))
        conversion_plot_data["Конверсия в депозит"] = conversion_plot_data[
            "deposit_rate"
        ]
        conversion_plot_data["Конверсия, формат"] = conversion_plot_data[
            "deposit_rate"
        ].map(format_percent_value)
        conversion_plot_data["Средний баланс"] = conversion_plot_data[
            "avg_balance"
        ].map(format_money_value)
        conversion_plot_data["Медианный баланс"] = conversion_plot_data[
            "median_balance"
        ].map(format_money_value)
        conversion_plot_data["Клиентов"] = conversion_plot_data["clients"].map(
            format_integer_value
        )
        conversion_plot_data["Депозитов"] = conversion_plot_data[
            "deposits"
        ].map(format_integer_value)

        fig = px.bar(
            conversion_plot_data,
            x="Семейное положение",
            y="Конверсия в депозит",
            color="Семейное положение",
            custom_data=[
                "Конверсия, формат",
                "Средний баланс",
                "Медианный баланс",
                "Клиентов",
                "Депозитов",
            ],
            labels={
                "Семейное положение": "Семейное положение",
                "Конверсия в депозит": "Конверсия в депозит",
            },
            height=350,
        )
        fig.update_traces(
            hovertemplate=(
                "Семейное положение: %{x}<br>"
                "Конверсия в депозит: %{customdata[0]}<br>"
                "Средний баланс: %{customdata[1]}<br>"
                "Медианный баланс: %{customdata[2]}<br>"
                "Клиентов: %{customdata[3]}<br>"
                "Депозитов: %{customdata[4]}"
                "<extra></extra>"
            )
        )
        fig.update_yaxes(title_text="Конверсия в депозит", tickformat=".0%")
        fig.update_layout(
            legend_title_text="Семейное положение",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        slt.plotly_chart(fig, use_container_width=True)


with driver_panel:
    slt.subheader("Баланс по образованию и семейному положению")

    education_balance = bank_stats["education_marital_balance"]
    slt.dataframe(
        education_balance_display_table(education_balance),
        use_container_width=True,
        hide_index=True,
        height=220
    )

    if education_balance is not None and not education_balance.empty:
        segment_data = education_balance.copy()
        segment_data["Семейное положение"] = segment_data["marital"].map(
            lambda value: localize_value("marital", value)
        )
        segment_data["Образование"] = segment_data["education"].map(
            lambda value: localize_value("education", value)
        )
        segment_data["Сегмент"] = (
            segment_data["Семейное положение"].astype(str)
            + " / "
            + segment_data["Образование"].astype(str)
        )
        segment_data["Медианный баланс"] = segment_data["median_balance"]
        segment_data["Медианный баланс, формат"] = segment_data[
            "median_balance"
        ].map(format_money_value)
        segment_data["Средний баланс"] = segment_data["avg_balance"].map(
            format_money_value
        )
        segment_data["Клиентов"] = segment_data["clients"].map(
            format_integer_value
        )

        fig = px.bar(
            segment_data,
            x="Сегмент",
            y="Медианный баланс",
            color="Семейное положение",
            custom_data=[
                "Медианный баланс, формат",
                "Средний баланс",
                "Клиентов",
            ],
            labels={
                "Сегмент": "Сегмент",
                "Медианный баланс": "Медианный баланс",
                "Семейное положение": "Семейное положение",
            },
            height=360,
        )
        fig.update_traces(
            hovertemplate=(
                "Сегмент: %{x}<br>"
                "Медианный баланс: %{customdata[0]}<br>"
                "Средний баланс: %{customdata[1]}<br>"
                "Клиентов: %{customdata[2]}"
                "<extra></extra>"
            )
        )
        fig.update_yaxes(
            title_text="Медианный баланс",
            tickformat=",.0f",
            ticksuffix=" $",
            separatethousands=True,
        )
        fig.update_layout(
            legend_title_text="Семейное положение",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        slt.plotly_chart(fig, use_container_width=True)


with balance_reason_panel:
    low_balance_factors = bank_stats["low_balance_factor_importance"]
    low_balance_limit = bank_stats["low_balance_limit"]
    low_balance_segments = bank_stats["low_balance_segments"]

    slt.subheader(f"Факторы низкого баланса")
    slt.dataframe(
        factor_importance_display_table(low_balance_factors),
        use_container_width=True,
        hide_index=True,
        height=230
    )

    if low_balance_factors is not None and not low_balance_factors.empty:
        low_balance_factor_plot = low_balance_factors.copy()
        low_balance_factor_plot["Признак"] = low_balance_factor_plot[
            "feature"
        ].map(localize_feature_name)
        low_balance_factor_plot["Важность"] = low_balance_factor_plot[
            "importance"
        ]

        fig = px.bar(
            low_balance_factor_plot,
            x="Признак",
            y="Важность",
            color="Признак",
            labels={"Признак": "Признак", "Важность": "Важность"},
            height=330,
        )
        fig.update_traces(
            hovertemplate=(
                "Признак: %{x}<br>"
                "Важность: %{y:.3f}"
                "<extra></extra>"
            )
        )
        fig.update_layout(
            legend_title_text="Признак",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        slt.plotly_chart(fig, use_container_width=True)

    slt.markdown("**Сегменты с риском низкого баланса**")
    slt.dataframe(
        low_balance_segments_display_table(low_balance_segments),
        use_container_width=True,
        hide_index=True,
        height=250
    )

    if low_balance_segments is not None and not low_balance_segments.empty:
        segment_data = low_balance_segments.head(15).copy()
        segment_data["Признак"] = segment_data["feature"].map(
            localize_feature_name
        )
        segment_data["Значение"] = [
            localize_value(feature, value)
            for feature, value in zip(segment_data["feature"], segment_data["value"])
        ]
        segment_data["Сегмент"] = (
            segment_data["Признак"].astype(str)
            + " = "
            + segment_data["Значение"].astype(str)
        )
        segment_data["Риск низкого баланса"] = segment_data[
            "low_balance_rate"
        ]
        segment_data["Риск, формат"] = segment_data["low_balance_rate"].map(
            format_percent_value
        )
        segment_data["Средний баланс"] = segment_data["avg_balance"].map(
            format_money_value
        )
        segment_data["Медианный баланс"] = segment_data[
            "median_balance"
        ].map(format_money_value)
        segment_data["Порог низкого баланса"] = segment_data[
            "low_balance_limit"
        ].map(format_money_value)
        segment_data["Клиентов"] = segment_data["clients"].map(
            format_integer_value
        )
        segment_data["Клиентов с низким балансом"] = segment_data[
            "low_balance_clients"
        ].map(format_integer_value)

        fig = px.bar(
            segment_data,
            x="Сегмент",
            y="Риск низкого баланса",
            color="Признак",
            custom_data=[
                "Риск, формат",
                "Средний баланс",
                "Медианный баланс",
                "Порог низкого баланса",
                "Клиентов",
                "Клиентов с низким балансом",
            ],
            labels={
                "Сегмент": "Сегмент",
                "Риск низкого баланса": "Риск низкого баланса",
                "Признак": "Признак",
            },
            height=360,
        )
        fig.update_traces(
            hovertemplate=(
                "Сегмент: %{x}<br>"
                "Риск низкого баланса: %{customdata[0]}<br>"
                "Средний баланс: %{customdata[1]}<br>"
                "Медианный баланс: %{customdata[2]}<br>"
                "Порог низкого баланса: %{customdata[3]}<br>"
                "Клиентов: %{customdata[4]}<br>"
                "Клиентов с низким балансом: %{customdata[5]}"
                "<extra></extra>"
            )
        )
        fig.update_yaxes(title_text="Риск низкого баланса", tickformat=".0%")
        fig.update_layout(
            legend_title_text="Признак",
            margin=dict(l=10, r=10, t=10, b=10),
        )
        slt.plotly_chart(fig, use_container_width=True)


with predictor_panel:
    slt.subheader("Прогноз открытия депозита")
    model_artifact = bank_ml_artif(bank_daf[bank_source_columns])
    slt.caption(
        "Заполните параметры, которые известны до контакта с клиентом. "
        "Модель Gradient Boosting не использует баланс, длительность звонка "
        "день контакта и историю прошлых контактов. Прогноз строится при "
        "допущении, что контакт прошёл по нормальному сценарию: диалог длился "
        "примерно 3-12 минут."
    )

    with slt.form("bank_deposit_prediction_form"):
        client_col_1, client_col_2, client_col_3 = slt.columns(3, gap="large")

        with client_col_1:
            age = slt.number_input(
                "Возраст",
                min_value=18,
                max_value=100,
                value=35,
                key="predict_age"
            )
            job = loc_selctbox(
                "Профессия",
                "job",
                [
                    "admin.", "blue-collar", "entrepreneur", "housemaid",
                    "management", "retired", "self-employed", "services",
                    "student", "technician", "unemployed", "unknown"
                ],
                key="predict_job"
            )
            marital = loc_selctbox(
                "Семейное положение",
                "marital",
                ["divorced", "married", "single"],
                key="predict_marital"
            )
            education = loc_selctbox(
                "Образование",
                "education",
                ["primary", "secondary", "tertiary", "unknown"],
                index=1,
                key="predict_education"
            )

        with client_col_2:
            default = loc_selctbox(
                "Дефолт по кредиту",
                "default",
                ["no", "yes"],
                key="predict_default"
            )
            housing = loc_selctbox(
                "Ипотека",
                "housing",
                ["no", "yes"],
                index=1,
                key="predict_housing"
            )
            loan = loc_selctbox(
                "Персональный заем",
                "loan",
                ["no", "yes"],
                key="predict_loan"
            )

        with client_col_3:
            contact = loc_selctbox(
                "Тип контакта",
                "contact",
                ["cellular", "telephone", "unknown"],
                key="predict_contact"
            )
            month = loc_selctbox(
                "Месяц",
                "month",
                [
                    "jan", "feb", "mar", "apr", "may", "jun",
                    "jul", "aug", "sep", "oct", "nov", "dec"
                ],
                index=4,
                key="predict_month"
            )

        submitted_prediction = slt.form_submit_button(
            "Сделать прогноз",
            type="primary",
            use_container_width=True
        )

    if submitted_prediction:
        client = {
            "age": age,
            "job": job,
            "marital": marital,
            "education": education,
            "default": default,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "month": month
        }

        result = pred_client(
            client_data=client,
            model_artifact=model_artifact
        )

        probability = result["deposit_probability"]
        label = "Да" if result["prediction_label"] == "yes" else "Нет"

        result_col_1, result_col_2 = slt.columns([1, 2], gap="large")

        with result_col_1:
            slt.metric("Вероятность открытия депозита", f"{probability:.2%}")
            slt.write("Прогноз модели:", label)

        with result_col_2:
            slt.progress(min(max(probability, 0), 1))

            if probability >= 0.75:
                slt.success("Высокий приоритет клиента")
            elif probability >= 0.45:
                slt.warning("Средний приоритет клиента")
            else:
                slt.error("Низкий приоритет клиента")
