# page2.py
import numpy as npy
import pandas as pds
import plotly.express as px
import streamlit as slt
from sklearn.ensemble import RandomForestClassifier
from ml_bank_model import pred_client, train_gb_artifact
from src.auth import init_session_state, require_login, check_session

file_path = "Kaggle Database/bank.csv"
bank_daf = pds.read_csv(file_path)
bank_source_columns = bank_daf.columns.tolist()
init_session_state()

VAL_LABLS = {
    "job": {
        "admin.": "Администратор",
        "blue-collar": "Рабочий",
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
    }
}

bank_daf["deposit_flag"] = (bank_daf["deposit"] == "yes").astype(int)

slt.set_page_config(page_title="Банк-X: рекламные контакты", layout="wide")

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
            height=600,
            border=True
        )

    with right_lane:
        segment_panel = slt.container(
            height=600,
            border=True
        )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    lower_left_lane, lower_right_lane = slt.columns([1, 1], gap="large")

    with lower_left_lane:
        conversion_panel = slt.container(
            height=500,
            border=True
        )

    with lower_right_lane:
        driver_panel = slt.container(
            height=500,
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
def bank_ml_artif(path):
    artifact, *_ = train_gb_artifact(path)
    return artifact


def rounded_table(data, digits=3):
    table = data.copy()
    numeric_columns = table.select_dtypes(include="number").columns
    table[numeric_columns] = table[numeric_columns].round(digits)
    return table


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
    treemap_data = treemap_data.rename(columns={
        "job": "Occupation",
        "active_balance": "Active Balance",
        "active_balance_for_plot": "Active Balance For Plot",
    })

    top_occupation = None
    if not active_balance.empty:
        top_occupation = active_balance.iloc[0]["job"]

    marital_work = balance_work[["marital", "balance", "deposit", "deposit_flag", "job"]].copy()
    if top_occupation is not None:
        marital_top_occupation = marital_work[marital_work["job"] == top_occupation].copy()
    else:
        marital_top_occupation = marital_work.copy()

    marital_balance_deposit = marital_top_occupation[
        ["marital", "balance", "deposit", "deposit_flag"]
    ].copy()

    marital_deposit_summary = (
        marital_balance_deposit
        .groupby(["marital", "deposit"], as_index=False)
        .agg(
            clients=("deposit_flag", "count"),
            avg_balance=("balance", "mean"),
            median_balance=("balance", "median"),
        )
        .sort_values(by=["marital", "deposit"])
    )

    marital_conversion = (
        marital_work
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
    treemap_data = bank_stats["occupation_treemap_data"]

    slt.subheader("Активный баланс по профессиям")
    slt.dataframe(
        rounded_table(active_balance),
        use_container_width=True,
        hide_index=True,
        height=300
    )

    if treemap_data is not None and not treemap_data.empty:
        fig = px.treemap(
            treemap_data,
            path=["Occupation"],
            values="Active Balance For Plot",
            hover_data=["Occupation", "Active Balance"],
            height=420,
        )
        fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
        slt.plotly_chart(fig, use_container_width=True)


with profile_panel:
    slt.subheader("Все факторы открытия депозита с duration")

    deposit_factors = bank_stats["deposit_factor_importance"]
    slt.dataframe(
        rounded_table(deposit_factors),
        use_container_width=True,
        hide_index=True,
        height=510
    )


with segment_panel:
    slt.subheader("Семейное положение, баланс и депозит")

    marital_balance_data = bank_stats["marital_balance_deposit_data"]
    marital_summary = bank_stats["marital_deposit_summary"]
    top_occupation = bank_stats["top_active_balance_occupation"]

    if top_occupation is not None:
        slt.caption(f"Сегмент профессии с максимальным активным балансом: {top_occupation}")

    if marital_balance_data is not None and not marital_balance_data.empty:
        fig = px.box(
            marital_balance_data,
            x="marital",
            y="balance",
            color="deposit",
            points="all",
            height=330,
        )
        fig.update_layout(margin=dict(l=10, r=10, t=10, b=10))
        slt.plotly_chart(fig, use_container_width=True)

    slt.dataframe(
        rounded_table(marital_summary),
        use_container_width=True,
        hide_index=True,
        height=190
    )


with conversion_panel:
    slt.subheader("Конверсия в депозит по семейному положению")

    marital_conversion = bank_stats["marital_conversion"]
    slt.dataframe(
        rounded_table(marital_conversion),
        use_container_width=True,
        hide_index=True,
        height=210
    )

    if marital_conversion is not None and not marital_conversion.empty:
        slt.bar_chart(
            marital_conversion,
            x="marital",
            y="deposit_rate",
            color="marital"
        )


with driver_panel:
    slt.subheader("Баланс по образованию и семейному положению")

    education_balance = bank_stats["education_marital_balance"]
    slt.dataframe(
        rounded_table(education_balance),
        use_container_width=True,
        hide_index=True,
        height=220
    )

    if education_balance is not None and not education_balance.empty:
        segment_data = education_balance.copy()
        segment_data["segment"] = (
            segment_data["marital"].astype(str)
            + " / "
            + segment_data["education"].astype(str)
        )
        slt.bar_chart(
            segment_data,
            x="segment",
            y="median_balance",
            color="marital"
        )


with balance_reason_panel:
    low_balance_factors = bank_stats["low_balance_factor_importance"]
    low_balance_limit = bank_stats["low_balance_limit"]
    low_balance_segments = bank_stats["low_balance_segments"]

    slt.subheader(f"Факторы низкого баланса: порог {low_balance_limit:.2f}")
    slt.dataframe(
        rounded_table(low_balance_factors),
        use_container_width=True,
        hide_index=True,
        height=230
    )

    if low_balance_factors is not None and not low_balance_factors.empty:
        slt.bar_chart(
            low_balance_factors,
            x="feature",
            y="importance",
            color="feature"
        )

    slt.markdown("**Сегменты с риском низкого баланса**")
    slt.dataframe(
        rounded_table(low_balance_segments),
        use_container_width=True,
        hide_index=True,
        height=250
    )

    if low_balance_segments is not None and not low_balance_segments.empty:
        segment_data = low_balance_segments.head(15).copy()
        segment_data["segment"] = (
            segment_data["feature"].astype(str)
            + " = "
            + segment_data["value"].astype(str)
        )
        slt.bar_chart(
            segment_data,
            x="segment",
            y="low_balance_rate",
            color="feature"
        )


with predictor_panel:
    slt.subheader("Прогноз открытия депозита")
    model_artifact = bank_ml_artif(str(file_path))
    slt.caption(
        "Заполните параметры клиента и оцените вероятность положительного отклика "
        "по обученной модели Gradient Boosting."
    )

    with slt.form("bank_deposit_prediction_form"):
        client_col_1, client_col_2, client_col_3, client_col_4 = slt.columns(4, gap="large")

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
            balance = slt.number_input(
                "Баланс (в долларах)",
                value=1000,
                key="predict_balance"
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
            day = slt.number_input(
                "День контакта",
                min_value=1,
                max_value=31,
                value=15,
                key="predict_day"
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
            duration = slt.number_input(
                "Длительность звонка",
                min_value=0,
                value=300,
                key="predict_duration"
            )

        with client_col_4:
            campaign = slt.number_input(
                "Контактов в кампании",
                min_value=1,
                value=2,
                key="predict_campaign"
            )
            pdays = slt.number_input(
                "Дней после прошлого контакта",
                value=-1,
                key="predict_pdays"
            )
            previous = slt.number_input(
                "Прошлых контактов",
                min_value=0,
                value=0,
                key="predict_previous"
            )
            poutcome = loc_selctbox(
                "Результат прошлой кампании",
                "poutcome",
                ["failure", "other", "success", "unknown"],
                index=3,
                key="predict_poutcome"
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
            "balance": balance,
            "housing": housing,
            "loan": loan,
            "contact": contact,
            "day": day,
            "month": month,
            "duration": duration,
            "campaign": campaign,
            "pdays": pdays,
            "previous": previous,
            "poutcome": poutcome
        }

        result = pred_client(
            cd=client,
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
