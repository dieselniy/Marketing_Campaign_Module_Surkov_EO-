import numpy as npy
import pandas as pds
import plotly.graph_objects as go
import plotly.express as px
import streamlit as slt
from plotly.subplots import make_subplots
from src.auth import init_session_state, require_login, check_session
from ml_bank_model import pred_client, train_gb_artifact

file_path = "Kaggle Database/bank.csv"
bank_daf = pds.read_csv(file_path)
init_session_state()

mth_numbs = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
mth_seq = list(mth_numbs.keys())
mth_labls = {
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
    "dec": "Декабрь",
}

bank_daf["deposit_flag"] = (bank_daf["deposit"] == "yes").astype(int)
bank_daf["campaign_date"] = pds.to_datetime({
    "year": 2024,
    "month": bank_daf["month"].map(mth_numbs),
    "day": bank_daf["day"],
})

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

slt.header("Аналитика рекламной кампании Банка-X")
slt.markdown("Динамика контактов и открытий депозитов")

# --- Контейнеры ---
main_cont = slt.container(
    key="bank_main_cont",
    horizontal_alignment="center",
    vertical_alignment="center",
    border=False,
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

    predictor_panel = slt.container(
        border=True
    )



with overview_panel:
    slt.subheader("Динамика контактов и отклика")

    view_mode = slt.radio(
        ":gray[Выберите график:]",
        ["Контакты и конверсия", "Нагрузка по месяцам"],
        horizontal=True,
        key="bank_overview_mode",
    )

    if view_mode == "Контакты и конверсия":
        daily_contact = (
            bank_daf
            .dropna(subset=["campaign_date"])
            .groupby("campaign_date")
            .agg(
                calls_count=("campaign", "sum"),
                clients_count=("deposit", "count"),
                deposits_count=("deposit_flag", "sum"),
            )
            .asfreq("D")
            .fillna(0)
        )

        daily_contact["conversion_rate"] = npy.where(
            daily_contact["clients_count"] > 0,
            daily_contact["deposits_count"] / daily_contact["clients_count"] * 100,
            0,
        )

        monthly_conversion = (
            bank_daf
            .groupby("month", as_index=False)
            .agg(
                clients_count=("deposit", "count"),
                deposits_count=("deposit_flag", "sum"),
            )
        )
        monthly_conversion["month"] = pds.Categorical(
            monthly_conversion["month"],
            categories=mth_seq,
            ordered=True,
        )
        monthly_conversion = monthly_conversion.sort_values("month")
        monthly_conversion["conversion_rate"] = npy.where(
            monthly_conversion["clients_count"] > 0,
            monthly_conversion["deposits_count"] / monthly_conversion["clients_count"] * 100,
            0,
        )
        monthly_conversion["month_name"] = monthly_conversion["month"].map(mth_labls)

        show_metrics = slt.multiselect(
            ":gray[Показывать линии:]",
            options=["Контакты", "Клиенты", "Депозиты"],
            default=["Контакты", "Клиенты", "Депозиты"],
            key="bank_contact_lines",
        )

        if not show_metrics:
            slt.warning("Выберите хотя бы одну линию для отображения")
            slt.stop()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        metric_map = {
            "Контакты": ("calls_count", "#3fa7d6"),
            "Клиенты": ("clients_count", "#f4b942"),
            "Депозиты": ("deposits_count", "#66d17a"),
        }

        for label in show_metrics:
            source_column, line_color = metric_map[label]
            fig.add_trace(
                go.Scatter(
                    x=daily_contact.index,
                    y=daily_contact[source_column],
                    mode="lines",
                    name=label,
                    line=dict(width=2, color=line_color),
                    hovertemplate=(
                        "Дата: %{x|%d.%m}<br>"
                        f"{label}: " + "%{y:.0f}<extra></extra>"
                    ),
                ),
                secondary_y=False,
            )

        fig.add_trace(
            go.Scatter(
                x=monthly_conversion["month_name"],
                y=monthly_conversion["conversion_rate"],
                mode="lines+markers",
                name="Конверсия",
                line=dict(width=3, dash="dash", color="#ff6b6b"),
                marker=dict(size=8),
                hovertemplate=(
                    "Месяц: %{x}<br>"
                    "Конверсия: %{y:.2f}%<extra></extra>"
                ),
            ),
            secondary_y=True,
        )

        fig.update_layout(
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.45,
                xanchor="left",
                x=0,
            ),
            margin=dict(l=20, r=20, t=20, b=80),
            xaxis_title="Дата контакта",
            template="plotly_dark",
        )
        fig.update_yaxes(title_text="Количество", secondary_y=False)
        fig.update_yaxes(title_text="Конверсия, %", secondary_y=True)

        slt.plotly_chart(fig, use_container_width=True)

    else:
        workload = (
            bank_daf
            .groupby(["month", "contact"], as_index=False)
            .agg(clients_count=("deposit", "count"))
        )
        workload["month"] = pds.Categorical(
            workload["month"],
            categories=mth_seq,
            ordered=True,
        )
        workload = workload.sort_values(["month", "contact"])
        workload["month_name"] = workload["month"].map(mth_labls)

        contacts = sorted(workload["contact"].dropna().unique())
        chosen_channels = slt.multiselect(
            "Каналы связи:",
            options=contacts,
            default=contacts,
            key="bank_contact_channels",
        )

        if not chosen_channels:
            slt.warning("Выберите хотя бы один канал связи")
            slt.stop()

        workload_slice = workload[workload["contact"].isin(chosen_channels)]

        channel_colors = {
            "cellular": "#3fa7d6",
            "telephone": "#f4b942",
            "unknown": "#9aa0a6",
        }

        fig = go.Figure()

        for channel in chosen_channels:
            channel_part = workload_slice[workload_slice["contact"] == channel]
            fig.add_trace(
                go.Bar(
                    x=channel_part["month_name"],
                    y=channel_part["clients_count"],
                    name=channel,
                    marker_color=channel_colors.get(channel, None),
                    hovertemplate=(
                        "Месяц: %{x}<br>"
                        "Канал: " + channel + "<br>"
                        "Клиентов: %{y}<extra></extra>"
                    ),
                )
            )

        fig.update_layout(
            barmode="stack",
            height=500,
            title="Объем обращений по месяцам и каналам связи",
            xaxis_title="Месяц",
            yaxis_title="Количество клиентов",
            legend_title="Канал",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=60, b=40),
        )

        slt.plotly_chart(fig, use_container_width=True)

with profile_panel:
    slt.subheader("Баланс и длительность разговора")

    balance_view = bank_daf[["balance", "duration", "deposit"]].dropna().rename(columns={
        "balance": "Баланс клиента",
        "duration": "Длительность звонка",
        "deposit": "Открыл депозит"
    })
    balance_view["Открыл депозит"] = balance_view["Открыл депозит"].map({
        "yes": "Да",
        "no": "Нет"
    })

    slt.scatter_chart(
        balance_view,
        x="Баланс клиента",
        y="Длительность звонка",
        color="Открыл депозит"
    )

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

with segment_panel:
    slt.subheader("Отклик по профессиям")

    all_jobs = sorted(bank_daf["job"].dropna().unique())
    picked_jobs = slt.multiselect(
        "Профессии клиентов:",
        options=all_jobs,
        default=all_jobs,
        key="bank_job_filter",
        format_func=lambda value: VAL_LABLS["job"].get(value, value)
    )

    if not picked_jobs:
        slt.warning("Выберите хотя бы одну профессию")
        slt.stop()

    job_reply = (
        bank_daf[bank_daf["job"].isin(picked_jobs)]
        .groupby(["job", "deposit"], as_index=False)
        .size()
        .rename(columns={
            "job": "Профессия",
            "deposit": "Депозит",
            "size": "Количество клиентов"
        })
    )
    job_reply["Депозит"] = job_reply["Депозит"].map({
        "yes": "Да",
        "no": "Нет"
    })

    slt.bar_chart(
        job_reply,
        x="Профессия",
        y="Количество клиентов",
        color="Депозит"
    )


with conversion_panel:
    slt.subheader("Лучшие клиентские группы")

    education_success = (
        bank_daf
        .groupby(["education", "marital"], as_index=False)
        .agg(
            clients_count=("deposit", "count"),
            deposits_count=("deposit_flag", "sum")
        )
    )
    education_success["conversion_rate"] = npy.where(
        education_success["clients_count"] >= 25,
        education_success["deposits_count"] / education_success["clients_count"] * 100,
        npy.nan
    )
    education_success = (
        education_success
        .dropna(subset=["conversion_rate"])
        .sort_values("conversion_rate", ascending=False)
        .head(10)
    )
    education_success["segment"] = (
        education_success["education"] + " / " + education_success["marital"]
    )
    education_success = education_success.rename(columns={
        "segment": "Сегмент",
        "conversion_rate": "Конверсия, %"
    })

    slt.bar_chart(
        education_success,
        x="Сегмент",
        y="Конверсия, %",
        color="Сегмент"
    )


def bank_ml_artif(path):
    artifact, *_ = train_gb_artifact(path)
    return artifact

with driver_panel:
    slt.subheader("Топ 10 факторов влияющих на открытие депозита")

    model_artifact = bank_ml_artif(str(file_path))
    leading_features = model_artifact["feature_importance"].head(10).copy()
    leading_features.columns = ["Фактор", "Важность"]

    slt.bar_chart(
        leading_features,
        x="Фактор",
        y="Важность",
        color="Фактор"
    )


with balance_reason_panel:
    slt.subheader("Факторы влияющие на баланс клиента")

    balance_reason_view = (
        bank_daf[["balance", "duration", "marital", "education", "loan"]]
        .dropna()
        .copy()
    )
    balance_reason_view["Семейное положение"] = balance_reason_view["marital"].map(VAL_LABLS["marital"])
    balance_reason_view["Образование"] = balance_reason_view["education"].map(VAL_LABLS["education"])
    balance_reason_view["Персональный заем"] = balance_reason_view["loan"].map(VAL_LABLS["loan"])
    balance_reason_view = balance_reason_view.rename(columns={
        "balance": "Баланс",
        "duration": "Длительность звонка"
    })

    fig = px.scatter(
        balance_reason_view,
        x="Длительность звонка",
        y="Баланс",
        color="Семейное положение",
        marginal_x="box",
        marginal_y="violin",
        hover_data=["Семейное положение", "Образование", "Персональный заем"],
        color_discrete_sequence=["#636EFA", "#EF553B", "#00CC96"],
        title="Влияние факторов на баланс клиента"
    )
    fig.update_traces(marker=dict(size=6, opacity=0.65), selector=dict(mode="markers"))
    fig.update_layout(
        height=560,
        template="plotly_dark",
        title=dict(
            text="Возможные причины низкого баланса",
            y=0.96,
            x=0.5,
            xanchor="center",
            yanchor="top"
        ),
        xaxis_title="Длительность звонка, сек.",
        yaxis_title="Баланс",
        legend_title="Семейное положение",
        font=dict(
            family="Arial",
            size=15
        ),
        margin=dict(l=20, r=20, t=70, b=30)
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)

    slt.plotly_chart(fig, use_container_width=True)
    slt.markdown(
        """
        - **Семейное положение:** распределение баланса отличается между группами, поэтому этот признак полезно учитывать при сегментации клиентов.
        - **Образование:** уровень образования может быть связан с финансовым профилем клиента и размером остатка на счете.
        - **Займы:** наличие персонального займа часто снижает свободный остаток и может объяснять низкий баланс.
        """
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

with predictor_panel:
    slt.subheader("Прогноз открытия депозита")
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



# time.sleep(5)
# slt.rerun()
