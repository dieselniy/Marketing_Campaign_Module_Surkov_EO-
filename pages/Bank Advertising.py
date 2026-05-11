import numpy as npy
import pandas as pds
import plotly.graph_objects as go
import streamlit as slt
from plotly.subplots import make_subplots
from src.auth import init_session_state, require_login, check_session

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

    balance_view = bank_frame[["balance", "duration", "deposit"]].dropna().rename(columns={
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


# time.sleep(5)
# slt.rerun()
