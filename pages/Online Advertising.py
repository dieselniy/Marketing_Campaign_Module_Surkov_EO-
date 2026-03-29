import streamlit as slt
from src.auth import init_session_state, require_login, check_session
import time
import numpy as npy
import pandas as pds
from sklearn.ensemble import RandomForestRegressor
import plotly.graph_objects as go
from plotly.subplots import make_subplots

file_path = "Kaggle Database/online_advertising_performance_data.csv"

daf = pds.read_csv(file_path)
# --- Инициализация session_state ---
init_session_state()

slt.set_page_config(page_title="Онлайн Кампании", layout="wide")

# --- Восстановление session_id ---
params = slt.query_params
if "session_id" in params:
    slt.session_state.session_id = params["session_id"][0]
    if check_session():
        slt.session_state.authenticated = True

# --- Проверка авторизации ---
#require_login()

slt.header("Аналитика Онлайн Маркетинговых Кампаний")

# --- Контейнеры ---
main_container = slt.container(
    key='form',
    horizontal_alignment='center',
    vertical_alignment='center',
    border=False
)


with main_container:
    graph_container = slt.container(
        height=800,
        border=True
    )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    col1, col2 = slt.columns([1, 1], gap="large")

    with col1:
        scatter_container = slt.container(
            height=600,
            border=True
        )

    with col2:
        barchart_container = slt.container(
            height=600,
            border=True
        )

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    col3, col4 = slt.columns([1, 1], gap="large")

    with col3:
        success_container = slt.container(
            height=500,
            border=True
        )

    with col4:
        importance_container = slt.container(
            height=500,
            border=True
        )
    

daf['ROMI'] = npy.where(
    daf['post_click_conversions'] > 0,
    daf['cost'] / daf['post_click_conversions'],
    0
)

with graph_container:
    slt.subheader("Анализ кампаний")

    chart_type = slt.radio(
        ':gray[Выберите график:]',
        ["Затраты и ROMI", "Использование кампаний по месяцам"],
        horizontal=True,
        key="graph_mode"
    )

    # --- Дата ---
    daf['date'] = pds.to_datetime(
        daf['month'] + ' ' + daf['day'].astype(str) + ' 2024'
    )

    if chart_type == "Затраты и ROMI":
        # --- Ежедневные затраты ---
        campaign_grouped = (
            daf.groupby(['date', 'campaign_number'])['cost']
            .sum()
            .unstack()
            .asfreq('D')
        )

        campaign_grouped = campaign_grouped.reindex(
            sorted(campaign_grouped.columns, key=lambda x: int(x.split()[1])),
            axis=1
        )

        total_grouped = daf.groupby('date')['cost'].sum().asfreq('D').fillna(0)

        final_df = campaign_grouped.copy()
        final_df["Total"] = total_grouped

        # --- Средний ROMI по месяцам по кампаниям ---
        monthly_ROMI_df = (
            daf.groupby(['month', 'campaign_number'])['ROMI']
            .mean()
            .reset_index()
        )

        month_order = ['April', 'May', 'June']
        monthly_ROMI_df['month'] = pds.Categorical(
            monthly_ROMI_df['month'],
            categories=month_order,
            ordered=True
        )

        monthly_ROMI_df = monthly_ROMI_df.sort_values(['month', 'campaign_number'])

        month_date_map = {
            'April': pds.Timestamp('2024-04-15'),
            'May': pds.Timestamp('2024-05-15'),
            'June': pds.Timestamp('2024-06-15')
        }
        monthly_ROMI_df['date'] = monthly_ROMI_df['month'].map(month_date_map)

        # --- Total ROMI по месяцам ---
        monthly_total_ROMI = (
            daf.groupby('month')[['cost', 'post_click_conversions']]
            .sum()
            .reset_index()
        )

        monthly_total_ROMI['month'] = pds.Categorical(
            monthly_total_ROMI['month'],
            categories=month_order,
            ordered=True
        )

        monthly_total_ROMI = monthly_total_ROMI.sort_values('month')

        monthly_total_ROMI['ROMI'] = npy.where(
            monthly_total_ROMI['post_click_conversions'] > 0,
            monthly_total_ROMI['cost'] / monthly_total_ROMI['post_click_conversions'],
            0
        )

        monthly_total_ROMI['date'] = monthly_total_ROMI['month'].map(month_date_map)

        # --- UI: выбор кампаний ---
        all_columns = final_df.columns.tolist()

        selected_columns = slt.multiselect(
            ":gray[Выберите кампании:]",
            options=all_columns,
            default=all_columns,
            key="cost_campaigns"
        )

        if not selected_columns:
            slt.warning("Выберите хотя бы одну кампанию")
            slt.stop()

        filtered_cost_df = final_df[selected_columns]

        selected_ROMI_campaigns = [col for col in selected_columns if col != "Total"]
        filtered_ROMI_df = monthly_ROMI_df[
            monthly_ROMI_df['campaign_number'].isin(selected_ROMI_campaigns)
        ]

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        color_map = {
            "Total": "#7ec8ff",
            "camp 1": "#1f77b4",
            "camp 2": "#ffb6b6",
            "camp 3": "#ff2d2d",
            "Total ROMI": "#00ffcc"
        }

        # --- Линии затрат ---
        for col in filtered_cost_df.columns:
            fig.add_trace(
                go.Scatter(
                    x=filtered_cost_df.index,
                    y=filtered_cost_df[col],
                    mode='lines',
                    name=f"{col} — Затраты",
                    connectgaps=False,
                    line=dict(
                        width=2,
                        color=color_map.get(col, None)
                    ),
                    hovertemplate=(
                        "Дата: %{x}<br>"
                        "Кампания: " + col + "<br>"
                        "Затраты: %{y}<extra></extra>"
                    )
                ),
                secondary_y=False
            )

        # --- Линии ROMI по кампаниям ---
        for campaign in selected_ROMI_campaigns:
            campaign_ROMI = filtered_ROMI_df[
                filtered_ROMI_df['campaign_number'] == campaign
            ]

            fig.add_trace(
                go.Scatter(
                    x=campaign_ROMI['date'],
                    y=campaign_ROMI['ROMI'],
                    mode='lines+markers',
                    name=f"{campaign} — ROMI",
                    line=dict(
                        width=3,
                        dash='dash',
                        color=color_map.get(campaign, None)
                    ),
                    marker=dict(size=8),
                    hovertemplate=(
                        "Месяц: %{x|%B}<br>"
                        "Кампания: " + campaign + "<br>"
                        "Средний ROMI: %{y:.3f}<extra></extra>"
                    )
                ),
                secondary_y=True
            )

        # --- Линия Total ROMI ---
        if "Total" in selected_columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_total_ROMI['date'],
                    y=monthly_total_ROMI['ROMI'],
                    mode='lines+markers',
                    name="Total — ROMI",
                    line=dict(
                        width=4,
                        dash='dot',
                        color=color_map["Total ROMI"]
                    ),
                    marker=dict(size=9),
                    hovertemplate=(
                        "Месяц: %{x|%B}<br>"
                        "Общий ROMI: %{y:.3f}<extra></extra>"
                    )
                ),
                secondary_y=True
            )

        fig.update_layout(
            height=500,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.50,
                xanchor="left",
                x=0
            ),
            margin=dict(l=20, r=20, t=20, b=80),
            xaxis_title="Дата",
            template="plotly_dark"
        )

        fig.update_yaxes(title_text="Затраты", secondary_y=False)
        fig.update_yaxes(title_text="Средний ROMI", secondary_y=True)

        slt.plotly_chart(fig, use_container_width=True)

    else:
        # --- Использование кампаний по месяцам ---
        campaign_usage = (
            daf.groupby(['date', 'campaign_number'])
            .agg(
                usage_count=('campaign_number', 'count'),
                avg_ROMI=('ROMI', 'mean')
            )
            .reset_index()
        )

        campaign_usage['month'] = campaign_usage['date'].dt.strftime('%B')

        monthly_campaign_usage = (
            campaign_usage.groupby(['month', 'campaign_number'])
            .agg(
                usage_count=('usage_count', 'sum'),
                avg_ROMI=('avg_ROMI', 'mean')
            )
            .reset_index()
        )

        month_order = ['April', 'May', 'June']
        monthly_campaign_usage['month'] = pds.Categorical(
            monthly_campaign_usage['month'],
            categories=month_order,
            ordered=True
        )
        monthly_campaign_usage = monthly_campaign_usage.sort_values('month')

        campaigns = sorted(
            monthly_campaign_usage['campaign_number'].unique(),
            key=lambda x: int(x.split()[1])
        )

        selected_campaigns = slt.multiselect(
            "Выберите кампании:",
            options=campaigns,
            default=campaigns,
            key="usage_campaigns"
        )

        if not selected_campaigns:
            slt.warning("Выберите хотя бы одну кампанию")
            slt.stop()

        filtered_usage = monthly_campaign_usage[
            monthly_campaign_usage['campaign_number'].isin(selected_campaigns)
        ]

        color_map = {
            "camp 1": "#1f77b4",
            "camp 2": "#ff7f0e",
            "camp 3": "#2ca02c",
        }

        fig = go.Figure()

        for campaign in selected_campaigns:
            campaign_data = filtered_usage[
                filtered_usage['campaign_number'] == campaign
            ]

            fig.add_trace(
                go.Bar(
                    x=campaign_data['month'],
                    y=campaign_data['usage_count'],
                    name=campaign,
                    marker_color=color_map.get(campaign, None),
                    hovertemplate=(
                        "Месяц: %{x}<br>"
                        "Кампания: " + campaign + "<br>"
                        "Количество использований: %{y}<extra></extra>"
                    )
                )
            )

        fig.update_layout(
            barmode='stack',
            height=500,
            title="Использование кампаний по месяцам",
            xaxis_title="Месяц",
            yaxis_title="Количество кампаний за месяц",
            legend_title="Кампания",
            template="plotly_dark",
            margin=dict(l=20, r=20, t=60, b=40)
        )

        slt.plotly_chart(fig, use_container_width=True)


with scatter_container:
    slt.subheader("ROMI по местам размещения")
    
    scatter_df = daf[['placement', 'ROMI']].dropna().copy()
    scatter_df = scatter_df.rename(columns={
        'placement': 'Места размещения рекламных баннеров'
    })

    slt.scatter_chart(
        scatter_df,
        x='Места размещения рекламных баннеров',
        y='ROMI',
    )    


with barchart_container:
    slt.subheader("Кол-во конверсий после клика по вовлеченности")

    # --- UI: выбор кампаний ---
    campas = sorted(daf['campaign_number'].unique(), key=lambda x: int(x.split()[1]))

    selected_campaigns = slt.multiselect(
        "Выберите кампании:",
        options=campas,
        default=campas
    )

    if not selected_campaigns:
        slt.warning("Выберите хотя бы одну кампанию")
        slt.stop()

    # --- Фильтрация ---
    filtered_daf = daf[daf['campaign_number'].isin(selected_campaigns)]

    # --- Группировка ---
    engagement_daf = (
        filtered_daf.groupby(['user_engagement', 'campaign_number'])['post_click_conversions']
        .sum()
        .reset_index()
    )

    # --- Переименование ---
    engagement_daf = engagement_daf.rename(columns={
        'user_engagement': 'Вовлеченность',
        'post_click_conversions': 'Конверсии',
        'campaign_number': 'Кампания'
    })

    # --- График ---
    slt.bar_chart(
        engagement_daf,
        x='Вовлеченность',
        y='Конверсии',
        color='Кампания'
    )

    with success_container:
        slt.subheader("Успешные кампании по ROMI")

        successful_campaigns = daf[daf['ROMI'] > 1]

        success_df = (
            successful_campaigns.groupby('campaign_number', as_index=False)['ROMI']
            .mean()
            .sort_values(by='ROMI', ascending=False)
        )

        success_df = success_df.rename(columns={
            'campaign_number': 'Кампания',
            'ROMI': 'ROMI'
        })

        slt.bar_chart(
            success_df,
            x='Кампания',
            y='ROMI',
            color='Кампания'
        )

        with importance_container:
            slt.subheader("Топ 10 признаков для ROMI")

            X = daf.drop(columns=['ROMI', 'date']).copy()
            X = pds.get_dummies(X)
            y = daf['ROMI']

            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)

            importances = pds.Series(rf.feature_importances_, index=X.columns)

            top_features = (
                importances.sort_values(ascending=False)
                .head(10)
                .reset_index()
            )

            top_features.columns = ['Признак', 'Важность']

            slt.bar_chart(
                top_features,
                x='Признак',
                y='Важность',
                color='Признак'
            )



# --- Автообновление ---
# time.sleep(5)
# slt.rerun()
