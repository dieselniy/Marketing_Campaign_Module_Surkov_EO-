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

    slt.markdown("<div style='margin-top: 24px;'></div>", unsafe_allow_html=True)

    roas_pred = slt.container(
        border=True
    )
    

daf['ROI'] = npy.where(
    daf['post_click_conversions'] > 0,
    daf['cost'] / daf['post_click_conversions'],
    0
)

daf['CTR'] = npy.where(
    daf['displays'] > 0,
    daf['clicks'] / daf['displays'],
    0
)
daf['CVR'] = npy.where(
    daf['clicks'] > 0,
    daf['post_click_conversions'] / daf['clicks'],
    0
)
daf['CPC'] = npy.where(
    daf['clicks'] > 0,
    daf['cost'] / daf['clicks'],
    0
)
daf['CPA'] = npy.where(
    daf['post_click_conversions'] > 0,
    daf['cost'] / daf['post_click_conversions'],
    0
)
daf['CPM'] = npy.where(
    daf['displays'] > 0,
    daf['cost'] / daf['displays'] * 1000,
    0
)
adv_lbls = {
    "month": {
        "April": "Апрель",
        "May": "Май",
        "June": "Июнь"
    },
    "campaign_number": {
        "camp 1": "Кампания 1",
        "camp 2": "Кампания 2",
        "camp 3": "Кампания 3"
    },
    "user_engagement": {
        "High": "Высокая",
        "Medium": "Средняя",
        "Low": "Низкая"
    },
    "placement": {
        "abc": "Площадка ABC",
        "def": "Площадка DEF",
        "ghi": "Площадка GHI",
        "jkl": "Площадка JKL",
        "mno": "Площадка MNO"
    }
}

forc_year = 2026
forc_modelcache = "budget_funnel_v3"
month_ordr = ["April", "May", "June"]
month_numbr = {
    "April": 4,
    "May": 5,
    "June": 6,
}
mname_bynumbr = {
    month_number: month_name
    for month_name, month_number in month_numbr.items()
}


def adv_selectbox(label, field_name, options, key, index=0):
    labels = adv_lbls.get(field_name, {})
    return slt.selectbox(
        label,
        options,
        index=index,
        key=key,
        format_func=lambda value: labels.get(value, value)
    )


@slt.cache_resource
def get_campaign_forecast_model(data, cache_version):
    return train_campaign_forecast(data)


def data_forecboud():
    available_months = [
        month_name
        for month_name in month_ordr
        if month_name in set(daf["month"].dropna().unique())
    ]
    first_month = month_numbr[available_months[0]]
    last_month = month_numbr[available_months[-1]]

    min_day = int(daf.loc[daf["month"] == available_months[0], "day"].min())
    max_day = int(daf.loc[daf["month"] == available_months[-1], "day"].max())

    return (
        date(forc_year, first_month, min_day),
        date(forc_year, last_month, max_day),
    )


def build_campaign_days(start_dt, end_dt, banner, placement):
    days = []
    current_dt = start_dt

    while current_dt <= end_dt:
        month_name = mname_bynumbr.get(current_dt.month)
        if month_name is None:
            current_dt += timedelta(days=1)
            continue

        days.append({
            "month": month_name,
            "day": current_dt.day,
            "banner": banner,
            "placement": placement,
        })
        current_dt += timedelta(days=1)

    return days


def render_performance_metrics(data, selected_campaigns):
    selected_label = ", ".join(selected_campaigns) if selected_campaigns else "Total"
    slt.subheader(f"Ключевые метрики: {selected_label}")
    total_displays = data['displays'].sum()
    total_clicks = data['clicks'].sum()
    total_cost = data['cost'].sum()
    total_conversions = data['post_click_conversions'].sum()
    total_ctr = total_clicks / total_displays if total_displays > 0 else 0
    total_cvr = total_conversions / total_clicks if total_clicks > 0 else 0
    total_cpc = total_cost / total_clicks if total_clicks > 0 else 0
    total_cpa = total_cost / total_conversions if total_conversions > 0 else 0
    total_cpm = total_cost / total_displays * 1000 if total_displays > 0 else 0

    metric_col_1, metric_col_2, metric_col_3, metric_col_4, metric_col_5 = slt.columns(5)

    with metric_col_1:
        slt.metric("CTR", f"{total_ctr:.2%}")
    with metric_col_2:
        slt.metric("CVR", f"{total_cvr:.2%}")
    with metric_col_3:
        slt.metric("CPC", f"${total_cpc:.4f}")
    with metric_col_4:
        slt.metric("CPA", f"${total_cpa:.4f}")
    with metric_col_5:
        slt.metric("CPM", f"${total_cpm:.4f}")

    campaign_metrics = (
        data.groupby('campaign_number', as_index=False)
        .agg(
            displays=('displays', 'sum'),
            clicks=('clicks', 'sum'),
            cost=('cost', 'sum'),
            conversions=('post_click_conversions', 'sum')
        )
    )
    campaign_metrics['CTR'] = npy.where(
        campaign_metrics['displays'] > 0,
        campaign_metrics['clicks'] / campaign_metrics['displays'],
        0
    )
    campaign_metrics['CVR'] = npy.where(
        campaign_metrics['clicks'] > 0,
        campaign_metrics['conversions'] / campaign_metrics['clicks'],
        0
    )
    campaign_metrics['CPC'] = npy.where(
        campaign_metrics['clicks'] > 0,
        campaign_metrics['cost'] / campaign_metrics['clicks'],
        0
    )
    campaign_metrics['CPA'] = npy.where(
        campaign_metrics['conversions'] > 0,
        campaign_metrics['cost'] / campaign_metrics['conversions'],
        0
    )
    campaign_metrics['CPM'] = npy.where(
        campaign_metrics['displays'] > 0,
        campaign_metrics['cost'] / campaign_metrics['displays'] * 1000,
        0
    )

    campaign_metrics = campaign_metrics.sort_values(
        'campaign_number',
        key=lambda column: column.str.extract(r'(\d+)')[0].astype(int)
    )

    display_campaign_metrics = campaign_metrics.copy()
    display_campaign_metrics['CTR'] = display_campaign_metrics['CTR'] * 100
    display_campaign_metrics['CVR'] = display_campaign_metrics['CVR'] * 100
    display_campaign_metrics = display_campaign_metrics.rename(columns={
        'campaign_number': 'Кампания',
        'displays': 'Показы',
        'clicks': 'Клики',
        'cost': 'Расходы',
        'conversions': 'Конверсии'
    })

    slt.dataframe(
        display_campaign_metrics,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Расходы": slt.column_config.NumberColumn(format="$%.2f"),
            "CTR": slt.column_config.NumberColumn(format="%.2f%%"),
            "CVR": slt.column_config.NumberColumn(format="%.2f%%"),
            "CPC": slt.column_config.NumberColumn(format="$%.4f"),
            "CPA": slt.column_config.NumberColumn(format="$%.4f"),
            "CPM": slt.column_config.NumberColumn(format="$%.4f"),
        }
    )


with graph_container:
    slt.subheader("Анализ кампаний")

    chart_type = slt.radio(
        ':gray[Выберите график:]',
        ["Затраты и ROI", "Использование кампаний по месяцам"],
        horizontal=True,
        key="graph_mode"
    )

    # --- Дата ---
    daf['date'] = pds.to_datetime(
        daf['month'] + ' ' + daf['day'].astype(str) + ' 2024'
    )

    if chart_type == "Затраты и ROI":
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

        # --- Средний ROI по месяцам по кампаниям ---
        monthly_roi_df = (
            daf.groupby(['month', 'campaign_number'])['ROI']
            .mean()
            .reset_index()
        )

        month_order = ['April', 'May', 'June']
        monthly_roi_df['month'] = pds.Categorical(
            monthly_roi_df['month'],
            categories=month_order,
            ordered=True
        )

        monthly_roi_df = monthly_roi_df.sort_values(['month', 'campaign_number'])

        month_date_map = {
            'April': pds.Timestamp('2024-04-15'),
            'May': pds.Timestamp('2024-05-15'),
            'June': pds.Timestamp('2024-06-15')
        }
        monthly_roi_df['date'] = monthly_roi_df['month'].map(month_date_map)

        # --- Total ROI по месяцам ---
        monthly_total_roi = (
            daf.groupby('month')[['cost', 'post_click_conversions']]
            .sum()
            .reset_index()
        )

        monthly_total_roi['month'] = pds.Categorical(
            monthly_total_roi['month'],
            categories=month_order,
            ordered=True
        )

        monthly_total_roi = monthly_total_roi.sort_values('month')

        monthly_total_roi['ROI'] = npy.where(
            monthly_total_roi['post_click_conversions'] > 0,
            monthly_total_roi['cost'] / monthly_total_roi['post_click_conversions'],
            0
        )

        monthly_total_roi['date'] = monthly_total_roi['month'].map(month_date_map)

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

        selected_roi_campaigns = [col for col in selected_columns if col != "Total"]
        filtered_roi_df = monthly_roi_df[
            monthly_roi_df['campaign_number'].isin(selected_roi_campaigns)
        ]

        metric_campaigns = selected_roi_campaigns
        metric_data = (
            daf[daf['campaign_number'].isin(metric_campaigns)]
            if metric_campaigns
            else daf
        )
        render_performance_metrics(metric_data, metric_campaigns)

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        color_map = {
            "Total": "#7ec8ff",
            "camp 1": "#1f77b4",
            "camp 2": "#ffb6b6",
            "camp 3": "#ff2d2d",
            "Total ROI": "#00ffcc"
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

        # --- Линии ROI по кампаниям ---
        for campaign in selected_roi_campaigns:
            campaign_roi = filtered_roi_df[
                filtered_roi_df['campaign_number'] == campaign
            ]

            fig.add_trace(
                go.Scatter(
                    x=campaign_roi['date'],
                    y=campaign_roi['ROI'],
                    mode='lines+markers',
                    name=f"{campaign} — ROI",
                    line=dict(
                        width=3,
                        dash='dash',
                        color=color_map.get(campaign, None)
                    ),
                    marker=dict(size=8),
                    hovertemplate=(
                        "Месяц: %{x|%B}<br>"
                        "Кампания: " + campaign + "<br>"
                        "Средний ROI: %{y:.3f}<extra></extra>"
                    )
                ),
                secondary_y=True
            )

        # --- Линия Total ROI ---
        if "Total" in selected_columns:
            fig.add_trace(
                go.Scatter(
                    x=monthly_total_roi['date'],
                    y=monthly_total_roi['ROI'],
                    mode='lines+markers',
                    name="Total — ROI",
                    line=dict(
                        width=4,
                        dash='dot',
                        color=color_map["Total ROI"]
                    ),
                    marker=dict(size=9),
                    hovertemplate=(
                        "Месяц: %{x|%B}<br>"
                        "Общий ROI: %{y:.3f}<extra></extra>"
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
        fig.update_yaxes(title_text="Средний ROI", secondary_y=True)

        slt.plotly_chart(fig, use_container_width=True)

    else:
        # --- Использование кампаний по месяцам ---
        campaign_usage = (
            daf.groupby(['date', 'campaign_number'])
            .agg(
                usage_count=('campaign_number', 'count'),
                avg_roi=('ROI', 'mean')
            )
            .reset_index()
        )

        campaign_usage['month'] = campaign_usage['date'].dt.strftime('%B')

        monthly_campaign_usage = (
            campaign_usage.groupby(['month', 'campaign_number'])
            .agg(
                usage_count=('usage_count', 'sum'),
                avg_roi=('avg_roi', 'mean')
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

        metric_data = daf[daf['campaign_number'].isin(selected_campaigns)]
        render_performance_metrics(metric_data, selected_campaigns)

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
    slt.subheader("ROI по местам размещения")
    
    scatter_df = daf[['placement', 'ROI']].dropna().copy()
    scatter_df = scatter_df.rename(columns={
        'placement': 'Места размещения рекламных баннеров'
    })

    slt.scatter_chart(
        scatter_df,
        x='Места размещения рекламных баннеров',
        y='ROI',
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
        slt.subheader("Успешные кампании по ROI")

        successful_campaigns = daf[daf['ROI'] > 1]

        success_df = (
            successful_campaigns.groupby('campaign_number', as_index=False)['ROI']
            .mean()
            .sort_values(by='ROI', ascending=False)
        )

        success_df = success_df.rename(columns={
            'campaign_number': 'Кампания',
            'ROI': 'ROI'
        })

        slt.bar_chart(
            success_df,
            x='Кампания',
            y='ROI',
            color='Кампания'
        )

        with importance_container:
            slt.subheader("Топ 10 признаков для ROI")

            X = daf.drop(columns=['ROI', 'date']).copy()
            X = pds.get_dummies(X)
            y = daf['ROI']

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

with roas_pred:
    slt.subheader("Прогноз окупаемости рекламного размещения")
    slt.caption(
        "Произведите оценку вашего размещения на разных платформах по заданным параметрам"
    )

    forecast_min_date, forecast_max_date = data_forecboud()
    default_start_date = forecast_min_date
    default_end_date = min(
        forecast_max_date,
        default_start_date + timedelta(days=30)
    )

    with slt.form("prognoz_form"):
        adv_col_1, adv_col_2, adv_col_3 = slt.columns(3, gap="large")

        with adv_col_1:
            budget = slt.number_input(
                "Бюджет кампании ($)",
                min_value=1.0,
                value=1000.0,
                step=100.0,
                key="campaign_budget"
            )

        with adv_col_2:
            start_dt = slt.date_input(
                "Дата начала",
                value=default_start_date,
                min_value=forecast_min_date,
                max_value=forecast_max_date,
                key="campaign_start_date"
            )
            default_end_for_start = max(start_dt, default_end_date)
            end_dt = slt.date_input(
                "Дата окончания",
                value=default_end_for_start,
                min_value=start_dt,
                max_value=forecast_max_date,
                key="campaign_end_date"
            )

        with adv_col_3:
            banner = slt.selectbox(
                "Формат баннера",
                sorted(daf["banner"].dropna().unique()),
                key="forecast_banner"
            )
            placement = adv_selectbox(
                "Площадка",
                "placement",
                sorted(daf["placement"].dropna().unique()),
                key="forecast_placement"
            )

        submitted_roas = slt.form_submit_button("Рассчитать прогноз", type="primary", use_container_width=True)

    if submitted_roas:
        campaign_days = build_campaign_days(start_dt, end_dt, banner, placement)
        forecast_model = get_campaign_forecast_model(daf, forc_modelcache)
        forecast = predict_campaign_forecast(
            forecast_model,
            campaign_days=campaign_days,
            budget=budget
        )

        roas_col_1, roas_col_2, roas_col_3, roas_col_4 = slt.columns(4, gap="large")

        with roas_col_1:
            slt.metric("Ожидаемые показы", f"{forecast['expected_displays']:,.0f}")

        with roas_col_2:
            slt.metric("Ожидаемые клики", f"{forecast['expected_clicks']:,.0f}")

        with roas_col_3:
            slt.metric("ROI", f"{forecast['roi_percent']:.1f}%")

        with roas_col_4:
            slt.metric("Прогноз выручки", f"${forecast['expected_sales']:,.2f}")

        slt.progress(min(max(forecast["roas"] / 3, 0), 1.0))

        if forecast["roi_percent"] >= 50:
            slt.success("Высокая ожидаемая окупаемость размещения")
        elif forecast["roi_percent"] >= 0:
            slt.warning("Ожидаемая окупаемость около точки безубыточности")
        else:
            slt.error("Ожидаемая окупаемость ниже бюджета кампании")

        slt.caption(
            f"Период: {forecast['duration_days']} дней. "
            f"ROAS: {forecast['roas']:.2f}. "
            f"Дневной бюджет: ${forecast['daily_budget']:,.2f}. "
            f"Прогноз построен для периода {forecast_min_date:%d.%m.%Y} - {forecast_max_date:%d.%m.%Y} "
            "по месяцам, представленным в датасете: апрель, май и июнь."
        )
# time.sleep(5)
# slt.rerun()
