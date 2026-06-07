import streamlit as slt


slt.set_page_config(
    page_title='Модуль анализа результатов рекламной кампании',
    layout='wide',
)


main_container = slt.container(
    height=500,
    horizontal_alignment='center',
    vertical_alignment='center',
    border=False,
)

with main_container:
    content_container = slt.container(
        width=900,
        gap='medium',
        border=False,
    )

    with content_container:
        slt.title('Добро пожаловать на веб-ресурс "Модуль анализа результатов рекламной кампании"')

        slt.markdown(
            """
            <div style="font-size: 20px; line-height: 1.6;">
                <p>Модуль состоит из нескольких функциональных страниц.</p>
                <p>
                    Первая, <strong>Bank Advertising</strong>, посвящена анализу данных
                    рекламной кампании банка X. Датасет содержит в себе пользователей,
                    взаимодействовавших с кампанией банка на протяжении одного года.
                    Цель кампании - открытие вклада в банке (deposit). Страница содержит
                    в себе инструменты аналитики и созданную на основе проанализированных
                    данных из датасета модель машинного обучения.
                </p>
                <p>
                    Вторая страница <strong>Online Advertising</strong> также содержит
                    в себе инструменты аналитики, созданные для анализа нескольких
                    онлайн-кампаний длительностью 3 месяца. Цель кампании - возврат инвестиций. 
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def random_forest_importance(frame, target, drop_columns=None, top_n=None):
    drop_columns = list(drop_columns or [])
    x_data = frame.drop(columns=drop_columns + [target], errors="ignore")
    y_data = frame[target]

    prepared = x_data.replace([npy.inf, -npy.inf], npy.nan).copy()

    for column in prepared.columns:
        if pds.api.types.is_numeric_dtype(prepared[column]):
            median_value = prepared[column].median()
            fill_value = 0 if pds.isna(median_value) else median_value
            prepared[column] = prepared[column].fillna(fill_value)
        else:
            prepared[column] = prepared[column].fillna("")

    x_model = pds.get_dummies(prepared)

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(x_model, y_data)

    factors = (
        pds.Series(
            model.feature_importances_,
            index=x_model.columns,
            name="importance",
        )
        .sort_values(ascending=False)
    )

    if top_n is not None:
        factors = factors.head(top_n)

    return (
        factors
        .reset_index()
        .rename(columns={"index": "feature"})
    )


balance_work = work.copy()
balance_work["balance"] = numeric_from("balance").fillna(0)

low_balance_limit = balance_work["balance"].quantile(0.25)

low_balance_work = work.copy()
low_balance_work["balance"] = balance_work["balance"]

low_balance_work["low_balance_flag"] = (
    low_balance_work["balance"] <= low_balance_limit
).astype(int)

low_balance_factors = random_forest_importance(
    frame=low_balance_work,
    target="low_balance_flag",
    drop_columns=["balance"],
    top_n=10,
)

with balance_reason_panel:
    low_balance_factors = bank_stats["low_balance_factor_importance"]
    low_balance_limit = bank_stats["low_balance_limit"]
    low_balance_segments = bank_stats["low_balance_segments"]

    slt.subheader("Факторы низкого баланса")

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
            labels={
                "Признак": "Признак",
                "Важность": "Важность",
            },
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
