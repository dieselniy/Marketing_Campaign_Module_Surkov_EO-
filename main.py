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
