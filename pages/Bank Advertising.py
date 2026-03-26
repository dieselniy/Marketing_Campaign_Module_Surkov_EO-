# page2.py
import streamlit as slt
from src.auth import init_session_state, require_login, check_session
import time


# --- Инициализация session_state ---
init_session_state()

# --- Восстановление session_id из query params ---
params = slt.query_params  # новый метод вместо experimental_get_query_params
if "session_id" in params:
    slt.session_state.session_id = params["session_id"][0]
    if check_session():
        slt.session_state.authenticated = True

# --- Проверка авторизации ---
#require_login()  # пропускает, если пользователь уже залогинен

# --- Контент страницы ---
slt.header("Аналитика Рекламной Кампании Банка-X")

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

#time.sleep(5)
#slt.rerun()