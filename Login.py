import streamlit as slt
from sqlalchemy import URL, create_engine, text
from src.config import settings
from datetime import datetime, timedelta
import uuid
from src.auth import init_session_state, require_login, login_user
import time
#------------------------------------------------------------------------

engine = create_engine(
    url=settings.DATABASE_URL_psycopg,
    echo=True,
    pool_size=5,
    max_overflow=10,
)

slt.set_page_config(page_title="Модуль АРК", layout="wide")

main_container = slt.container(
    height=500,
    key='form',
    horizontal_alignment='center',
    vertical_alignment='center',
    border=False
)

with main_container:
    content_container = slt.container(
        width=400,
        gap='medium',)

with content_container:
    field_container = slt.container()
    button_row = slt.container(
        horizontal=True,
        horizontal_alignment='distribute',
    )

with field_container:
    slt.header('Войдите в систему')
    e_log = slt.text_input('Имя пользователя', placeholder='Имя пользователя')
    e_pass = slt.text_input('Пароль', placeholder='Пароль', type='password')

with slt.sidebar:
    slt.write("**Статус аккаунта:**")
    

with button_row:
    if slt.button('Вход', type='primary'):
        # --- Используем login_user из auth.py ---
        from src.auth import login_user

        # Вызов функции login_user проверяет логин/пароль, создаёт сессию и обновляет session_state
        success = login_user(e_log, e_pass)

        with slt.sidebar:
            if success:
                slt.success("Вы вошли в систему")
            else:
                slt.error(" Неправильный логин или пароль")