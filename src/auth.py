import streamlit as slt
from sqlalchemy import create_engine, text
from datetime import datetime, timedelta
import uuid
from src.config import settings

# --- Подключение к БД ---
engine = create_engine(
    url=settings.DATABASE_URL_psycopg,
    echo=True,
    pool_size=5,
    max_overflow=10,
)

# --- Инициализация session_state ---
def init_session_state():
    if "authenticated" not in slt.session_state:
        slt.session_state.authenticated = False
    if "session_id" not in slt.session_state:
        slt.session_state.session_id = None
    if "username" not in slt.session_state:
        slt.session_state.username = None

# --- Проверка активной сессии ---
def check_session():
    with engine.begin() as conn:
        conn.execute(text("DELETE FROM sessions WHERE expires_at < NOW()"))
    if slt.session_state.session_id is None:
        return False
    with engine.connect() as conn:
        res = conn.execute(
            text("""
                SELECT user_id
                FROM sessions
                WHERE session_id = :sid
                  AND expires_at > NOW()
            """),
            {"sid": slt.session_state.session_id}
        ).fetchone()
        return bool(res)

# --- logout ---
def logout():
    if slt.session_state.session_id:
        with engine.begin() as conn:  # ✅ FIX: begin вместо connect
            conn.execute(
                text("DELETE FROM sessions WHERE session_id = :sid"),
                {"sid": slt.session_state.session_id}
            )

    slt.session_state.authenticated = False
    slt.session_state.session_id = None
    slt.session_state.username = None
    slt.rerun()  # ✅ FIX: вместо experimental_rerun

# --- login ---
def login_user(login: str, password: str):
    with engine.connect() as conn:
        user = conn.execute(
            text("""
                SELECT user_id
                FROM Users
                WHERE login = :login
                  AND password = :password
                LIMIT 1
            """),
            {"login": login, "password": password}
        ).fetchone()

    if not user:
        return False

    user_id = user[0]
    session_id = str(uuid.uuid4())

    with engine.begin() as conn:
        # ✅ Чистим старые сессии
        conn.execute(text("DELETE FROM sessions WHERE expires_at < NOW()"))

        # ✅ Создаём новую
        conn.execute(
            text("""
                INSERT INTO sessions (session_id, user_id, created_at, expires_at)
                VALUES (:sid, :uid, NOW(), NOW() + INTERVAL '1 hour')
            """),
            {"sid": session_id, "uid": user_id}
        )

    slt.session_state.authenticated = True
    slt.session_state.session_id = session_id
    slt.session_state.username = login

    return True


# --- защита страницы ---
def require_login():
    init_session_state()
    # если нет сессии или она истекла
    if not slt.session_state.authenticated or not check_session():
        # ❗ сбрасываем состояние (важно при истекшей сессии)
        slt.session_state.authenticated = False
        slt.session_state.session_id = None
        slt.session_state.username = None

        slt.error("🔒 Войдите в систему, чтобы получить доступ к этой странице!")
        slt.stop()