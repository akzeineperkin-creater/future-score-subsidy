# =============================================================================
#  АгроСубсидия РК — Главная точка входа
#  main.py · Выбор роли · Тёмная тема · Неоновый дизайн
# =============================================================================

import os
import pandas as pd
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ СТРАНИЦЫ (только здесь, один раз)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="АгроСубсидия РК",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ (общий CSV)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "data/applications.csv"
CSV_COLUMNS = [
    "farm_name", "bin", "region", "livestock", "deaths", "death_rate",
    "years_work", "requested_amount", "score", "shap_values", "feature_names",
    "status", "submitted_at", "reviewed_by", "reviewed_at", "review_comment",
]

def init_database():
    """Создаёт директорию и CSV если не существуют."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)

init_database()

# ─────────────────────────────────────────────────────────────────────────────
#  ГЛОБАЛЬНЫЕ СТИЛИ — тёмная тема с неоновыми акцентами
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=Space+Mono:wght@400;700&display=swap');

/* ── Глобальный сброс — светлая чистая тема ── */
html, body, [class*="css"], .stApp {
    background-color: #f7f9f4 !important;
    color: #1a2e0d !important;
    font-family: 'DM Sans', sans-serif !important;
}

header[data-testid="stHeader"] { background: transparent; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ddecd0 !important;
}
[data-testid="stSidebar"] * { color: #3B6D11 !important; }

/* Кнопки */
.stButton > button {
    background: #ffffff !important;
    border: 1px solid #97C459 !important;
    color: #3B6D11 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    border-radius: 10px !important;
    transition: all 0.18s ease !important;
}
.stButton > button:hover {
    background: #EAF3DE !important;
    border-color: #3B6D11 !important;
}
.stButton > button[kind="primary"] {
    background: #3B6D11 !important;
    color: #EAF3DE !important;
    border: none !important;
}
.stButton > button[kind="primary"]:hover {
    background: #2D5016 !important;
}

/* Инпуты */
.stTextInput > div > div > input,
.stNumberInput > div > div > input,
.stSelectbox > div > div {
    background: #ffffff !important;
    border: 1px solid #C0DD97 !important;
    border-radius: 10px !important;
    color: #1a2e0d !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #3B6D11 !important;
    box-shadow: 0 0 0 3px rgba(59,109,17,0.1) !important;
}

/* Табы */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid #ddecd0 !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    color: #639922 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
}
.stTabs [aria-selected="true"] {
    color: #2D5016 !important;
    border-bottom: 2px solid #3B6D11 !important;
    font-weight: 500 !important;
}

/* Экспандеры */
.streamlit-expanderHeader {
    background: #ffffff !important;
    border: 1px solid #ddecd0 !important;
    border-radius: 10px !important;
    color: #3B6D11 !important;
    font-family: 'DM Sans', sans-serif !important;
}

/* Метрики */
[data-testid="stMetricValue"] {
    color: #3B6D11 !important;
    font-family: 'Space Mono', monospace !important;
}
[data-testid="stMetricLabel"] { color: #639922 !important; }

/* Датафреймы */
[data-testid="stDataFrame"] {
    background: #ffffff !important;
    border: 1px solid #ddecd0 !important;
    border-radius: 10px !important;
}

/* Разделители */
hr { border-color: #ddecd0 !important; }

/* ── Приветственный экран ── */
.hero-container {
    min-height: 88vh;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    text-align: center;
    padding: 40px 20px;
}

.hero-badge {
    display: inline-block;
    background: #EAF3DE;
    border: 1px solid #C0DD97;
    color: #3B6D11;
    font-family: 'Space Mono', monospace;
    font-size: 10px;
    letter-spacing: 2.5px;
    text-transform: uppercase;
    padding: 5px 16px;
    border-radius: 999px;
    margin-bottom: 24px;
}

.hero-title {
    font-family: 'DM Sans', sans-serif;
    font-size: clamp(30px, 5vw, 54px);
    font-weight: 300;
    color: #1a2e0d;
    line-height: 1.1;
    margin-bottom: 6px;
    letter-spacing: -1px;
}

.hero-title span {
    font-weight: 500;
    color: #3B6D11;
}

.hero-subtitle {
    font-size: 15px;
    color: #639922;
    margin-bottom: 52px;
    max-width: 460px;
    line-height: 1.6;
    font-weight: 300;
}

/* ── Карточки выбора роли ── */
.role-card {
    background: #ffffff;
    border: 1px solid #ddecd0;
    border-radius: 16px;
    padding: 28px 24px;
    text-align: left;
    transition: border-color 0.18s ease, box-shadow 0.18s ease;
}

.role-card:hover {
    border-color: #97C459;
    box-shadow: 0 4px 24px rgba(59,109,17,0.08);
}

.role-icon {
    font-size: 32px;
    margin-bottom: 14px;
    display: block;
}

.role-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 16px;
    font-weight: 500;
    color: #1a2e0d;
    margin-bottom: 6px;
}

.role-desc {
    font-size: 13px;
    color: #639922;
    line-height: 1.55;
    margin-bottom: 18px;
    font-weight: 300;
}

.role-tag {
    display: inline-block;
    background: #EAF3DE;
    border: 1px solid #C0DD97;
    color: #27500A;
    font-family: 'Space Mono', monospace;
    font-size: 9px;
    letter-spacing: 1px;
    padding: 3px 9px;
    border-radius: 4px;
}

/* ── Статистика ── */
.stats-row {
    display: flex;
    gap: 40px;
    justify-content: center;
    margin-top: 8px;
}

.stat-item { text-align: center; }

.stat-value {
    font-family: 'Space Mono', monospace;
    font-size: 22px;
    font-weight: 700;
    color: #3B6D11;
}

.stat-label {
    font-size: 11px;
    color: #97C459;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-top: 4px;
    font-family: 'Space Mono', monospace;
}

.dot-divider { color: #ddecd0; font-size: 20px; margin-top: 16px; }

.role-card-active { border-color: #3B6D11 !important; }

.role-btn-official {
    background: linear-gradient(135deg, rgba(52,152,219,0.15), rgba(41,128,185,0.08)) !important;
    border: 1.5px solid #3498db !important;
    color: #3498db !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None

# ─────────────────────────────────────────────────────────────────────────────
#  ФУНКЦИЯ ВЫБОРА РОЛИ (главный экран)
# ─────────────────────────────────────────────────────────────────────────────
def show_role_selector():
    """Отображает приветственный экран с выбором роли."""

    # Читаем статистику из CSV
    try:
        df_stats = pd.read_csv(CSV_PATH) if os.path.exists(CSV_PATH) else pd.DataFrame()
        total_apps    = len(df_stats)
        approved_apps = len(df_stats[df_stats["status"] == "approved"]) if not df_stats.empty else 0
        pending_apps  = len(df_stats[df_stats["status"] == "pending"])  if not df_stats.empty else 0
    except Exception:
        total_apps = approved_apps = pending_apps = 0

    st.markdown(f"""
<div class="hero-container">
    <div class="hero-title">Агро<span>Субсидия</span></div>
    <div style="font-size:clamp(13px,1.8vw,17px);color:#97C459;font-weight:300;
                margin-bottom:10px;letter-spacing:0.01em;">
        Цифровой аудит субсидий · Республика Казахстан
    </div>
    <div class="hero-subtitle">
        Прозрачная платформа для подачи заявок<br>и государственного аудита с применением ИИ
    </div>
    <div style="font-size:11px;color:#C0DD97;text-transform:uppercase;
                letter-spacing:2px;margin-bottom:20px;font-family:'Space Mono',monospace;">
        Выберите режим входа
    </div>
</div>
""", unsafe_allow_html=True)

    # Центрированные кнопки выбора роли
    c_pad1, c_farmer, c_mid, c_official, c_pad2 = st.columns([1, 2, 0.5, 2, 1])

    with c_farmer:
        st.markdown("""
<div class="role-card">
    <span class="role-icon">🌾</span>
    <div class="role-title">Кабинет фермера</div>
    <div class="role-desc">
        Подача заявки на субсидию. Автоматический XGBoost-скоринг.
        SHAP-объяснение решения. Отслеживание статуса.
    </div>
    <span class="role-tag">FARMER PORTAL</span>
</div>
""", unsafe_allow_html=True)
        if st.button("🌾  Войти как Фермер", use_container_width=True, key="btn_farmer"):
            st.session_state.role = "farmer"
            st.rerun()

    with c_mid:
        st.markdown("""
<div style="display:flex;align-items:center;justify-content:center;height:100%;padding-top:40px;">
    <div style="color:#ddecd0;font-size:24px;font-weight:300;">|</div>
</div>
""", unsafe_allow_html=True)

    with c_official:
        st.markdown("""
<div class="role-card" style="border-color:#ddecd0;">
    <span class="role-icon">🏛</span>
    <div class="role-title">Кабинет аудитора</div>
    <div class="role-desc">
        Просмотр и анализ заявок. Одобрение и отклонение.
        KPI-дашборд. Журнал аудита. Управление инспекторами.
    </div>
    <span class="role-tag" style="background:#f7f9f4;border-color:#ddecd0;color:#639922;">AUDITOR PORTAL</span>
</div>
""", unsafe_allow_html=True)
        if st.button("🏛  Войти как Аудитор", use_container_width=True, key="btn_official"):
            st.session_state.role = "official_pending"
            st.rerun()

    # Статистика внизу
    st.markdown(f"""
<div style="text-align:center; margin-top:36px;">
    <div class="stats-row">
        <div class="stat-item">
            <div class="stat-value">{total_apps}</div>
            <div class="stat-label">Заявок в системе</div>
        </div>
        <div class="stat-item" style="border-left:1px solid #ddecd0; padding-left:40px;">
            <div class="stat-value" style="color:#27500A;">{approved_apps}</div>
            <div class="stat-label">Одобрено</div>
        </div>
        <div class="stat-item" style="border-left:1px solid #ddecd0; padding-left:40px;">
            <div class="stat-value" style="color:#BA7517;">{pending_apps}</div>
            <div class="stat-label">На рассмотрении</div>
        </div>
    </div>
    <div style="margin-top:24px;font-size:10px;color:#C0DD97;
                font-family:'Space Mono',monospace;letter-spacing:1.5px;">
        СИСТЕМА АКТИВНА · data/applications.csv
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR — кнопка смены роли
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
<div style="padding:14px 0 8px 0;">
    <div style="font-family:'DM Sans',sans-serif;font-size:17px;font-weight:500;
                color:#2D5016;letter-spacing:-0.3px;">АгроСубсидия</div>
    <div style="font-size:10px;color:#97C459;letter-spacing:1.5px;margin-top:2px;
                font-family:'Space Mono',monospace;">МСХ РК · DECENTRATHON 5.0</div>
</div>
""", unsafe_allow_html=True)

    st.divider()

    if st.session_state.role:
        role_label = (
            "🌾 Фермер" if st.session_state.role == "farmer"
            else "🏛 Аудитор" if st.session_state.role == "official"
            else "🔐 Вход аудитора..."
        )
        st.markdown(f"""
<div style="background:#EAF3DE;border:1px solid #C0DD97;
            border-radius:10px;padding:10px 14px;margin-bottom:12px;">
    <div style="font-size:10px;color:#639922;text-transform:uppercase;
                letter-spacing:1px;font-family:'Space Mono',monospace;">Текущий режим</div>
    <div style="font-size:15px;color:#2D5016;font-weight:500;margin-top:4px;">{role_label}</div>
</div>
""", unsafe_allow_html=True)
        if st.button("← Сменить роль", use_container_width=True):
            st.session_state.role = None
            for key in ["authenticated", "current_user"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    st.divider()
    st.markdown("""
<div style="font-size:11px;color:#97C459;line-height:2;font-family:'Space Mono',monospace;">
    <div>📁 data/applications.csv</div>
    <div>🔄 синхронизация активна</div>
    <div style="margin-top:6px;color:#C0DD97;">v1.0 · 2025</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  ЭКРАН ВВОДА ПАРОЛЯ АУДИТОРА
# ─────────────────────────────────────────────────────────────────────────────
AUDITOR_PASSWORD = "admin777"

def show_auditor_login():
    """Экран проверки пароля для аудитора."""
    _, col, _ = st.columns([1, 1.2, 1])
    with col:
        st.markdown("""
<div style="text-align:center;padding:64px 0 36px 0;">
    <div style="width:64px;height:64px;border-radius:16px;background:#EAF3DE;
                border:1px solid #C0DD97;display:flex;align-items:center;
                justify-content:center;margin:0 auto 20px;font-size:28px;">🏛</div>
    <div style="font-family:'DM Sans',sans-serif;font-size:22px;font-weight:500;
                color:#1a2e0d;margin-bottom:6px;">Кабинет аудитора</div>
    <div style="font-size:13px;color:#639922;margin-bottom:32px;font-weight:300;">
        Доступ ограничен. Введите пароль для входа.
    </div>
</div>
""", unsafe_allow_html=True)

        pwd = st.text_input(
            "Пароль",
            type="password",
            placeholder="Введите пароль аудитора",
            key="auditor_pwd_input",
        )

        c1, c2 = st.columns(2)
        with c1:
            if st.button("🔓 Войти", type="primary", use_container_width=True, key="btn_pwd_enter"):
                if pwd == AUDITOR_PASSWORD:
                    st.session_state.role = "official"
                    st.rerun()
                else:
                    st.error("Неверный пароль. Попробуйте ещё раз.")
        with c2:
            if st.button("← Назад", use_container_width=True, key="btn_pwd_back"):
                st.session_state.role = None
                st.rerun()


# ─────────────────────────────────────────────────────────────────────────────
#  МАРШРУТИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.role is None:
    show_role_selector()

elif st.session_state.role == "official_pending":
    show_auditor_login()

elif st.session_state.role == "farmer":
    import farmer_cabinet
    farmer_cabinet.main()

elif st.session_state.role == "official":
    import app
    app.main()
