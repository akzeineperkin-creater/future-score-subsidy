# =============================================================================
#  МСХ РК — Кабинет цифрового аудитора субсидий
#  Decentrathon 5.0 · Участник №2 · Backend & Digital Auditor
# =============================================================================

import os
import datetime
import random
import hashlib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ─────────────────────────────────────────────────────────────────────────────
#  ОБЩИЙ CSV-ФАЙЛ (синхронизация с фермером)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "data/applications.csv"

def ensure_csv_exists():
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(CSV_PATH):
        cols = [
            "farm_name","bin","iin","email","phone","region",
            "livestock","hectares","deaths","death_rate",
            "years_work","requested_amount","score","shap_values","feature_names",
            "status","submitted_at","reviewed_by","reviewed_at","review_comment",
        ]
        pd.DataFrame(columns=cols).to_csv(CSV_PATH, index=False)

def load_farmer_applications() -> pd.DataFrame:
    """Загружает реальные заявки фермеров из CSV.
    Нормализует колонки: main.py пишет cows_count/mortality_rate/district,
    app.py ожидает livestock/death_rate. Маппинг делается здесь.
    """
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_PATH)
        if df.empty:
            return pd.DataFrame()

        # Маппинг колонок main.py -> app.py
        rename_map = {
            "cows_count":     "livestock",
            "mortality_rate": "death_rate",
        }
        df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

        # deaths вычисляем из livestock * death_rate, если колонки deaths нет
        if "deaths" not in df.columns:
            if "livestock" in df.columns and "death_rate" in df.columns:
                df["deaths"] = (
                    pd.to_numeric(df["livestock"], errors="coerce").fillna(0) *
                    pd.to_numeric(df["death_rate"], errors="coerce").fillna(0)
                ).astype(int)
            else:
                df["deaths"] = 0

        return df
    except Exception:
        return pd.DataFrame()

def update_application_status_in_csv(bin_val: str, submitted_at: str,
                                       new_status: str, reviewer: str):
    """Обновляет статус конкретной заявки в CSV."""
    ensure_csv_exists()
    try:
        df = pd.read_csv(CSV_PATH)
        mask = (df["bin"].astype(str) == str(bin_val)) & \
               (df["submitted_at"].astype(str) == str(submitted_at))
        if mask.any():
            df.loc[mask, "status"]      = new_status
            df.loc[mask, "reviewed_by"] = reviewer
            df.loc[mask, "reviewed_at"] = datetime.datetime.now().isoformat()
            df.to_csv(CSV_PATH, index=False)
    except Exception as e:
        st.error(f"Ошибка обновления CSV: {e}")

# ─────────────────────────────────────────────────────────────────────────────
#  КОНФИГУРАЦИЯ (set_page_config вынесен в main.py)
# ─────────────────────────────────────────────────────────────────────────────

MORTALITY_LIMIT  = 2.0      # % — Приказ №2
TOTAL_BUDGET     = 120_000_000
PASSWORD         = "admin777"

# ─────────────────────────────────────────────────────────────────────────────
#  СТИЛИ (вызываются внутри main())
# ─────────────────────────────────────────────────────────────────────────────
_APP_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=Space+Mono:wght@400;700&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f9f4;
    color: #1a2e0d;
}

/* Главная шапка */
.msh-header {
    background: #2D5016;
    padding: 20px 28px;
    border-radius: 14px;
    color: #EAF3DE;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.msh-header::before {
    content: '';
    position: absolute;
    top: -50px; right: -50px;
    width: 200px; height: 200px;
    background: rgba(160,220,100,0.07);
    border-radius: 50%;
}
.msh-header::after {
    content: '';
    position: absolute;
    bottom: -30px; right: 120px;
    width: 120px; height: 120px;
    background: rgba(160,220,100,0.05);
    border-radius: 50%;
}
.msh-header h1 { font-size: 20px; font-weight: 500; margin: 0 0 4px 0; color: #EAF3DE; }
.msh-header p  { font-size: 12px; opacity: 0.7; margin: 0; font-family: 'Space Mono', monospace; }

/* KPI карточки */
.kpi-card {
    background: #ffffff;
    border: 1px solid #ddecd0;
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.kpi-card .kpi-label {
    font-size: 11px; color: #97C459; font-weight: 400;
    text-transform: uppercase; letter-spacing: 1px;
    font-family: 'Space Mono', monospace;
}
.kpi-card .kpi-value {
    font-size: 22px; font-weight: 700; color: #2D5016;
    margin: 4px 0; font-family: 'Space Mono', monospace;
}
.kpi-card .kpi-sub { font-size: 11px; color: #C0DD97; }

/* Строки таблицы */
.row-danger  { background: #fff5f5 !important; border-left: 3px solid #E24B4A !important; }
.row-warning { background: #fffbf0 !important; border-left: 3px solid #BA7517 !important; }
.row-ok      { background: #f4faf0 !important; border-left: 3px solid #639922 !important; }

/* Статус-бейджи */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 500;
    font-family: 'Space Mono', monospace;
}
.badge-approved { background: #EAF3DE; color: #27500A; }
.badge-rejected { background: #FCEBEB; color: #A32D2D; }
.badge-pending  { background: #FAEEDA; color: #633806; }
.badge-blocked  { background: #FCEBEB; color: #A32D2D; }

/* Блокировка */
.blocked-banner {
    background: #FCEBEB;
    border: 1px solid #F7C1C1;
    border-left: 3px solid #E24B4A;
    border-radius: 0 8px 8px 0;
    padding: 10px 16px;
    color: #A32D2D;
    font-size: 13px;
    font-weight: 500;
}

/* Анонимный ID */
.anon-id {
    font-family: 'Space Mono', monospace;
    background: #EAF3DE;
    color: #27500A;
    padding: 2px 8px;
    border-radius: 4px;
    font-size: 12px;
    letter-spacing: 0.5px;
}

/* Приоритет-слайдер метки */
.priority-label {
    font-size: 13px;
    font-weight: 500;
    color: #3B6D11;
    margin-bottom: 2px;
}

/* Карточка профиля в sidebar */
.profile-card {
    background: #ffffff;
    border: 2px solid #3B6D11;
    border-radius: 12px;
    padding: 16px 18px;
    color: #1a2e0d;
    margin-bottom: 4px;
    position: relative;
    overflow: hidden;
    box-shadow: 0 2px 12px rgba(59,109,17,0.15);
}
.profile-card::after {
    content: '';
    position: absolute;
    bottom: -20px; right: -20px;
    width: 80px; height: 80px;
    background: rgba(59,109,17,0.05);
    border-radius: 50%;
}
.profile-name  { font-size: 15px; font-weight: 600; margin-bottom: 2px; color: #1a2e0d !important; }
.profile-role  { font-size: 10px; opacity: 1; text-transform: uppercase;
                 letter-spacing: 1.5px; margin-bottom: 7px;
                 font-family: 'Space Mono', monospace; color: #3B6D11 !important; }
.profile-dept  {
    display: inline-block;
    background: #EAF3DE;
    border-radius: 6px;
    padding: 2px 10px;
    font-size: 11px; font-weight: 500;
    border: 1px solid #97C459;
    color: #2D5016 !important;
}
.profile-online {
    display: flex; align-items: center; gap: 5px;
    font-size: 11px; margin-top: 9px;
    font-family: 'Space Mono', monospace;
    color: #3B6D11;
}
.profile-dot {
    width: 6px; height: 6px;
    background: #27ae60; border-radius: 50%;
    flex-shrink: 0;
}

/* Журнал аудита */
.audit-approve { color: #27500A; font-weight: 500; }
.audit-reject  { color: #A32D2D; font-weight: 500; }
</style>
"""


# ─────────────────────────────────────────────────────────────────────────────
#  БАЗА ДАННЫХ ПОЛЬЗОВАТЕЛЕЙ И АВТОРИЗАЦИЯ
# ─────────────────────────────────────────────────────────────────────────────
DEPARTMENTS = [
    "Департамент субсидирования АПК",
    "Отдел животноводства",
    "Комитет государственного контроля",
    "Финансово-бюджетный отдел",
    "Отдел цифровизации МСХ",
    "Региональный инспекторат",
]

def hash_pwd(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

# Инициализация БД пользователей с дефолтным аккаунтом admin

# ─────────────────────────────────────────────────────────────────────────────
#  ТОЧКА ВХОДА
# ─────────────────────────────────────────────────────────────────────────────
def main():
    # ── Применяем стили ──────────────────────────────────────────────────────
    st.markdown(_APP_CSS, unsafe_allow_html=True)

    if "users_db" not in st.session_state:
        st.session_state.users_db = {
            "admin": {
                "password_hash": hash_pwd("admin777"),
                "full_name":     "Администратор Системы",
                "department":    "Отдел цифровизации МСХ",
                "role":          "Главный инспектор",
                "created_at":    datetime.datetime.now().strftime("%d.%m.%Y"),
            }
        }

    if "authenticated" not in st.session_state:
        st.session_state.authenticated = False
    if "current_user" not in st.session_state:
        st.session_state.current_user = None

    # ── Экран входа / регистрации ─────────────────────────────────────────────────
    if not st.session_state.authenticated:
        col_l, col_m, col_r = st.columns([1, 1.4, 1])
        with col_m:
            st.markdown("""
            <div style="text-align:center; padding: 48px 0 24px 0;">
                <span style="font-size:52px;">🌾</span>
                <h2 style="color:#0f2744; margin: 10px 0 5px 0; font-size:22px;">МСХ РК — Кабинет аудитора</h2>
                <p style="color:#64748b; font-size:13px; margin:0 0 28px 0;">
                    Министерство сельского хозяйства Республики Казахстан<br>
                    Цифровой аудит субсидий · Decentrathon 5.0
                </p>
            </div>
            """, unsafe_allow_html=True)

            auth_tab_login, auth_tab_reg = st.tabs(["🔑  Вход", "📝  Регистрация"])

            # ── ВХОД ──
            with auth_tab_login:
                st.write("")
                login_username = st.text_input("Логин", placeholder="Введите логин",
                                               key="login_user")
                login_password = st.text_input("Пароль", type="password",
                                               placeholder="Введите пароль", key="login_pwd")
                st.write("")
                if st.button("Войти в систему", type="primary",
                             use_container_width=True, key="btn_login"):
                    users_db = st.session_state.users_db
                    if login_username in users_db:
                        if users_db[login_username]["password_hash"] == hash_pwd(login_password):
                            st.session_state.authenticated = True
                            st.session_state.current_user  = login_username
                            st.rerun()
                        else:
                            st.error("❌ Неверный пароль.")
                    else:
                        st.error("❌ Пользователь не найден.")
                st.caption("🔒 Все действия инспектора журналируются в системе.")

            # ── РЕГИСТРАЦИЯ ──
            with auth_tab_reg:
                st.write("")
                reg_login    = st.text_input("Логин *", placeholder="латиница, без пробелов",
                                             key="reg_login")
                reg_fullname = st.text_input("ФИО *", placeholder="Фамилия Имя Отчество",
                                             key="reg_name")
                reg_dept     = st.selectbox("Департамент *", DEPARTMENTS, key="reg_dept")
                reg_pwd      = st.text_input("Пароль *", type="password",
                                             placeholder="Минимум 6 символов", key="reg_pwd")
                reg_pwd2     = st.text_input("Подтвердите пароль *", type="password",
                                             placeholder="Повторите пароль", key="reg_pwd2")
                st.write("")
                if st.button("Зарегистрироваться", type="primary",
                             use_container_width=True, key="btn_register"):
                    errors = []
                    if not reg_login.strip():
                        errors.append("Введите логин.")
                    elif reg_login in st.session_state.users_db:
                        errors.append("Логин уже занят.")
                    elif not reg_login.isascii() or " " in reg_login:
                        errors.append("Логин — только латиница без пробелов.")
                    if not reg_fullname.strip():
                        errors.append("Введите ФИО.")
                    if len(reg_pwd) < 6:
                        errors.append("Пароль не менее 6 символов.")
                    if reg_pwd != reg_pwd2:
                        errors.append("Пароли не совпадают.")

                    if errors:
                        for e in errors:
                            st.error(f"❌ {e}")
                    else:
                        st.session_state.users_db[reg_login] = {
                            "password_hash": hash_pwd(reg_pwd),
                            "full_name":     reg_fullname.strip(),
                            "department":    reg_dept,
                            "role":          "Инспектор",
                            "created_at":    datetime.datetime.now().strftime("%d.%m.%Y"),
                        }
                        st.success(f"✅ Аккаунт «{reg_login}» создан! Теперь войдите на вкладке «Вход».")
        st.stop()


    # ─────────────────────────────────────────────────────────────────────────────
    #  ГЕНЕРАЦИЯ ДЕМО-ДАННЫХ (имитация 'Выгрузка по выданным субсидиям 2025')
    # ─────────────────────────────────────────────────────────────────────────────
    @st.cache_data
    def generate_demo_data() -> pd.DataFrame:
        random.seed(42)
        REGIONS = [
            "Алматинская", "Акмолинская", "Жамбылская", "Туркестанская",
            "ВКО", "СКО", "Костанайская", "Павлодарская", "Карагандинская", "ЗКО",
        ]
        DISTRICTS = {
            "Алматинская": ["Алакольский", "Балхашский", "Карасайский", "Талгарский"],
            "Акмолинская": ["Атбасарский", "Буландынский", "Целиноградский", "Есильский"],
            "Жамбылская":  ["Байзакский", "Жамбылский", "Меркенский", "Таласский"],
            "Туркестанская": ["Арысский", "Кентауский", "Сарыагашский", "Шардаринский"],
            "ВКО":         ["Абайский", "Аягозский", "Катон-Карагайский", "Зыряновский"],
            "СКО":         ["Айыртауский", "Акжарский", "Есильский", "Мамлютский"],
            "Костанайская":["Аулиекольский", "Денисовский", "Костанайский", "Рудненский"],
            "Павлодарская":["Актогайский", "Баянаульский", "Иртышский", "Павлодарский"],
            "Карагандинская":["Абайский", "Каркаралинский", "Нуринский", "Шетский"],
            "ЗКО":         ["Акжаикский", "Бурлинский", "Казталовский", "Теректинский"],
        }
        DIRECTIONS = ["Молочное скотоводство", "Мясное скотоводство", "Племенное дело",
                      "Овцеводство", "Птицеводство", "Молочно-мясное"]
        KH_PREFIXES = ["КХ", "ТОО", "ИП", "АФ"]
        KH_NAMES = [
            "Болашак", "Нұрлы жол", "АгроСтеп", "Береке", "Жасыл дала",
            "Алтын бидай", "Арман", "Даму", "Жетісу", "Сарыарқа",
            "Өрлеу", "Мереке", "Тулпар", "Байтерек", "Табыс",
            "Атамекен", "Шапағат", "Ұлан", "Ынтымақ", "Даланың ұлы",
            "Агро-Батыс", "Степной", "Казагро", "АлтайАгро", "Примула",
        ]

        rows = []
        for i in range(120):
            region   = random.choice(REGIONS)
            district = random.choice(DISTRICTS[region])
            direction = random.choices(
                DIRECTIONS,
                weights=[25, 30, 15, 15, 5, 10]
            )[0]

            # Поголовье и падёж
            livestock = random.randint(50, 800)
            # ~12% хозяйств — нарушители по падежу
            if random.random() < 0.12:
                mort_pct = round(random.uniform(2.1, 12.0), 2)
            else:
                mort_pct = round(random.uniform(0.0, 1.9), 2)
            mort_head = int(livestock * mort_pct / 100)

            # Базовая сумма субсидии
            rate_map = {
                "Молочное скотоводство": 18000,
                "Мясное скотоводство":   15000,
                "Племенное дело":        25000,
                "Овцеводство":           8000,
                "Птицеводство":          3000,
                "Молочно-мясное":        16000,
            }
            base_rate = rate_map[direction]
            amount = livestock * base_rate

            # БИН
            bin_num = "".join([str(random.randint(0, 9)) for _ in range(12)])

            # Название
            prefix = random.choice(KH_PREFIXES)
            name   = f"{prefix} «{random.choice(KH_NAMES)}» ({region[:3]})"

            # Исторический рейтинг района
            hist_score = round(random.uniform(0.35, 0.95), 2)
            climate    = round(random.uniform(0.15, 0.85), 2)
            is_breed   = 1 if direction == "Племенное дело" else random.randint(0, 1)
            is_sel     = 1 if is_breed else random.randint(0, 1)

            # Базовый FutureScore
            base_fs = max(10, min(99, int(hist_score * 70 - climate * 30
                                           + is_breed * 15 + is_sel * 10)))

            rows.append({
                "БИН":             bin_num,
                "Название КХ":     name,
                "Регион":          region,
                "Район":           district,
                "Направление":     direction,
                "Поголовье":       livestock,
                "Падёж (гол)":     mort_head,
                "Падёж %":         mort_pct,
                "Причит. сумма":   amount,
                "is_breeding":     is_breed,
                "is_selection":    is_sel,
                "hist_score":      hist_score,
                "climate_risk":    climate,
                "base_score":      base_fs,
                "Статус":          "⏳ На рассмотрении",
            })

        return pd.DataFrame(rows)


    # ─────────────────────────────────────────────────────────────────────────────
    #  SESSION STATE
    # ─────────────────────────────────────────────────────────────────────────────
    if "df_raw" not in st.session_state:
        st.session_state.df_raw = generate_demo_data()

    if "statuses" not in st.session_state:
        st.session_state.statuses = {}   # idx -> "✅ Одобрено" | "❌ Отклонено"

    if "audit_log" not in st.session_state:
        st.session_state.audit_log = []   # список записей журнала


    # ─────────────────────────────────────────────────────────────────────────────
    #  ПЕРЕСЧЁТ SCORE С УЧЁТОМ ПРИОРИТЕТОВ
    # ─────────────────────────────────────────────────────────────────────────────
    def compute_scores(df: pd.DataFrame, p_milk: int, p_meat: int, p_breed: int) -> pd.Series:
        """Пересчитывает FutureScore с учётом приоритетов слайдеров."""
        scores = []
        for _, row in df.iterrows():
            s = row["base_score"]
            direction = row["Направление"]
            # Влияние слайдеров: каждый балл приоритета = ±0.3 к итоговому score
            if "Молоч" in direction:
                s = s + (p_milk - 50) * 0.3
            elif "Мясн" in direction:
                s = s + (p_meat - 50) * 0.3
            elif "Племен" in direction:
                s = s + (p_breed - 50) * 0.3
            scores.append(max(5, min(99, int(s))))
        return pd.Series(scores, index=df.index)


    # ─────────────────────────────────────────────────────────────────────────────
    #  АНОНИМИЗАЦИЯ
    # ─────────────────────────────────────────────────────────────────────────────
    def anonymize_id(val: str, salt: str = "msh2025") -> str:
        h = hashlib.md5(f"{val}{salt}".encode()).hexdigest()[:8].upper()
        return f"ANON-{h}"


    # ─────────────────────────────────────────────────────────────────────────────
    #  SIDEBAR
    # ─────────────────────────────────────────────────────────────────────────────
    with st.sidebar:
        # ── Профиль пользователя ──────────────────────────────────────────────────
        cur_user_data = st.session_state.users_db.get(st.session_state.current_user, {})
        full_name  = cur_user_data.get("full_name", "Инспектор")
        role       = cur_user_data.get("role", "Инспектор")
        department = cur_user_data.get("department", "МСХ РК")
        # Инициалы для аватара
        initials = "".join([w[0].upper() for w in full_name.split()[:2]])

        st.markdown(f"""
        <div class="profile-card">
            <div style="display:flex; align-items:center; gap:12px; margin-bottom:10px;">
                <div style="width:42px; height:42px; border-radius:50%;
                            background: #3B6D11;
                            display:flex; align-items:center; justify-content:center;
                            font-size:16px; font-weight:700; color:#ffffff;
                            border:2px solid #97C459; flex-shrink:0;">
                    {initials}
                </div>
                <div>
                    <div class="profile-name">{full_name}</div>
                    <div class="profile-role">{role}</div>
                </div>
            </div>
            <span class="profile-dept">🏢 {department}</span>
            <div class="profile-online">
                <div class="profile-dot"></div>
                <span>В системе · {datetime.datetime.now().strftime('%H:%M')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        # Приоритеты
        st.markdown("### ⚖️ Приоритеты направлений")
        st.caption("Влияют на итоговый рейтинг (FutureScore) заявок")

        p_milk  = st.slider(" Приоритет: Молоко",   0, 100, 50, step=5)
        p_meat  = st.slider(" Приоритет: Мясо",      0, 100, 50, step=5)
        p_breed = st.slider(" Приоритет: Племенное дело", 0, 100, 50, step=5)

        st.divider()

        # Blind Review
        st.markdown("### 🕶 Режим проверки")
        blind_mode = st.toggle("Режим Blind Review", value=False,
                               help="Скрывает БИН и название КХ — показывает только анонимный ID")
        if blind_mode:
            st.markdown("""
            <div style="background:#1e293b;color:#38bdf8;border-radius:8px;
                        padding:10px 14px;font-size:12px;font-family:'JetBrains Mono',monospace;">
                🔒 BLIND MODE ACTIVE<br>
                <span style="opacity:0.7;">БИН → ANON-XXXXXXXX</span>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        # Фильтры
        st.markdown("### 🔍 Фильтры")
        df_all = st.session_state.df_raw
        regions = ["Все"] + sorted(df_all["Регион"].unique().tolist())
        sel_region = st.selectbox("Регион", regions)
        directions = ["Все"] + sorted(df_all["Направление"].unique().tolist())
        sel_dir = st.selectbox("Направление", directions)
        only_violations = st.checkbox("🚨 Только нарушения (падёж > 2%)", value=False)

        st.divider()

        # Мини-статистика
        total_n = len(df_all)
        approved_n = sum(1 for v in st.session_state.statuses.values() if v == "✅ Одобрено")
        rejected_n = sum(1 for v in st.session_state.statuses.values() if v == "❌ Отклонено")
        violations_n = int((df_all["Падёж %"] > MORTALITY_LIMIT).sum())

        st.markdown(f"""
        <div style="font-size:13px; line-height:2.0;">
        📋 Всего заявок: <b>{total_n}</b><br>
        🚨 Нарушений: <b style="color:#dc2626;">{violations_n}</b><br>
        ✅ Одобрено: <b style="color:#059669;">{approved_n}</b><br>
        ❌ Отклонено: <b style="color:#6b7280;">{rejected_n}</b><br>
        ⏳ Ожидают: <b>{total_n - approved_n - rejected_n}</b>
        </div>
        """, unsafe_allow_html=True)

        st.divider()
        if st.button("🚪 Выйти из системы", use_container_width=True):
            st.session_state.authenticated = False
            st.session_state.current_user  = None
            st.session_state.statuses = {}
            st.rerun()


    # ─────────────────────────────────────────────────────────────────────────────
    #  ШАПКА СТРАНИЦЫ
    # ─────────────────────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="msh-header">
        <h1>🌾 МСХ РК — Кабинет цифрового аудитора субсидий</h1>
        <p>Министерство сельского хозяйства Республики Казахстан &nbsp;·&nbsp;
           Антикоррупционный модуль &nbsp;·&nbsp;
           {datetime.date.today().strftime('%d.%m.%Y')} &nbsp;·&nbsp;
           🔐 {full_name} · {department}</p>
    </div>
    """, unsafe_allow_html=True)


    # ─────────────────────────────────────────────────────────────────────────────
    #  ПЕРЕСЧЁТ ДАННЫХ
    # ─────────────────────────────────────────────────────────────────────────────
    df = st.session_state.df_raw.copy()
    df["FutureScore"] = compute_scores(df, p_milk, p_meat, p_breed)

    # Применяем фильтры
    if sel_region != "Все":
        df = df[df["Регион"] == sel_region]
    if sel_dir != "Все":
        df = df[df["Направление"] == sel_dir]
    if only_violations:
        df = df[df["Падёж %"] > MORTALITY_LIMIT]

    # Получаем статусы
    df["_status"] = df.index.map(lambda i: st.session_state.statuses.get(i, "⏳ На рассмотрении"))
    df["_blocked"] = df["Падёж %"] > MORTALITY_LIMIT


    # ─────────────────────────────────────────────────────────────────────────────
    #  ФИНАНСОВЫЕ KPI
    # ─────────────────────────────────────────────────────────────────────────────
    approved_sum = sum(
        st.session_state.df_raw.loc[i, "Причит. сумма"]
        for i, v in st.session_state.statuses.items()
        if v == "✅ Одобрено" and i in st.session_state.df_raw.index
    )
    requested_sum = df["Причит. сумма"].sum()
    balance = TOTAL_BUDGET - approved_sum
    pct_used = min(approved_sum / TOTAL_BUDGET * 100, 100) if TOTAL_BUDGET > 0 else 0

    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("💰 Общий бюджет",    f"{TOTAL_BUDGET:,.0f} ₸")
    k2.metric("📥 Запрошено (фильтр)", f"{requested_sum:,.0f} ₸")
    k3.metric("✅ Одобрено итого",  f"{approved_sum:,.0f} ₸",
              delta=f"{pct_used:.1f}% бюджета", delta_color="inverse" if pct_used > 80 else "off")
    k4.metric("💵 Остаток бюджета", f"{balance:,.0f} ₸",
              delta="⚠️ Критично" if balance < TOTAL_BUDGET * 0.1 else None,
              delta_color="inverse")
    k5.metric("🚨 Блокировок",      f"{int((st.session_state.df_raw['Падёж %'] > MORTALITY_LIMIT).sum())}",
              delta="Приказ №2", delta_color="inverse")

    # Прогресс-бар бюджета
    st.progress(pct_used / 100, text=f"Использовано бюджета: {pct_used:.1f}%")
    st.divider()


    # ─────────────────────────────────────────────────────────────────────────────
    #  ВКЛАДКИ
    # ─────────────────────────────────────────────────────────────────────────────
    tab_registry, tab_analytics, tab_auto, tab_farmer, tab_audit = st.tabs([
        "📋 Реестр заявок",
        "📊 Аналитика",
        "⚡ Авто-распределение",
        "🌾 Заявки фермеров",
        "📜 Журнал аудита",
    ])


    # ══════════════════════════════════════════════════════════════════════════════
    #  ТАБ 1 — РЕЕСТР
    # ══════════════════════════════════════════════════════════════════════════════
    with tab_registry:
        st.subheader(f"📋 Реестр субсидий {'🕶 [BLIND REVIEW]' if blind_mode else ''}")

        if df.empty:
            st.info("Нет записей по выбранным фильтрам.")
            st.stop()

        # Сортировка по FutureScore
        df_sorted = df.sort_values("FutureScore", ascending=False).reset_index()

        for _, row in df_sorted.iterrows():
            orig_idx   = row["index"]
            status     = st.session_state.statuses.get(orig_idx, "⏳ На рассмотрении")
            blocked    = bool(row["_blocked"])
            score      = int(row["FutureScore"])
            mort       = float(row["Падёж %"])
            amount     = float(row["Причит. сумма"])
            direction  = row["Направление"]

            # Иконка строки
            if status == "✅ Одобрено":
                row_icon = "✅"
            elif status == "❌ Отклонено":
                row_icon = "❌"
            elif blocked:
                row_icon = "🚨"
            else:
                row_icon = "📄"

            score_icon = "🟢" if score > 70 else ("🟡" if score >= 40 else "🔴")

            # Название для заголовка expander
            if blind_mode:
                display_name = f"ID: {anonymize_id(row['БИН'])}"
            else:
                display_name = row["Название КХ"]

            expander_label = (
                f"{row_icon}  {display_name}  ·  {row['Район']}  ·  "
                f"{score_icon} Score: {score}  ·  "
                f"Падёж: {mort:.1f}%  ·  "
                f"{amount:,.0f} ₸  ·  {status}"
            )

            # Подсветка нарушений
            if blocked:
                st.markdown(
                    f'<div class="blocked-banner">🚨 НАРУШЕНИЕ (Приказ №2): Падёж {mort:.1f}% — '
                    f'заявка {display_name} заблокирована</div>',
                    unsafe_allow_html=True,
                )

            with st.expander(expander_label, expanded=blocked and status == "⏳ На рассмотрении"):

                c1, c2, c3 = st.columns([1, 1.2, 1.5])

                # Данные хозяйства
                with c1:
                    st.markdown("**📋 Данные хозяйства**")
                    if blind_mode:
                        anon_id = anonymize_id(row["БИН"])
                        st.markdown(f'БИН: <span class="anon-id">{anon_id}</span>', unsafe_allow_html=True)
                        st.markdown(f'КХ: <span class="anon-id">{anonymize_id(row["Название КХ"])}</span>', unsafe_allow_html=True)
                    else:
                        st.write(f"🔢 БИН: `{row['БИН']}`")
                        st.write(f"🏢 КХ: **{row['Название КХ']}**")

                    st.write(f"📍 Район: {row['Район']}, {row['Регион']}")
                    st.write(f"🏷 Направление: **{direction}**")
                    st.write(f"🐄 Поголовье: **{row['Поголовье']}** гол.")
                    st.write(f"💀 Падёж: **{row['Падёж (гол)']}** гол. ({mort:.1f}%)")
                    st.write(f"💰 Запрошено: **{amount:,.0f} ₸**")

                # Анализ рисков
                with c2:
                    st.markdown("**🛡 Анализ рисков**")

                    if blocked:
                        st.markdown(f"""
                        <div style="background:#fde8e8;border-left:5px solid #dc2626;
                                    padding:12px 16px;border-radius:6px;color:#7f1d1d;">
                            <strong>🚨 БЛОКИРОВКА ВЫПЛАТЫ</strong><br>
                            Падёж {mort:.1f}% превышает норматив {MORTALITY_LIMIT}%.<br>
                            <em>Приказ №2 МСХ РК.</em>
                        </div>
                        """, unsafe_allow_html=True)
                    elif mort > 1.0:
                        st.markdown(f"""
                        <div style="background:#fffbeb;border-left:5px solid #f59e0b;
                                    padding:12px 16px;border-radius:6px;color:#78350f;">
                            <strong>⚠️ Повышенный падёж</strong><br>
                            Падёж {mort:.1f}% приближается к порогу {MORTALITY_LIMIT}%.
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div style="background:#d1fae5;border-left:5px solid #059669;
                                    padding:12px 16px;border-radius:6px;color:#064e3b;">
                            <strong>✅ Норма по гибели скота</strong><br>
                            Падёж {mort:.1f}% — в пределах норматива ≤ {MORTALITY_LIMIT}%.
                        </div>
                        """, unsafe_allow_html=True)

                    st.markdown("**📈 Прогноз эффективности**")
                    color  = "#059669" if score > 70 else ("#d97706" if score >= 40 else "#dc2626")
                    label  = "Высокий потенциал" if score > 70 else ("Средний" if score >= 40 else "Низкий потенциал")
                    advice = ("К приоритетному одобрению" if score > 70
                              else ("Требует стандартной проверки" if score >= 40
                                    else "Рекомендован отказ"))
                    st.markdown(f"""
                    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                                padding:12px 16px;margin-top:8px;">
                        <div style="display:flex;justify-content:space-between;align-items:center;">
                            <span style="font-weight:600;color:{color};">{label}</span>
                            <span style="font-size:22px;font-weight:700;color:{color};
                                         font-family:'JetBrains Mono',monospace;">{score}<span style="font-size:13px;">/100</span></span>
                        </div>
                        <div style="background:#e2e8f0;border-radius:999px;height:10px;margin:8px 0;overflow:hidden;">
                            <div style="width:{score}%;background:{color};height:100%;border-radius:999px;"></div>
                        </div>
                        <div style="font-size:12px;color:#64748b;">{advice}</div>
                    </div>
                    """, unsafe_allow_html=True)

                    # Влияние приоритетов
                    if p_milk != 50 or p_meat != 50 or p_breed != 50:
                        st.caption(f"⚙️ Приоритеты: Молоко={p_milk} / Мясо={p_meat} / Племенное={p_breed}")

                # Кнопки + объяснение
                with c3:
                    st.markdown("**🧠 Объяснение ИИ**")
                    hist_s = float(row.get("hist_score", 0.75))
                    clim_r = float(row.get("climate_risk", 0.45))
                    base_pts  = round(hist_s * 70)
                    clim_pen  = round(clim_r * 30)
                    breed_pts = 15 if int(row.get("is_breeding", 0)) else 0
                    sel_pts   = 10 if int(row.get("is_selection", 0)) else 0

                    # Приоритетная корректировка
                    corr = 0
                    if "Молоч" in direction: corr = round((p_milk - 50) * 0.3)
                    elif "Мясн" in direction: corr = round((p_meat - 50) * 0.3)
                    elif "Племен" in direction: corr = round((p_breed - 50) * 0.3)

                    st.markdown(f"""
                    <div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
                                padding:12px 16px;font-size:13px;color:#334155;line-height:1.7;">
                        <b>Итоговый балл: {score}/100</b><br>
                        • История района ({hist_s:.2f}): <b>+{base_pts} п.</b><br>
                        • Климатический риск ({clim_r:.2f}): <b>−{clim_pen} п.</b><br>
                        • Племенной статус: <b>+{breed_pts} п.</b><br>
                        • Селекционная программа: <b>+{sel_pts} п.</b><br>
                        {"• Корректировка приоритетов: <b>" + ('+' if corr >= 0 else '') + str(corr) + " п.</b><br>" if corr != 0 else ""}
                    </div>
                    """, unsafe_allow_html=True)

                    st.write("")

                    # КНОПКИ РЕШЕНИЯ
                    approve_disabled = (blocked or status == "✅ Одобрено")
                    reject_disabled  = (status == "❌ Отклонено")

                    b1, b2 = st.columns(2)
                    with b1:
                        if st.button("✅ Одобрить", key=f"app_{orig_idx}",
                                     disabled=approve_disabled,
                                     type="primary" if not approve_disabled else "secondary",
                                     use_container_width=True):
                            st.session_state.statuses[orig_idx] = "✅ Одобрено"
                            st.session_state.audit_log.append({
                                "Время":        datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                "Инспектор":    full_name,
                                "Логин":        st.session_state.current_user,
                                "Название КХ":  row["Название КХ"],
                                "БИН":          row["БИН"],
                                "Решение":      "✅ Одобрено",
                                "Сумма (₸)":    amount,
                                "Направление":  direction,
                                "Регион":       row["Регион"],
                            })
                            st.rerun()
                    with b2:
                        if st.button("❌ Отклонить", key=f"rej_{orig_idx}",
                                     disabled=reject_disabled,
                                     use_container_width=True):
                            st.session_state.statuses[orig_idx] = "❌ Отклонено"
                            st.session_state.audit_log.append({
                                "Время":        datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                "Инспектор":    full_name,
                                "Логин":        st.session_state.current_user,
                                "Название КХ":  row["Название КХ"],
                                "БИН":          row["БИН"],
                                "Решение":      "❌ Отклонено",
                                "Сумма (₸)":    amount,
                                "Направление":  direction,
                                "Регион":       row["Регион"],
                            })
                            st.rerun()

                    if blocked and status not in ["❌ Отклонено", "✅ Одобрено"]:
                        st.markdown(
                            '<div class="blocked-banner">🔒 Одобрение заблокировано — Приказ №2</div>',
                            unsafe_allow_html=True,
                        )
                    elif status == "✅ Одобрено":
                        st.success("✅ Субсидия одобрена")
                    elif status == "❌ Отклонено":
                        st.error("❌ Заявка отклонена")

        # Экспорт
        st.divider()
        export_df = df_sorted.copy()
        if blind_mode:
            export_df["БИН"]        = export_df["БИН"].apply(anonymize_id)
            export_df["Название КХ"] = export_df["Название КХ"].apply(anonymize_id)
        csv_data = export_df[
            ["БИН","Название КХ","Регион","Район","Направление",
             "Поголовье","Падёж %","Причит. сумма","FutureScore","_status"]
        ].rename(columns={"_status":"Статус"}).to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")

        st.download_button("📥 Выгрузить реестр (CSV)", data=csv_data,
                           file_name="reestr_msh_audit.csv", mime="text/csv")


    # ══════════════════════════════════════════════════════════════════════════════
    #  ТАБ 2 — АНАЛИТИКА
    # ══════════════════════════════════════════════════════════════════════════════
    with tab_analytics:
        st.subheader("📊 Аналитика реестра субсидий")

        df_a = st.session_state.df_raw.copy()
        df_a["FutureScore"] = compute_scores(df_a, p_milk, p_meat, p_breed)
        df_a["_status"]  = df_a.index.map(lambda i: st.session_state.statuses.get(i, "⏳ На рассмотрении"))
        df_a["_blocked"] = df_a["Падёж %"] > MORTALITY_LIMIT

        # Метрики
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Хозяйств всего",   len(df_a))
        a2.metric("Общий объём",      f"{df_a['Причит. сумма'].sum():,.0f} ₸")
        a3.metric("Средний Score",    f"{df_a['FutureScore'].mean():.1f}/100")
        a4.metric("Нарушений",        int(df_a["_blocked"].sum()))

        ch1, ch2 = st.columns(2)

        # График по районам
        with ch1:
            dist_sum = (
                df_a.groupby("Район")["Причит. сумма"].sum()
                .sort_values(ascending=True).tail(15)
            )
            fig_bar = go.Figure(go.Bar(
                x=dist_sum.values, y=dist_sum.index,
                orientation="h",
                marker=dict(color=dist_sum.values, colorscale="Blues", showscale=False),
                hovertemplate="%{y}: %{x:,.0f} ₸<extra></extra>",
            ))
            fig_bar.update_layout(
                title="Запрошено субсидий по районам (топ-15)",
                height=400, margin=dict(t=40, b=20, l=10, r=10),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_bar, use_container_width=True)

        # Доля по направлениям
        with ch2:
            dir_sum = df_a.groupby("Направление")["Причит. сумма"].sum().reset_index()
            fig_pie = go.Figure(go.Pie(
                labels=dir_sum["Направление"], values=dir_sum["Причит. сумма"],
                hole=0.42, textinfo="label+percent",
                hovertemplate="%{label}: %{value:,.0f} ₸<extra></extra>",
            ))
            fig_pie.update_layout(
                title="Распределение по направлениям хозяйства",
                height=400, margin=dict(t=40, b=20),
            )
            st.plotly_chart(fig_pie, use_container_width=True)

        ch3, ch4 = st.columns(2)

        # Гистограмма Score
        with ch3:
            fig_hist = go.Figure(go.Histogram(
                x=df_a["FutureScore"], nbinsx=20,
                marker_color="#2563a8", opacity=0.85,
            ))
            for x_val, color, label in [
                (40, "#ef4444", "Порог риска (40)"),
                (70, "#22c55e", "Высокая эффективность (70)"),
            ]:
                fig_hist.add_vline(x=x_val, line_dash="dash", line_color=color,
                                   annotation_text=label, annotation_position="top right",
                                   annotation_font_size=11)
            fig_hist.update_layout(
                title="Распределение FutureScore",
                xaxis_title="Балл", yaxis_title="Хозяйств",
                height=300, margin=dict(t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter: Score vs Сумма
        with ch4:
            colors_map = {True: "#dc2626", False: "#2563a8"}
            fig_sc = go.Figure()
            for blocked_val, grp in df_a.groupby("_blocked"):
                fig_sc.add_trace(go.Scatter(
                    x=grp["FutureScore"],
                    y=grp["Причит. сумма"] / 1_000_000,
                    mode="markers",
                    marker=dict(
                        color=colors_map[blocked_val],
                        size=7, opacity=0.7,
                    ),
                    name="Нарушение" if blocked_val else "Норма",
                    hovertemplate="Score: %{x}<br>Сумма: %{y:.2f} млн ₸<extra></extra>",
                ))
            fig_sc.update_layout(
                title="FutureScore vs Запрошенная сумма",
                xaxis_title="FutureScore", yaxis_title="Сумма (млн ₸)",
                height=300, margin=dict(t=40, b=20),
                plot_bgcolor="rgba(0,0,0,0)",
                legend=dict(x=0.01, y=0.99),
            )
            st.plotly_chart(fig_sc, use_container_width=True)


    # ══════════════════════════════════════════════════════════════════════════════
    #  ТАБ 3 — АВТО-РАСПРЕДЕЛЕНИЕ
    # ══════════════════════════════════════════════════════════════════════════════
    with tab_auto:
        st.subheader("⚡ Авто-распределение бюджета")
        st.markdown("""
        Алгоритм автоматически одобряет заявки **в порядке убывания FutureScore**,
        пропуская нарушителей (падёж > 2%), пока не будет исчерпан свободный остаток бюджета.
        """)

        df_auto = st.session_state.df_raw.copy()
        df_auto["FutureScore"] = compute_scores(df_auto, p_milk, p_meat, p_breed)
        df_auto["_blocked"]    = df_auto["Падёж %"] > MORTALITY_LIMIT
        df_auto["_status"]     = df_auto.index.map(
            lambda i: st.session_state.statuses.get(i, "⏳ На рассмотрении")
        )

        # Кандидаты: не рассмотрены, не заблокированы
        candidates = df_auto[
            (~df_auto["_blocked"]) &
            (df_auto["_status"] == "⏳ На рассмотрении")
        ].sort_values("FutureScore", ascending=False)

        current_balance = TOTAL_BUDGET - sum(
            df_auto.loc[i, "Причит. сумма"]
            for i, v in st.session_state.statuses.items()
            if v == "✅ Одобрено" and i in df_auto.index
        )

        # Предпросмотр плана
        plan_rows, plan_total = [], 0.0
        for idx, row in candidates.iterrows():
            amt = float(row["Причит. сумма"])
            if plan_total + amt <= current_balance:
                plan_rows.append({
                    "idx": idx,
                    "Название КХ": row["Название КХ"],
                    "Регион": row["Регион"],
                    "Направление": row["Направление"],
                    "Поголовье": row["Поголовье"],
                    "Причит. сумма": amt,
                    "FutureScore": int(row["FutureScore"]),
                })
                plan_total += amt

        p1, p2, p3 = st.columns(3)
        p1.metric("Свободный остаток", f"{current_balance:,.0f} ₸")
        p2.metric("Заявок в плане",    len(plan_rows))
        p3.metric("Сумма плана",       f"{plan_total:,.0f} ₸")

        if plan_rows:
            plan_df = pd.DataFrame(plan_rows).drop(columns=["idx"])
            st.dataframe(
                plan_df,
                column_config={
                    "Причит. сумма": st.column_config.NumberColumn("Сумма", format="%d ₸"),
                    "FutureScore": st.column_config.ProgressColumn(
                        "Score", format="%d", min_value=0, max_value=100),
                },
                hide_index=True, use_container_width=True, height=400,
            )

            col_btn1, col_btn2 = st.columns([1, 3])
            with col_btn1:
                if st.button("⚡ Одобрить весь план", type="primary", use_container_width=True):
                    for pr in plan_rows:
                        st.session_state.statuses[pr["idx"]] = "✅ Одобрено"
                    st.success(f"✅ Одобрено {len(plan_rows)} заявок на {plan_total:,.0f} ₸")
                    st.rerun()
            with col_btn2:
                st.info(f"ℹ️ После авто-распределения остаток бюджета составит "
                        f"**{current_balance - plan_total:,.0f} ₸**")
        else:
            st.warning("Нет подходящих кандидатов — все заявки уже рассмотрены или бюджет исчерпан.")

        # Нарушители — отдельный список
        blocked_list = df_auto[df_auto["_blocked"] & (df_auto["_status"] == "⏳ На рассмотрении")]
        if not blocked_list.empty:
            st.divider()
            st.markdown(f"### 🚨 Заблокированные заявки ({len(blocked_list)})")
            st.caption("Эти хозяйства не включены в план авто-распределения (нарушение Приказа №2)")
            st.dataframe(
                blocked_list[["Название КХ","Регион","Район","Падёж %","Причит. сумма","FutureScore"]],
                column_config={
                    "Причит. сумма": st.column_config.NumberColumn("Сумма", format="%d ₸"),
                    "Падёж %": st.column_config.NumberColumn("Падёж %", format="%.1f%%"),
                    "FutureScore": st.column_config.ProgressColumn(
                        "Score", format="%d", min_value=0, max_value=100),
                },
                hide_index=True, use_container_width=True,
            )


    # ══════════════════════════════════════════════════════════════════════════════
    #  ТАБ 4 — ЗАЯВКИ ФЕРМЕРОВ (из общего CSV)
    # ══════════════════════════════════════════════════════════════════════════════
    with tab_farmer:
        st.subheader("🌾 Заявки от фермеров")
        st.caption(
            "Реальные заявки, поданные через «Личный кабинет фермера». "
            "Данные синхронизируются через файл `data/applications.csv`."
        )

        col_ref, _ = st.columns([1, 4])
        with col_ref:
            if st.button("🔄 Обновить список", use_container_width=True):
                st.rerun()

        farmer_df = load_farmer_applications()

        if farmer_df.empty:
            st.info(
                "ℹ️ Заявок от фермеров пока нет. "
                "Попросите фермера подать заявку в «Личном кабинете фермера» — "
                "данные появятся здесь автоматически."
            )
        else:
            total_f    = len(farmer_df)
            pending_f  = int((farmer_df["status"] == "pending").sum())
            approved_f = int((farmer_df["status"] == "approved").sum())
            rejected_f = int((farmer_df["status"] == "rejected").sum())

            fm1, fm2, fm3, fm4 = st.columns(4)
            fm1.metric("Всего заявок",       total_f)
            fm2.metric("⏳ На рассмотрении",  pending_f)
            fm3.metric("✅ Одобрено",         approved_f)
            fm4.metric("❌ Отклонено",        rejected_f)

            st.divider()

            for _, frow in farmer_df.iterrows():
                bin_val      = str(frow.get("bin", ""))
                submitted_at = str(frow.get("submitted_at", ""))
                farm_name    = str(frow.get("farm_name", "—"))
                region       = str(frow.get("region", "—"))
                status_val   = str(frow.get("status", "pending"))

                def _safe_int(val, default=0):
                    try:
                        return int(float(str(val))) if str(val) not in ("", "nan", "None") else default
                    except (ValueError, TypeError):
                        return default

                def _safe_float(val, default=0.0):
                    try:
                        return float(str(val)) if str(val) not in ("", "nan", "None") else default
                    except (ValueError, TypeError):
                        return default

                score_val    = _safe_int(frow.get("score", 0))
                amount_val   = _safe_float(frow.get("requested_amount", 0))
                livestock_v  = _safe_int(frow.get("livestock", frow.get("cows_count", 0)))
                deaths_v     = _safe_int(frow.get("deaths", 0))
                hectares_v   = _safe_float(frow.get("hectares", 0.0))
                iin_v        = str(frow.get("iin", "—")) if str(frow.get("iin", "")) not in ("", "nan", "None") else "—"
                email_v      = str(frow.get("email", "—")) if str(frow.get("email", "")) not in ("", "nan", "None") else "—"
                phone_v      = str(frow.get("phone", "—")) if str(frow.get("phone", "")) not in ("", "nan", "None") else "—"

                status_map = {
                    "pending":  "⏳ На рассмотрении",
                    "approved": "✅ Одобрено",
                    "rejected": "❌ Отклонено",
                }
                status_label = status_map.get(status_val, status_val)
                score_icon   = "🟢" if score_val > 65 else ("🟡" if score_val >= 40 else "🔴")

                expander_label = (
                    f"{farm_name}  ·  БИН: {bin_val}  ·  {region}  ·  "
                    f"{score_icon} Балл: {score_val}  ·  {amount_val:,.0f} ₸  ·  {status_label}"
                )

                with st.expander(expander_label, expanded=(status_val == "pending")):
                    d1, d2, d3 = st.columns([1, 1, 1.2])

                    with d1:
                        st.markdown("**📋 Данные хозяйства**")
                        st.write(f"🏢 Хозяйство: **{farm_name}**")
                        st.write(f"🔢 БИН: `{bin_val}`")
                        st.write(f"🪪 ИИН: `{iin_v}`")
                        st.write(f"📍 Регион: {region}")
                        st.write(f"📧 Email: {email_v}")
                        st.write(f"📞 Телефон: {phone_v}")
                        st.write(f"🐄 Поголовье: **{livestock_v}** гол.")
                        st.write(f"💀 Падёж: **{deaths_v}** гол.")
                        if hectares_v > 0:
                            st.write(f"🌾 Площадь угодий: **{hectares_v:.1f}** га")
                        submitted_str = submitted_at[:16].replace("T", " ")
                        st.write(f"📅 Подано: {submitted_str}")

                    with d2:
                        st.markdown("**📊 Скоринговый балл**")
                        css_color = "#059669" if score_val > 65 else ("#d97706" if score_val >= 40 else "#dc2626")
                        score_label_text = (
                            "Высокий — рекомендуется одобрить" if score_val > 65
                            else ("Средний — требует проверки" if score_val >= 40
                                  else "Низкий — рекомендуется отклонить")
                        )
                        st.markdown(f"""
<div style="background:#f8fafc;border:1px solid #e2e8f0;border-radius:8px;
            padding:14px 18px;text-align:center;">
    <div style="font-size:36px;font-weight:700;color:{css_color};
                font-family:'JetBrains Mono',monospace;">{score_val}</div>
    <div style="background:#e2e8f0;border-radius:999px;height:8px;
                margin:8px 0;overflow:hidden;">
        <div style="width:{score_val}%;background:{css_color};
                    height:100%;border-radius:999px;"></div>
    </div>
    <div style="font-size:12px;color:#64748b;">{score_label_text}</div>
</div>
""", unsafe_allow_html=True)

                    with d3:
                        st.markdown("**✅ Решение аудитора**")
                        already_decided = status_val in ("approved", "rejected")

                        b_a, b_r = st.columns(2)
                        with b_a:
                            if st.button("✅ Одобрить",
                                         key=f"f_app_{bin_val}_{submitted_at}",
                                         disabled=already_decided,
                                         type="primary" if not already_decided else "secondary",
                                         use_container_width=True):
                                update_application_status_in_csv(
                                    bin_val, submitted_at, "approved", full_name
                                )
                                # Синхронизация db_apps фермера в той же сессии
                                for _a in st.session_state.get("db_apps", []):
                                    if str(_a.get("bin")) == bin_val and str(_a.get("submitted_at")) == submitted_at:
                                        _a["status"] = "approved"
                                        _a["reviewed_by"] = full_name
                                        _a["reviewed_at"] = __import__("datetime").datetime.now().isoformat()
                                st.session_state.audit_log.append({
                                    "Время":       datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                    "Инспектор":   full_name,
                                    "Логин":       st.session_state.current_user,
                                    "Название КХ": farm_name,
                                    "БИН":         bin_val,
                                    "Решение":     "✅ Одобрено (фермер)",
                                    "Сумма (₸)":   amount_val,
                                    "Направление": "Из кабинета фермера",
                                    "Регион":      region,
                                })
                                st.success("✅ Одобрено! Фермер увидит статус в своём кабинете.")
                                st.rerun()
                        with b_r:
                            if st.button("❌ Отклонить",
                                         key=f"f_rej_{bin_val}_{submitted_at}",
                                         disabled=already_decided,
                                         use_container_width=True):
                                update_application_status_in_csv(
                                    bin_val, submitted_at, "rejected", full_name
                                )
                                # Синхронизация db_apps фермера в той же сессии
                                for _a in st.session_state.get("db_apps", []):
                                    if str(_a.get("bin")) == bin_val and str(_a.get("submitted_at")) == submitted_at:
                                        _a["status"] = "rejected"
                                        _a["reviewed_by"] = full_name
                                        _a["reviewed_at"] = __import__("datetime").datetime.now().isoformat()
                                st.session_state.audit_log.append({
                                    "Время":       datetime.datetime.now().strftime("%d.%m.%Y %H:%M:%S"),
                                    "Инспектор":   full_name,
                                    "Логин":       st.session_state.current_user,
                                    "Название КХ": farm_name,
                                    "БИН":         bin_val,
                                    "Решение":     "❌ Отклонено (фермер)",
                                    "Сумма (₸)":   amount_val,
                                    "Направление": "Из кабинета фермера",
                                    "Регион":      region,
                                })
                                st.error("❌ Отклонено. Фермер увидит статус в своём кабинете.")
                                st.rerun()

                        if status_val == "approved":
                            st.success("✅ Заявка одобрена")
                            rb = str(frow.get("reviewed_by", ""))
                            ra = str(frow.get("reviewed_at", ""))[:16].replace("T", " ")
                            if rb and rb != "nan":
                                st.caption(f"Инспектор: {rb} · {ra}")
                        elif status_val == "rejected":
                            st.error("❌ Заявка отклонена")
                            rb = str(frow.get("reviewed_by", ""))
                            ra = str(frow.get("reviewed_at", ""))[:16].replace("T", " ")
                            if rb and rb != "nan":
                                st.caption(f"Инспектор: {rb} · {ra}")
                        else:
                            st.warning("⏳ Ожидает решения")

    # ══════════════════════════════════════════════════════════════════════════════
    #  ТАБ 5 — ЖУРНАЛ АУДИТА
    # ══════════════════════════════════════════════════════════════════════════════
    with tab_audit:
        st.subheader("📜 Журнал аудита решений")
        st.caption(
            "Автоматически фиксирует все действия инспекторов: "
            "одобрение и отклонение заявок с указанием времени, ФИО и суммы."
        )

        log = st.session_state.audit_log

        if not log:
            st.markdown("""
            <div style="background:#f8fafc;border:1px dashed #cbd5e1;border-radius:10px;
                        padding:40px;text-align:center;color:#94a3b8;">
                <div style="font-size:36px;margin-bottom:10px;">📋</div>
                <div style="font-size:15px;font-weight:500;color:#64748b;">Журнал пуст</div>
                <div style="font-size:13px;margin-top:4px;">
                    Действия появятся здесь после первого одобрения или отклонения заявки
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            log_df = pd.DataFrame(log[::-1])  # последние — вверху

            # Сводные метрики журнала
            jm1, jm2, jm3, jm4 = st.columns(4)
            approved_log = [r for r in log if r["Решение"] == "✅ Одобрено"]
            rejected_log = [r for r in log if r["Решение"] == "❌ Отклонено"]
            unique_inspectors = len({r["Логин"] for r in log})

            jm1.metric("Всего решений",  len(log))
            jm2.metric("✅ Одобрений",    len(approved_log),
                       delta=f"{sum(r['Сумма (₸)'] for r in approved_log):,.0f} ₸",
                       delta_color="off")
            jm3.metric("❌ Отклонений",   len(rejected_log))
            jm4.metric("Инспекторов",     unique_inspectors)

            st.divider()

            # Фильтры журнала
            jf1, jf2 = st.columns(2)
            with jf1:
                filter_decision = st.selectbox(
                    "Фильтр по решению",
                    ["Все решения", "✅ Одобрено", "❌ Отклонено"],
                    key="log_filter_decision",
                )
            with jf2:
                inspector_list = ["Все инспекторы"] + sorted({r["Инспектор"] for r in log})
                filter_inspector = st.selectbox(
                    "Фильтр по инспектору",
                    inspector_list,
                    key="log_filter_inspector",
                )

            filtered_log = log[::-1]
            if filter_decision != "Все решения":
                filtered_log = [r for r in filtered_log if r["Решение"] == filter_decision]
            if filter_inspector != "Все инспекторы":
                filtered_log = [r for r in filtered_log if r["Инспектор"] == filter_inspector]

            if not filtered_log:
                st.info("Нет записей по выбранным фильтрам.")
            else:
                display_df = pd.DataFrame(filtered_log)

                st.dataframe(
                    display_df,
                    column_config={
                        "Время":       st.column_config.TextColumn("🕐 Время", width="medium"),
                        "Инспектор":   st.column_config.TextColumn("👤 Инспектор", width="medium"),
                        "Логин":       st.column_config.TextColumn("Логин", width="small"),
                        "Название КХ": st.column_config.TextColumn("🏢 Название КХ", width="large"),
                        "БИН":         st.column_config.TextColumn("БИН", width="medium"),
                        "Решение":     st.column_config.TextColumn("✅ Решение", width="medium"),
                        "Сумма (₸)":   st.column_config.NumberColumn(
                            "💰 Сумма", format="%d ₸", width="medium"),
                        "Направление": st.column_config.TextColumn("Направление", width="medium"),
                        "Регион":      st.column_config.TextColumn("Регион", width="medium"),
                    },
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, 56 + len(filtered_log) * 35),
                )

                # Экспорт журнала
                st.divider()
                ec1, ec2 = st.columns([1, 3])
                with ec1:
                    log_csv = (
                        pd.DataFrame(filtered_log)
                        .to_csv(index=False, encoding="utf-8-sig")
                        .encode("utf-8-sig")
                    )
                    st.download_button(
                        "📥 Скачать журнал (CSV)",
                        data=log_csv,
                        file_name=f"audit_log_{datetime.date.today().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )
                with ec2:
                    st.caption(
                        f"Показано {len(filtered_log)} из {len(log)} записей · "
                        f"Последнее действие: {log[-1]['Время']}"
                    )
