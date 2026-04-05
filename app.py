# =============================================================================
#  FUTURE-SCORE: ПЛАТФОРМА УМНЫХ СУБСИДИЙ
#  Единый файл приложения (Frontend + ML + Backend)
# =============================================================================

import os
import datetime
import pandas as pd
import streamlit as st
import joblib

# ─────────────────────────────────────────────────────────────────────────────
#  НАСТРОЙКА СТРАНИЦЫ
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FutureScore | АгроСубсидии РК", page_icon="🌾", layout="wide")

# ─────────────────────────────────────────────────────────────────────────────
#  БАЗА ДАННЫХ (CSV)
# ─────────────────────────────────────────────────────────────────────────────
CSV_PATH = "applications.csv"
CSV_COLUMNS = [
    "farm_name", "bin", "region", "direction", "cows_count", "pasture_area", 
    "mortality_rate", "requested_amount", "normative", "score", "status", "alerts"
]

def init_database():
    if not os.path.exists(CSV_PATH):
        pd.DataFrame(columns=CSV_COLUMNS).to_csv(CSV_PATH, index=False)

init_database()

# ─────────────────────────────────────────────────────────────────────────────
#  ML ДВИЖОК (Твоя ИМБА)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_ml_engine():
    m_file = 'futurescore_model_pro.pkl'
    a_file = 'data_pipeline_artifacts_pro.pkl'
    if os.path.exists(m_file) and os.path.exists(a_file):
        try:
            model = joblib.load(m_file)
            arts = joblib.load(a_file)
            return model, arts, True
        except:
            return None, None, False
    return None, None, False

MODEL, ARTIFACTS, READY = init_ml_engine()

class SmartScoring:
    LIMITS = {
        "mortality": {"default": 0.03, "Субсидирование в скотоводстве": 0.03},
        "pasture": {"Акмолинская область": 8.5, "Туркестанская область": 11.0, "default": 10.0}
    }

    @staticmethod
    def calculate(data, levers):
        if not READY:
            return 50, ["⚠️ Демо-режим (Модели не найдены)"]

        cols = ['region', 'Направление водства', 'Норматив', 'Причитающая сумма', 
                'cows_count', 'pasture_area', 'mortality_rate']
        
        inp = data.copy()
        for c, le in ARTIFACTS.items():
            if c in inp:
                val = str(inp[c])
                inp[c] = le.transform([val])[0] if val in le.classes_ else 0
        
        df_inp = pd.DataFrame([inp])[cols]
        base_score = int(MODEL.predict_proba(df_inp)[0][1] * 100)

        alerts = []
        penalty = 0
        
        # Проверка пастбищ (Приказ №3)
        reg = data.get('region', 'default')
        p_norm = SmartScoring.LIMITS["pasture"].get(reg, SmartScoring.LIMITS["pasture"]["default"])
        current_load = data['pasture_area'] / max(1, data['cows_count'])
        if current_load < p_norm:
            penalty += 25
            alerts.append(f"🚩 Перевыпас (Приказ №3). Норма: {p_norm} га/гол, Факт: {current_load:.1f}")

        # Проверка падежа (Приказ №2)
        dir_name = data.get('Направление водства', 'default')
        m_norm = SmartScoring.LIMITS["mortality"].get(dir_name, SmartScoring.LIMITS["mortality"]["default"])
        if data['mortality_rate'] > m_norm:
            penalty += 30
            alerts.append(f"🚩 Падеж выше нормы (Приказ №2). Лимит: {m_norm*100}%, Факт: {data['mortality_rate']*100:.1f}%")

        mult = levers.get(dir_name, 1.0)
        final = int((base_score * mult) - penalty)
        return max(0, min(100, final)), alerts

# ─────────────────────────────────────────────────────────────────────────────
#  МАРШРУТИЗАЦИЯ И ИНТЕРФЕЙСЫ
# ─────────────────────────────────────────────────────────────────────────────
if "role" not in st.session_state:
    st.session_state.role = None

def show_role_selector():
    st.markdown("<h1 style='text-align: center;'>🌾 Платформа FutureScore</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Выберите вашу роль для входа в систему</p>", unsafe_allow_html=True)
    st.write("---")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("👨‍🌾 Я — Фермер", use_container_width=True, height=60):
            st.session_state.role = "farmer"
            st.rerun()
    with col2:
        if st.button("🏛 Я — Аудитор (МСХ)", use_container_width=True, height=60):
            st.session_state.role = "auditor_login"
            st.rerun()

def show_farmer_cabinet():
    st.button("⬅ Назад", on_click=lambda: st.session_state.update(role=None))
    st.title("📝 Подача заявки на субсидию")
    
    with st.form("farmer_form"):
        st.subheader("1. Основные данные")
        farm_name = st.text_input("Название хозяйства (КХ/ТОО)")
        bin_val = st.text_input("БИН/ИИН")
        region = st.selectbox("Регион", ["Акмолинская область", "Туркестанская область", "Алматинская область"])
        direction = st.selectbox("Направление субсидирования", ["Субсидирование в скотоводстве", "Субсидирование в овцеводстве"])
        
        st.subheader("2. Производственные показатели")
        c1, c2 = st.columns(2)
        with c1:
            cows_count = st.number_input("Поголовье скота (голов)", min_value=10, value=100)
            pasture_area = st.number_input("Площадь пастбищ (га)", min_value=0.0, value=1000.0)
        with c2:
            mortality_pct = st.number_input("Процент падежа за год (%)", min_value=0.0, max_value=100.0, value=1.0)
            requested_amount = st.number_input("Запрашиваемая сумма (₸)", min_value=100000.0, value=1500000.0)
        
        submitted = st.form_submit_button("Проверить заявку ИИ-скорингом", type="primary")

    if submitted:
        # Подготовка данных для ML
        farmer_data = {
            'region': region,
            'Направление водства': direction,
            'Норматив': 15000.0,  # Заглушка норматива
            'Причитающая сумма': requested_amount,
            'cows_count': cows_count,
            'pasture_area': pasture_area,
            'mortality_rate': mortality_pct / 100.0
        }
        
        # Расчет
        score, alerts = SmartScoring.calculate(farmer_data, levers={})
        
        # Вывод результатов
        st.divider()
        st.header(f"Ваш FutureScore: {score} / 100")
        
        if score >= 70:
            st.success("✅ Предварительно одобрено! Ваши показатели соответствуют нормам.")
            status = "Одобрено"
        else:
            st.error("❌ Выявлены риски. Заявка может быть отклонена.")
            status = "В зоне риска"
            for alert in alerts:
                st.warning(alert)
                
            # Симулятор What-If
            st.markdown("### 🚀 Инвестиционный симулятор")
            st.info("Посмотрите, как исправить ситуацию:")
            sim_pasture = st.slider("Добавить площадь пастбищ (га)", min_value=int(pasture_area), max_value=int(pasture_area)+5000, value=int(pasture_area))
            sim_data = farmer_data.copy()
            sim_data['pasture_area'] = sim_pasture
            sim_score, _ = SmartScoring.calculate(sim_data, levers={})
            st.metric("Балл при новых площадях", f"{sim_score} / 100", delta=sim_score - score)

        # Сохранение в базу
        new_row = pd.DataFrame([{
            "farm_name": farm_name, "bin": bin_val, "region": region, "direction": direction,
            "cows_count": cows_count, "pasture_area": pasture_area, "mortality_rate": mortality_pct/100.0,
            "requested_amount": requested_amount, "normative": 15000.0, "score": score, 
            "status": status, "alerts": " | ".join(alerts)
        }])
        df = pd.read_csv(CSV_PATH)
        df = pd.concat([df, new_row], ignore_index=True)
        df.to_csv(CSV_PATH, index=False)
        st.toast("Заявка сохранена в реестр!")

def show_auditor_login():
    st.button("⬅ Назад", on_click=lambda: st.session_state.update(role=None))
    st.subheader("🔐 Вход для Аудитора")
    pwd = st.text_input("Пароль", type="password")
    if st.button("Войти"):
        if pwd == "123": # Простой пароль для хакатона
            st.session_state.role = "auditor_dashboard"
            st.rerun()
        else:
            st.error("Неверный пароль")

def show_auditor_dashboard():
    st.sidebar.button("⬅ Выйти", on_click=lambda: st.session_state.update(role=None))
    
    # РЫЧАГИ МИНИСТРА
    st.sidebar.markdown("### 🎛️ Государственные приоритеты")
    weight_meat = st.sidebar.slider("Приоритет: Скотоводство", 0.5, 1.5, 1.0, 0.1)
    weight_sheep = st.sidebar.slider("Приоритет: Овцеводство", 0.5, 1.5, 1.0, 0.1)
    current_levers = {
        "Субсидирование в скотоводстве": weight_meat,
        "Субсидирование в овцеводстве": weight_sheep
    }

    st.title("🛡️ Кабинет Цифрового Аудитора")
    st.caption("Режим Слепого Одобрения (Blind Review) активирован")
    
    df = pd.read_csv(CSV_PATH)
    if df.empty:
        st.info("Пока нет новых заявок.")
        return

    # Динамический пересчет баллов
    st.write("### 📋 Реестр заявок")
    display_data = []
    
    for _, row in df.iterrows():
        # Собираем словарь для движка
        r_data = {
            'region': row['region'], 'Направление водства': row['direction'],
            'Норматив': row['normative'], 'Причитающая сумма': row['requested_amount'],
            'cows_count': row['cows_count'], 'pasture_area': row['pasture_area'],
            'mortality_rate': row['mortality_rate']
        }
        
        # ПЕРЕСЧЕТ С УЧЕТОМ РЫЧАГОВ ИЗ САЙДБАРА
        dyn_score, dyn_alerts = SmartScoring.calculate(r_data, current_levers)
        
        display_data.append({
            "ID заявки": f"REQ-{hash(row['bin']) % 10000}", # Слепое одобрение (Скрываем имя)
            "Регион": row['region'],
            "Направление": row['direction'],
            "Сумма (₸)": row['requested_amount'],
            "FutureScore": dyn_score,
            "Риски (NLP)": "🔴 " + dyn_alerts[0] if dyn_alerts else "🟢 Ок"
        })

    # Выводим таблицу
    st.dataframe(pd.DataFrame(display_data), use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
#  РОУТЕР
# ─────────────────────────────────────────────────────────────────────────────
if st.session_state.role is None:
    show_role_selector()
elif st.session_state.role == "farmer":
    show_farmer_cabinet()
elif st.session_state.role == "auditor_login":
    show_auditor_login()
elif st.session_state.role == "auditor_dashboard":
    show_auditor_dashboard()
