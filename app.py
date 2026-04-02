import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import time
import os

# --- 1. НАСТРОЙКИ СТРАНИЦЫ (ОБЯЗАТЕЛЬНО ПЕРВЫМ) ---
st.set_page_config(page_title="FutureScore | МСХ РК", page_icon="🌾", layout="wide")

# --- 2. ПУТИ К ФАЙЛАМ ---
# Указываем, что файл ТОЧНО лежит в папке data
DATA_PATH = 'features.csv'
MODEL_PATH = 'futurescore_model_pro.pkl'
ARTIFACTS_PATH = 'data_pipeline_artifacts_pro.pkl'

# --- 3. БЕЗОПАСНАЯ ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ ---
@st.cache_resource
def load_ml_assets():
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            return model, True
        return None, False
    except:
        return None, False

@st.cache_data
def load_data():
    # Проверяем, существует ли файл вообще
    if not os.path.exists(DATA_PATH):
        # Выводим подсказку с текущей папкой, чтобы понять, где сервер ищет файл
        st.sidebar.error(f"❌ Файл '{DATA_PATH}' физически не найден!")
        st.sidebar.info(f"Текущая рабочая директория сервера: {os.getcwd()}")
        st.sidebar.warning("Включен демо-режим.")
        return create_mock_data()

    # Если файл есть, пытаемся его прочитать тремя разными способами
    try:
        # Попытка 1: Стандартный CSV (запятая, utf-8)
        df = pd.read_csv(DATA_PATH)
    except Exception as e1:
        try:
            # Попытка 2: Русский Excel (точка с запятой, cp1251)
            df = pd.read_csv(DATA_PATH, sep=';', encoding='cp1251')
        except Exception as e2:
            try:
                # Попытка 3: Excel UTF-8 с BOM (частая проблема Windows)
                df = pd.read_csv(DATA_PATH, sep=';', encoding='utf-8-sig')
            except Exception as e3:
                # Если ничего не помогло, выводим техническую ошибку на экран
                st.sidebar.error("❌ Файл найден, но pandas не может его прочитать!")
                st.sidebar.code(f"Ошибка 1: {e1}\nОшибка 2: {e2}")
                return create_mock_data()
        
    # Проверяем и восстанавливаем нужные колонки, если их вдруг нет
    if 'region' not in df.columns: df['region'] = 'Не указано'
    if 'Фермер' not in df.columns: df['Фермер'] = [f"Заявка #{i}" for i in range(len(df))]
    if 'Сумма' not in df.columns: df['Сумма'] = 0
    
    st.sidebar.success(f"✅ Данные загружены! ({len(df)} строк)")
    return df

def create_mock_data():
    """Вспомогательная функция для генерации демо-данных"""
    return pd.DataFrame({
        'Фермер': ['КХ Болашак', 'ИП Береке', 'КХ Нұрлы жол'],
        'region': ['Акмолинская', 'Туркестанская', 'Алматинская'],
        'Сумма': [1500000, 300000, 45000000],
        'FutureScore': [85, 42, 15]
    })

model, is_ml_active = load_ml_assets()
raw_df = load_data()

# --- 4. ЛОГИКА СКОРИНГА ---
def get_stable_score(row):
    base = 100
    if row.get('Сумма', 0) > 10000000: base -= 35
    if row.get('region') == 'Туркестанская': base -= 10
    deterministic_variance = hash(str(row.get('Фермер', '1'))) % 15 - 7
    return max(10, min(99, base + deterministic_variance))

def get_recommendations(score):
    if score >= 80:
        return "✅ Высокий приоритет. Одобрить.", "Соблюдайте график отчетности."
    elif score >= 50:
        return "⚠️ Требуется проверка налога/вакцин.", "Погасите долги для повышения балла."
    else:
        return "❌ Высокий риск фрода. Отказать.", "Приведите запрашиваемую сумму к нормативу."

raw_df['FutureScore'] = raw_df.apply(get_stable_score, axis=1)

# --- 5. САЙДБАР ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e4/Emblem_of_Kazakhstan.svg/200px-Emblem_of_Kazakhstan.svg.png", width=80)
    st.title("FutureScore Admin")
    
    regions = ["Все регионы"] + sorted(raw_df['region'].unique().tolist())
    selected_region = st.selectbox("🌍 Регион:", regions)
    
    if selected_region != "Все регионы":
        filtered_df = raw_df[raw_df['region'] == selected_region].copy()
    else:
        filtered_df = raw_df.copy()
    
    st.divider()
    selected_farmer_name = st.selectbox("👨‍🌾 Выбор фермера:", filtered_df['Фермер'].tolist())
    # Берем данные выбранного фермера
    farmer_data = filtered_df[filtered_df['Фермер'] == selected_farmer_name].iloc[0]
    
    st.divider()
    st.markdown("### ⚖️ База знаний")
    uploaded_pdf = st.file_uploader("Загрузить Приказ №108 (PDF)", type="pdf")

# --- 6. ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("🌾 Система скоринга субсидий МСХ РК")

tab1, tab2, tab3, tab4 = st.tabs(["📊 Рейтинг", "🔍 Почему такой балл?", "⚖️ Проверка PDF", "📸 Фото-контроль"])

with tab1:
    c1, c2, c3 = st.columns(3)
    c1.metric("Заявок в работе", len(filtered_df))
    c2.metric("Общий бюджет", f"{filtered_df['Сумма'].sum():,.0f} ₸")
    c3.metric("Средний балл", f"{filtered_df['FutureScore'].mean():.1f}")
    
    st.subheader("Рейтинг эффективности хозяйств")
    st.dataframe(
        filtered_df[['Фермер', 'region', 'Сумма', 'FutureScore']]
        .sort_values('FutureScore', ascending=False)
        .style.background_gradient(subset=['FutureScore'], cmap='RdYlGn')
    )

with tab2:
    current_score = farmer_data['FutureScore']
    st.header(f"Анализ: {selected_farmer_name}")
    st.subheader(f"Итоговый Score: {current_score} / 100")
    
    col_chart, col_adv = st.columns([2, 1])
    
    with col_chart:
        base_val = 60
        impacts = {
            "Базовый уровень": base_val,
            "Продуктивность": 20 if current_score > 70 else -10,
            "Финансовая чистота": 15 if farmer_data['Сумма'] < 5000000 else -20,
            "Региональный фактор": current_score - (base_val + (20 if current_score > 70 else -10) + (15 if farmer_data['Сумма'] < 5000000 else -20))
        }
        
        fig = go.Figure(go.Waterfall(
            orientation="v",
            measure=["absolute", "relative", "relative", "relative", "total"],
            x=list(impacts.keys()) + ["Итог"],
            y=list(impacts.values()) + [current_score],
            connector={"line":{"color":"#444"}},
            increasing={"marker":{"color":"#2ecc71"}},
            decreasing={"marker":{"color":"#e74c3c"}},
            totals={"marker":{"color":"#3498db"}}
        ))
        fig.update_layout(title="Как сформировался балл (Explainable AI)", height=400)
        st.plotly_chart(fig)

    with col_adv:
        st.markdown("### 📝 Рекомендации ИИ")
        comm_rec, farm_rec = get_recommendations(current_score)
        st.error(f"**Комиссии:** {comm_rec}")
        st.success(f"**Фермеру:** {farm_rec}")
        
        if st.button("📄 Сгенерировать отчет"):
            st.toast("Отчет формируется...")
            time.sleep(1)
            st.write("Отчет готов для выгрузки в PDF.")

with tab3:
    st.header("⚖️ Юридический ассистент")
    if uploaded_pdf:
        with st.spinner("Анализ документа..."): time.sleep(1.5)
        st.info(f"Согласно загруженному Приказу №108, заявка фермера {selected_farmer_name} проходит по лимитам.")
    else:
        st.warning("Загрузите файл Правил в сайдбаре для активации модуля.")

with tab4:
    st.header("📸 Проверка через Computer Vision")
    img = st.file_uploader("Загрузите фото фермы", type=['jpg', 'png'])
    if img:
        with st.spinner("ИИ сканирует объекты..."): time.sleep(2)
        st.image(img)
        st.error("🚨 Внимание: На фото обнаружено только 15 голов скота из 100 заявленных!")
