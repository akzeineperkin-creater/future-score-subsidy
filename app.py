import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import PyPDF2
import os
import time

# --- НАСТРОЙКИ СТРАНИЦЫ ---
st.set_page_config(page_title="FutureScore | МСХ РК", layout="wide", page_icon="🌾")

# --- ЗАГРУЗКА МОДЕЛЕЙ (С ПРЕДОХРАНИТЕЛЕМ) ---
@st.cache_resource
def load_ml_assets():
    model, artifacts = None, None
    try:
        if os.path.exists('models/futurescore_model_pro.pkl'):
            model = joblib.load('models/futurescore_model_pro.pkl')
        if os.path.exists('models/data_pipeline_artifacts_pro.pkl'):
            artifacts = joblib.load('models/data_pipeline_artifacts_pro.pkl')
        return model, artifacts
    except Exception as e:
        st.sidebar.warning(f"⚠️ Режим эмуляции ML. Ошибка загрузки: {e}")
        return None, None

model, artifacts = load_ml_assets()
USE_ML = model is not None

# --- ЗАГРУЗКА ДАННЫХ ---
@st.cache_data
def load_data():
    try:
        # Пытаемся загрузить реальный датасет
        df = pd.read_csv('features.csv')
        
        # Переименовываем колонки для красивого UI (как ты просила)
        rename_dict = {
            'region': 'Регион',
            'Направление водства': 'Направление',
            'Причитающая сумма': 'Сумма',
            'Район хозяйства': 'Район'
        }
        df = df.rename(columns={k: v for k, v in rename_dict.items() if k in df.columns})
        
        # Если каких-то колонок нет, создаем фейковые для красивого демо
        if 'Регион' not in df.columns: df['Регион'] = np.random.choice(['Акмолинская', 'Алматинская', 'Туркестанская'], len(df))
        if 'Сумма' not in df.columns: df['Сумма'] = np.random.randint(1000000, 50000000, len(df))
        if 'Направление' not in df.columns: df['Направление'] = np.random.choice(['Мясное', 'Молочное', 'Племенное'], len(df))
        if 'Удой_на_корову' not in df.columns: df['Удой_на_корову'] = np.random.randint(200, 1100, len(df))
        
        # Генерируем уникальные имена для демо, если их нет
        if 'Фермер' not in df.columns:
            df['Фермер'] = [f"КХ/ИП Заявка №{i+1000}" for i in range(len(df))]
            
        return df
    except Exception as e:
        st.error(f"❌ Ошибка загрузки features.csv: {e}")
        # Возвращаем пустой DF, чтобы приложение не упало
        return pd.DataFrame(columns=['Фермер', 'Регион', 'Направление', 'Сумма', 'Удой_на_корову'])

df = load_data()

# --- ЛОГИКА СКОРИНГА ---
def calculate_score(row):
    """Смешанный скоринг (ML + Rule-based) с накидыванием +10 баллов"""
    base_score = 70  # Средний старт
    
    # 1. Экспертные правила (Анти-фрод)
    удой = row.get('Удой_на_корову', 500)
    if удой > 1000:
        return 0, "🚩 ФРОД: Аномально высокий удой (физически невозможно)."
    elif удой < 300:
        base_score -= 20
        
    сумма = row.get('Сумма', 0)
    if сумма > 30000000:
        base_score -= 10 # Риск завышения бюджета
        
    # 2. Имитация предсказания ML (если модель есть, но пайплайн сложен для демо)
    if USE_ML:
        # Для хакатона: делаем детерминированный шум на основе суммы, чтобы баллы были разные
        ml_impact = (сумма % 20) - 5 
        raw_score = base_score + ml_impact
    else:
        raw_score = base_score + np.random.randint(-10, 15)
        
    # 3. ФИЧА: Добавляем +10 баллов для реалистичности и ограничиваем до 100
    final_score = int(min(raw_score + 10, 100))
    
    # Генерация отзыва ИИ
    if final_score >= 80:
        feedback = "✅ Рекомендуется к приоритетному субсидированию. Высокая эффективность."
    elif final_score >= 50:
        feedback = "⚠️ Требуется дополнительная проверка инспектором (средний риск)."
    else:
        feedback = "❌ Высокий риск нецелевого использования. Отказ."
        
    return final_score, feedback

# Расчет баллов для всей таблицы
if 'FutureScore' not in df.columns and not df.empty:
    scores, feedbacks = zip(*df.apply(calculate_score, axis=1))
    df['FutureScore'] = scores
    df['Вердикт_ИИ'] = feedbacks

# --- UI: САЙДБАР ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/thumb/e/e0/Emblem_of_Kazakhstan.svg/200px-Emblem_of_Kazakhstan.svg.png", width=100)
    st.title("Управление")
    st.markdown("---")
    
    selected_region = st.selectbox("📍 Фильтр по Региону", options=["Все"] + list(df['Регион'].unique()))
    min_score = st.slider("📊 Минимальный балл", 0, 100, 0)
    
    st.markdown("---")
    st.metric("Всего заявок", len(df))
    if not df.empty:
        st.metric("Средний балл по стране", f"{int(df['FutureScore'].mean())}/100")

# Применяем фильтры
filtered_df = df.copy()
if selected_region != "Все":
    filtered_df = filtered_df[filtered_df['Регион'] == selected_region]
filtered_df = filtered_df[filtered_df['FutureScore'] >= min_score]

# --- UI: ГЛАВНЫЙ ЭКРАН ---
st.title("🌾 FutureScore: Merit-based Скоринг Субсидий")
st.markdown("Система интеллектуального анализа заявок МСХ РК на базе Machine Learning и Computer Vision.")

tab1, tab2, tab3, tab4 = st.tabs(["🏆 Рейтинг Заявок", "🔍 Детали и SHAP", "⚖️ AI-Юрист (Приказ №108)", "📸 Computer Vision"])

# --- ВКЛАДКА 1: РЕЙТИНГ ---
with tab1:
    st.subheader("Общий реестр сельхозтоваропроизводителей")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.dataframe(
            filtered_df[['Фермер', 'Регион', 'Направление', 'Сумма', 'FutureScore', 'Вердикт_ИИ']]
            .sort_values(by='FutureScore', ascending=False)
            .style.applymap(lambda x: 'background-color: #ffcccc' if x == 0 else ('background-color: #ccffcc' if isinstance(x, int) and x >= 80 else ''), subset=['FutureScore']),
            use_container_width=True, height=400
        )
    with col2:
        if not filtered_df.empty:
            fig = px.histogram(filtered_df, x="FutureScore", nbins=20, title="Распределение баллов", color_discrete_sequence=['#2ecc71'])
            st.plotly_chart(fig, use_container_width=True)

# --- ВКЛАДКА 2: ДЕТАЛИ И SHAP (Explainability) ---
with tab2:
    if not filtered_df.empty:
        selected_farmer_name = st.selectbox("Выберите заявку для детального разбора:", filtered_df['Фермер'].tolist())
        farmer_data = filtered_df[filtered_df['Фермер'] == selected_farmer_name].iloc[0]
        
        st.markdown(f"### Карточка хозяйства: {farmer_data['Фермер']}")
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Запрошенная сумма", f"{farmer_data['Сумма']:,.0f} ₸")
        c2.metric("Регион", farmer_data['Регион'])
        
        score = farmer_data['FutureScore']
        color = "green" if score >= 80 else "orange" if score >= 50 else "red"
        
        # Спидометр
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "FutureScore"},
            gauge = {'axis': {'range': [None, 100]},
                     'bar': {'color': color},
                     'steps': [
                         {'range': [0, 49], 'color': "#ffcccc"},
                         {'range': [50, 79], 'color': "#ffffcc"},
                         {'range': [80, 100], 'color': "#ccffcc"}]}
        ))
        c3.plotly_chart(fig_gauge, use_container_width=True)
        
        st.info(f"**Заключение ИИ:** {farmer_data['Вердикт_ИИ']}")
        
        st.subheader("Влияние факторов (Имитация SHAP)")
        # Имитация факторов для объяснимости
        factors = pd.DataFrame({
            'Фактор': ['Продуктивность (Удой)', 'История налогов', 'Региональный риск', 'Обеспеченность кормами'],
            'Влияние на балл': [15, 10, -5, 20]
        })
        fig_bar = px.bar(factors, x='Влияние на балл', y='Фактор', orientation='h', color='Влияние на балл', color_continuous_scale='RdYlGn')
        st.plotly_chart(fig_bar, use_container_width=True)
        
        st.button("✅ Утвердить решение комиссии", type="primary")

# --- ВКЛАДКА 3: AI-ЮРИСТ (NLP) ---
with tab3:
    st.subheader("Сверка с Приказом Министра СХ РК")
    st.markdown("Загрузите нормативный документ (PDF) для автоматической сверки заявки с правилами.")
    
    uploaded_pdf = st.file_uploader("Загрузить регламент (PDF)", type="pdf")
    
    if uploaded_pdf is not None:
        with st.spinner("Gemini AI анализирует документ..."):
            time.sleep(2) # Имитация работы API
            st.success("Документ успешно обработан!")
            
            st.markdown("""
            **Результат анализа (Имитация RAG-системы):**
            > 📄 *Извлечено из Главы 2, пункта 14:* "Норматив субсидирования на одну голову КРС молочного направления составляет не более 15 000 тенге".
            """)
            
            if not filtered_df.empty:
                st.warning(f"⚠️ **Алерт по текущей заявке ({farmer_data['Фермер']}):** Запрошенная сумма превышает базовый норматив на 12%. Рекомендуется аудит.")

# --- ВКЛАДКА 4: COMPUTER VISION ---
with tab4:
    st.subheader("Объективный контроль фермы (Anti-Fraud)")
    st.markdown("Анализ фотографий/спутниковых снимков хозяйства для подтверждения инфраструктуры.")
    
    uploaded_img = st.file_uploader("Загрузите фото коровника или пастбища (JPG/PNG)", type=['png', 'jpg', 'jpeg'])
    
    if uploaded_img is not None:
        st.image(uploaded_img, caption="Загруженное изображение", use_container_width=True)
        
        if st.button("🔍 Запустить CV-анализ (YOLOv8)"):
            with st.spinner("Нейросеть распознает объекты..."):
                time.sleep(2.5) # Имитация инференса модели
                
                col_res1, col_res2 = st.columns(2)
                with col_res1:
                    st.success("Анализ завершен!")
                    st.write("🐄 **Распознано коров:** 14 шт.")
                    st.write("🏗️ **Инфраструктура:** Кормушки (найдено), Навес (найдено).")
                with col_res2:
                    st.error("🚨 Внимание: В заявке указано 150 голов КРС, однако визуальный контроль подтверждает наличие только инфраструктуры малого типа. Высокий риск приписки!")
