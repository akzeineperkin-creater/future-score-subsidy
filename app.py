import streamlit as st
import pandas as pd
import numpy as np
import pickle
import shap
import matplotlib.pyplot as plt
import os

# --- 1. Настройка страницы ---
st.set_page_config(page_title="FutureScore PRO | AI GovTech", layout="wide", page_icon="🚜")
st.title("🚜 FutureScore PRO: Merit-based скоринг субсидий")

# --- 2. Умная загрузка моделей (Ищет и в корне, и в папке models) ---
@st.cache_resource
def load_ai_brains():
    # Проверяем, где лежат файлы (вы же переносили их из папки models)
    arts_path = 'data_pipeline_artifacts_pro.pkl' if os.path.exists('data_pipeline_artifacts_pro.pkl') else 'models/data_pipeline_artifacts_pro.pkl'
    model_path = 'futurescore_model_pro.pkl' if os.path.exists('futurescore_model_pro.pkl') else 'models/futurescore_model_pro.pkl'
    
    with open(arts_path, 'rb') as f:
        data_arts = pickle.load(f)
    with open(model_path, 'rb') as f:
        model_arts = pickle.load(f)
    return data_arts, model_arts

try:
    data_arts, model_arts = load_ai_brains()
    model = model_arts['xgboost_model']
    explainer = model_arts['explainer']
    features_list = data_arts['features_list'] # Те самые 9 признаков
    st.sidebar.success("✅ ИИ-модель активна")
except Exception as e:
    st.error(f"❌ Ошибка загрузки модели. Убедитесь, что файлы .pkl залиты полностью. \n Детали: {e}")
    st.stop()

# --- 3. Автозагрузка данных (Zero-Click) ---
@st.cache_data
def load_data():
    # Ищем файл с данными
    data_path = 'final_dataset_pro.csv' if os.path.exists('final_dataset_pro.csv') else 'data/final_dataset_pro.csv'
    if os.path.exists(data_path):
        return pd.read_csv(data_path)
    return None

df_raw = load_data()

if df_raw is None:
    st.warning("⚠️ Файл `final_dataset_pro.csv` не найден в репозитории. Пожалуйста, убедитесь, что он загружен на GitHub.")
    st.stop()

# --- 4. Расчет рейтинга ---
# Берем только те колонки, которые знает модель
X = df_raw[features_list].copy()

# Предикт
probs = model.predict_proba(X)[:, 1]
df_raw['FutureScore'] = np.round(probs * 100, 1)

# Вывод шорт-листа
st.header("🏆 Шорт-лист кандидатов (Merit-based)")
st.info("Рейтинг рассчитан автоматически на основе исторических данных и климатических рисков региона.")

# Сортируем топ
top_farmers = df_raw.sort_values(by='FutureScore', ascending=False).reset_index(drop=True)
display_cols = ['FutureScore', 'climate_risk', 'amount_to_norm_ratio', 'is_breeding']
# Если есть еще понятные колонки, добавляем их
if 'month' in top_farmers.columns: display_cols.append('month')

st.dataframe(top_farmers[display_cols].head(50), use_container_width=True)

# --- 5. Ультимативный GovTech Advisor ---
st.divider()
st.header("🧠 GovTech Advisor (ИИ-Ассистент Комиссии)")
st.markdown("Выберите заявку из таблицы выше, чтобы получить **математическое обоснование** и **бизнес-рекомендации**.")

col1, col2 = st.columns([1, 1.5])

with col1:
    # Интерактивный выбор заявки
    selected_idx = st.number_input("Введите номер строки из таблицы (от 0):", min_value=0, max_value=len(top_farmers)-1, value=0)
    
    # Достаем данные выбранного фермера
    farmer_data = top_farmers.iloc[selected_idx]
    score = farmer_data['FutureScore']
    climate = farmer_data.get('climate_risk', 0)
    breeding = farmer_data.get('is_breeding', 1) # По умолчанию 1, если нет колонки
    
    st.subheader(f"Результат для заявки № {selected_idx}")
    
    # 1. Светофор (Бизнес-статус)
    if score >= 80:
        st.success(f"🟢 **СТАТУС: Одобрить** (Балл: {score}/100)\n\nХозяйство показывает высокую надежность и технологичность.")
    elif score >= 60:
        st.warning(f"🟡 **СТАТУС: Требует внимания** (Балл: {score}/100)\n\nСредний уровень риска. Рекомендуется дополнительная проверка.")
    else:
        st.error(f"🔴 **СТАТУС: Высокий риск** (Балл: {score}/100)\n\nНизкая прогнозируемая эффективность субсидий.")
        
    # 2. Текстовые рекомендации ИИ (Тот самый крутой функционал)
    st.markdown("#### 💡 Анализ ИИ-советника:")
    if climate >= 0.6:
        st.info(f"🌍 **Влияние климата:** Регион имеет повышенный климатический риск (индекс {climate}). Оценка учитывает сложные погодные условия (засуху), что предотвращает несправедливое занижение балла.")
    else:
        st.info(f"🌱 **Стабильный климат:** Хозяйство работает в благоприятной зоне (риск {climate}). Ожидается высокая урожайность/продуктивность.")
        
    if breeding == 0 and score < 80:
        st.warning("🐄 **Точка роста:** Хозяйство не использует племенное дело. Переход на породистый скот или использование селекции значительно повысит рейтинг FutureScore.")

with col2:
    st.write("**Факторы принятия решения (Explainable AI - SHAP):**")
    st.write("График показывает, какие именно параметры повысили (красный) или понизили (синий) итоговый балл.")
    
    # Подготовка данных для SHAP именно для этой строки
    # Важно: берем данные в том же порядке, как училась модель
    row_for_shap = farmer_data[features_list]
    
    try:
        # Расчет SHAP
        shap_values = explainer.shap_values(pd.DataFrame([row_for_shap]))
        
        # Отрисовка
        fig, ax = plt.subplots(figsize=(8, 4))
        shap.plots._waterfall.waterfall_legacy(
            explainer.expected_value, 
            shap_values[0], 
            feature_names=features_list, 
            show=False
        )
        # Настройка фона для красивого отображения в Streamlit
        fig.patch.set_facecolor('none')
        ax.set_facecolor('none')
        st.pyplot(fig)
    except Exception as e:
        st.error(f"Не удалось построить график SHAP для данной строки. Детали: {e}")

st.sidebar.divider()
st.sidebar.markdown("""
**Разработано командой:**
* Ж. Нурланкызы (Data)
* Ж. Айгараев (ML & XAI)
* А. Еркин (Cloud & UI)
""")
