import streamlit as st
import pandas as pd
import joblib
import os
import plotly.express as px
import numpy as np

# Настройка страницы
st.set_page_config(page_title="FutureScore", layout="wide", page_icon="🐄")

# --- 1. ЗАГРУЗКА AI-АКТИВОВ ---
@st.cache_resource
def load_ai_assets():
    model_path = 'models/futurescore_model_pro.pkl'
    artifacts_path = 'models/data_pipeline_artifacts_pro (1).pkl'
    
    if os.path.exists(model_path) and os.path.exists(artifacts_path):
        try:
            raw_data = joblib.load(model_path)
            # Извлекаем модель из словаря, если нужно
            if isinstance(raw_data, dict):
                model = raw_data.get('xgboost_model') or raw_data.get('model') or list(raw_data.values())[0]
            else:
                model = raw_data
                
            artifacts = joblib.load(artifacts_path)
            return model, artifacts
        except Exception as e:
            return None, None
    return None, None

# Сначала получаем переменные!
model, artifacts = load_ai_assets()

# --- 2. САЙДБАР (Теперь он знает про model) ---
with st.sidebar:
    st.title("🐄 FutureScore")
    if model is not None:
        st.success("🚀 AI-модель готова!")
    else:
        st.warning("⚠️ Файлы модели не найдены в /models")
    
    st.info("MVP для Минсельхоза РК")

# --- 3. ГЛАВНЫЙ ИНТЕРФЕЙС ---
st.title("Система скоринга субсидий")

tab1, tab2, tab3 = st.tabs(["📊 Рейтинг", "🔍 Детали", "📋 Рекомендации"])

with tab1:
    st.subheader("📊 Реальный рейтинг FutureScore")
    
    # Сначала создаем пустую переменную
    df_final = None
    
    # 1. Загружаем данные
    try:
        if os.path.exists('data/final_dataset_pro.csv'):
            df_final = pd.read_csv('data/final_dataset_pro.csv')
        else:
            st.error("📁 Файл 'data/final_dataset_pro.csv' не найден!")
    except Exception as e:
        st.error(f"❌ Ошибка при чтении CSV: {e}")

    # 2. Если данные есть — считаем
    if df_final is not None:
        if model is not None:
            try:
                # Готовим цифры для модели
                X = df_final.copy()
                if 'target' in X.columns:
                    X = X.drop(columns=['target'])
                
                X_values = X.values 
                
                # Добиваем до 36 колонок (как просит модель)
                if X_values.shape[1] == 35:
                    X_values = np.hstack([X_values, np.zeros((X_values.shape[0], 1))])
                X_values = X_values[:, :36]

                # Прогноз
                if hasattr(model, "predict_proba"):
                    df_final['FutureScore'] = model.predict_proba(X_values)[:, 1] * 100
                else:
                    df_final['FutureScore'] = model.predict(X_values)

                df_final = df_final.sort_values('FutureScore', ascending=False)
                
                st.success(f"✅ Рейтинг рассчитан для {len(df_final)} хозяйств")
                
                # Таблица
                cols = ['FutureScore'] + [c for c in df_final.columns if c != 'FutureScore'][:5]
                st.dataframe(df_final[cols].head(50), use_container_width=True)
                
                # График
                fig = px.histogram(df_final, x='FutureScore', title="Распределение баллов надежности")
                st.plotly_chart(fig, use_container_width=True)

            except Exception as predict_error:
                st.error(f"⚠️ Ошибка в расчетах: {predict_error}")
        else:
            st.warning("🤖 Модель загружена, но не готова к предикту.")

            # Добавляем поиск
        st.divider()
        search_id = st.text_input("🔍 Найти хозяйство по ID", "")
        if search_id:
            # Ищем строку, где id совпадает (приводим к строке для верности)
            result = df_final[df_final['id'].astype(str) == search_id]
            if not result.empty:
                st.write(f"### Результат для ID {search_id}:")
                st.metric("Балл надежности", f"{result['FutureScore'].values[0]:.2f}%")
                st.dataframe(result)
            else:
                st.warning("Хозяйство с таким ID не найдено")
with tab2:
    st.info("🔍 Здесь будет детальный разбор факторов (SHAP) для каждой заявки.")

with tab3:
    st.info("📋 Здесь будут формироваться автоматические рекомендации для комиссии.")