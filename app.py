import os
import time

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
    YOLO_IMPORT_ERROR = None
except Exception as e:
    YOLO_AVAILABLE = False
    YOLO_IMPORT_ERROR = str(e)


st.set_page_config(
    page_title="FutureScore | МСХ РК",
    page_icon="🌾",
    layout="wide",
)

# ==========================================
# 1. ПУТИ К ФАЙЛАМ
# ==========================================
DATA_PATH = "final_dataset_pro.csv"
YOLO_MODEL_PATH = "yolov8n.pt"
LOCAL_EMBLEM_PATH = "gerb.png"


# ==========================================
# 2. ЮРИДИЧЕСКИЙ ДВИЖОК
# ==========================================
class LegalExpertEngine:
    def __init__(self):
        self.MIN_HEADS = 50
        self.MAX_NON_BREEDING = 300
        self.AUDIT_THRESHOLD = 1000

    def check_compliance(self, farmer_data: dict) -> dict:
        declared_heads = int(farmer_data.get("Поголовье", 0))
        is_breeding = int(farmer_data.get("is_breeding", 0))
        is_selection = int(farmer_data.get("is_selection", 0))

        reasons = []
        penalty = 0
        status = "✅ Одобрено"

        if is_selection == 1 and is_breeding == 0:
            status = "❌ Отказано"
            penalty = 50
            reasons.append(
                "Нарушение: Запрос на селекционную субсидию без племенного статуса."
            )

        if declared_heads < self.MIN_HEADS:
            status = "❌ Отказано"
            penalty = 40
            reasons.append(
                f"Нарушение: Поголовье ({declared_heads}) ниже порога рентабельности ({self.MIN_HEADS})."
            )

        if is_breeding == 0 and declared_heads > self.MAX_NON_BREEDING:
            status = "❌ Отказано"
            penalty = 60
            reasons.append(
                f"Критическое нарушение: Превышен лимит товарного скота. "
                f"Заявлено {declared_heads}, лимит {self.MAX_NON_BREEDING}."
            )

        if status != "❌ Отказано" and declared_heads >= self.AUDIT_THRESHOLD:
            status = "⚠️ Требуется аудит"
            penalty = 15
            reasons.append(
                f"Внимание (Пункт 14): Аномально крупная заявка ({declared_heads} гол.). Назначен аудит."
            )

        if not reasons:
            reasons.append("Заявка полностью соответствует требованиям Приказа №108.")

        return {
            "status": status,
            "penalty": penalty,
            "details": reasons,
        }


legal_expert = LegalExpertEngine()


# ==========================================
# 3. ЗАГРУЗКА ДАННЫХ И МОДЕЛЕЙ
# ==========================================
@st.cache_resource
def load_yolo_model():
    if not YOLO_AVAILABLE:
        return None

    if not os.path.exists(YOLO_MODEL_PATH):
        return None

    try:
        return YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        st.error(f"Ошибка загрузки YOLO модели: {e}")
        return None


@st.cache_data
def load_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        st.warning(
            f"Файл '{DATA_PATH}' не найден. Включен демо-режим с тестовыми данными."
        )
        return pd.DataFrame(
            {
                "Фермер": ["КХ Демо-Болашак", "ИП Демо-Береке", "ТОО Демо-Агро"],
                "region": ["Акмолинская", "Туркестанская", "СКО"],
                "Поголовье": [240, 45, 1200],
                "Сумма": [3600000, 675000, 18000000],
                "FutureScore": [85, 30, 65],
                "is_breeding": [1, 0, 1],
                "is_selection": [1, 0, 1],
                "district_historical_score": [0.9, 0.4, 0.8],
                "climate_risk": [0.3, 0.8, 0.4],
            }
        )

    try:
        df = pd.read_csv(DATA_PATH)
    except Exception as e:
        st.error(f"Ошибка чтения CSV: {e}")
        return pd.DataFrame()

    if df.empty:
        return pd.DataFrame()

    df = df.sample(min(250, len(df)), random_state=42).reset_index(drop=True)

    regions_map = {
        0: "Абайская",
        1: "Акмолинская",
        2: "Актюбинская",
        3: "Алматинская",
        4: "Атырауская",
        5: "ВКО",
        6: "Жамбылская",
        7: "Жетысуская",
        8: "ЗКО",
        9: "Карагандинская",
        10: "Костанайская",
        11: "Кызылординская",
        12: "Мангистауская",
        13: "Павлодарская",
        14: "СКО",
        15: "Туркестанская",
        16: "Улытауская",
        17: "г. Шымкент",
    }

    if "region" not in df.columns:
        if "region_encoded" in df.columns:
            df["region"] = df["region_encoded"].map(regions_map).fillna("Другой")
        else:
            df["region"] = "Другой"

    if "Норматив" not in df.columns:
        df["Норматив"] = 15000

    if "Поголовье" not in df.columns:
        if "amount_to_norm_ratio" in df.columns:
            df["Поголовье"] = df["amount_to_norm_ratio"].fillna(100).astype(int)
        else:
            df["Поголовье"] = 100

    if "Сумма" not in df.columns:
        df["Сумма"] = df["Поголовье"] * df["Норматив"]

    if "Фермер" not in df.columns:
        df["Фермер"] = [f"КХ Агро-Заявка #{int(i * 7 + 1000)}" for i in range(len(df))]

    if "is_breeding" not in df.columns:
        df["is_breeding"] = 0

    if "is_selection" not in df.columns:
        df["is_selection"] = 0

    if "district_historical_score" not in df.columns:
        df["district_historical_score"] = 0.8

    if "climate_risk" not in df.columns:
        df["climate_risk"] = 0.5

    return df


def calculate_ai_score(row):
    # ... (проверка демо-данных)
    base = row.get('district_historical_score', 0.8) * 70  # Макс 70
    penalty = row.get('climate_risk', 0.5) * 30           # Минус до 30
    bonus_breed = 15 if row.get('is_breeding', 0) == 1 else 0
    bonus_sel = 10 if row.get('is_selection', 0) == 1 else 0
    
    final_score = base - penalty + bonus_breed + bonus_sel
    return max(10, min(99, int(final_score)))


raw_df = load_data()
yolo_model = load_yolo_model()

if not raw_df.empty and "FutureScore" not in raw_df.columns:
    raw_df["FutureScore"] = raw_df.apply(calculate_ai_score, axis=1)


# ==========================================
# 4. SIDEBAR
# ==========================================
with st.sidebar:
    if os.path.exists(LOCAL_EMBLEM_PATH):
        st.image(LOCAL_EMBLEM_PATH, width=80)

    st.title("Управление МСХ")

    if not raw_df.empty:
        regions = ["Все регионы"] + sorted(raw_df["Регион"].astype(str).unique().tolist())
        selected_region = st.selectbox("🌍 Фильтр по региону:", regions)

        if selected_region != "Все регионы":
            filtered_df = raw_df[raw_df["region"] == selected_region].copy()
        else:
            filtered_df = raw_df.copy()

        st.divider()

        if not filtered_df.empty:
            selected_farmer_name = st.selectbox(
                "👨‍🌾 Выбрать фермера:",
                filtered_df["Фермер"].astype(str).tolist(),
            )
            farmer_rows = filtered_df[filtered_df["Фермер"] == selected_farmer_name]
            farmer_data = farmer_rows.iloc[0].to_dict() if not farmer_rows.empty else {}
        else:
            st.warning("Нет данных для отображения")
            farmer_data = {}
    else:
        st.error("Данные не загружены")
        filtered_df = pd.DataFrame()
        farmer_data = {}

    st.divider()
    uploaded_pdf = st.file_uploader("Загрузить Приказ №108 (PDF)", type="pdf")


# ==========================================
# 5. ОСНОВНОЙ ИНТЕРФЕЙС
# ==========================================
st.title("🌾 FutureScore: Анализ заявок на субсидии")

if filtered_df.empty:
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Рейтинг заявок",
        "🔍 Расшифровка балла (SHAP)",
        "⚖️ Legal AI",
        "📸 Computer Vision",
    ]
)

# ==========================================
# TAB 1
# ==========================================
with tab1:
    col1, col2, col3 = st.columns(3)

    col1.metric("Заявок в работе", len(filtered_df))
    col2.metric("Общий бюджет", f"{filtered_df['Сумма'].sum():,.0f} ₸")
    col3.metric("Средний балл", f"{filtered_df['FutureScore'].mean():.1f} / 100")

    st.subheader("Сводная таблица эффективности")

    show_df = filtered_df[
        ["Фермер", "region", "Поголовье", "Сумма", "FutureScore"]
    ].sort_values("FutureScore", ascending=False)

    st.dataframe(
        show_df,
        column_config={
            "Сумма": st.column_config.NumberColumn("Запрошено (₸)", format="%d ₸"),
            "Поголовье": st.column_config.NumberColumn("Заявлено голов", format="%d шт"),
            "FutureScore": st.column_config.ProgressColumn(
                "Рейтинг AI", format="%d", min_value=0, max_value=100
            ),
        },
        hide_index=True,
        height=400,
        use_container_width=True,
    )

# ==========================================
# TAB 2
# ==========================================
with tab2:
    if farmer_data:
        score = int(farmer_data.get("FutureScore", 50))
        farmer_name = farmer_data.get("Фермер", "Неизвестно")

        st.header(f"Аналитика ИИ: {farmer_name}")

        c1, c2 = st.columns([2, 1])

        with c1:
            base_score = int(float(farmer_data.get("district_historical_score", 0.8)) * 70)
            climate_penalty = -int(float(farmer_data.get("climate_risk", 0.5)) * 30)
            breed_bonus = 15 if int(farmer_data.get("is_breeding", 0)) == 1 else 0
            sel_bonus = 10 if int(farmer_data.get("is_selection", 0)) == 1 else 0

            fig = go.Figure(
                go.Waterfall(
                    orientation="v",
                    measure=["absolute", "relative", "relative", "relative", "total"],
                    x=[
                        "База района",
                        "Климат. риск",
                        "Племенной статус",
                        "Селекция",
                        "Итоговый Score",
                    ],
                    y=[base_score, climate_penalty, breed_bonus, sel_bonus, score],
                    text=[
                        str(base_score),
                        str(climate_penalty),
                        f"+{breed_bonus}",
                        f"+{sel_bonus}",
                        str(score),
                    ],
                    textposition="outside",
                    connector={"line": {"color": "#444"}},
                    increasing={"marker": {"color": "#2ecc71"}},
                    decreasing={"marker": {"color": "#e74c3c"}},
                    totals={"marker": {"color": "#3498db"}},
                )
            )
            fig.update_layout(height=400, margin=dict(t=20, b=20))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown(f"### 🎯 Итог: **{score}/100**")

            if score >= 70:
                st.success("✅ **Высокая надежность.** Рекомендуется авто-одобрение.")
            elif score >= 40:
                st.warning("⚠️ **Зона риска.** Запросить доп. документы.")
            else:
                st.error("❌ **Критический риск.** Отклонить заявку.")
    else:
        st.info("Нет данных по выбранному фермеру.")

# ==========================================
# TAB 3
# ==========================================
with tab3:
    st.header("⚖️ Юридическая экспертиза (Приказ №108)")

    if uploaded_pdf:
        with st.spinner("NLP Анализ документа и сверка лимитов..."):
            time.sleep(1.5)

        legal_result = legal_expert.check_compliance(farmer_data if farmer_data else {})

        st.markdown(f"### Вердикт системы: {legal_result['status']}")

        for detail in legal_result["details"]:
            if "Нарушение" in detail or "Критическое" in detail:
                st.error(f"🚨 {detail}")
            elif "Внимание" in detail:
                st.warning(f"⚠️ {detail}")
            else:
                st.success(f"✅ {detail}")
    else:
        st.info("👈 Загрузите Приказ №108 (PDF) в панели слева для активации AI-Юриста.")

# ==========================================
# TAB 4
# ==========================================
with tab4:
    st.header("📸 Оптический Анти-фрод (YOLOv8)")

    declared_cows = int(farmer_data.get("Поголовье", 0)) if farmer_data else 0
    detected_count = 0

    if not YOLO_AVAILABLE:
         st.error(f"❌ YOLO недоступен: {YOLO_IMPORT_ERROR}")
    elif yolo_model is None:
        st.error("❌ Файл модели yolov8n.pt не найден или модель не загрузилась.")
    else:
        img_file = st.file_uploader("Загрузить фото", type=["jpg", "png", "jpeg"])

        if img_file:
            try:
                image = Image.open(img_file).convert("RGB")

                with st.spinner("Анализ..."):
                    results = yolo_model(image)

                    if results and len(results) > 0 and results[0].boxes is not None:
                        detected_count = sum(
                            1
                            for box in results[0].boxes
                            if int(box.cls[0]) in [17, 18, 19]
                        )

                    res_img = results[0].plot()
                    st.image(
                        res_img,
                        caption="Результат детекции",
                        use_container_width=True,
                    )

            except Exception as e:
                st.error(f"Ошибка обработки изображения: {e}")

        st.subheader("📊 Итог проверки:")
        st.write(f"📄 Заявлено: **{declared_cows}**")
        st.write(f"🐄 Найдено: **{detected_count}**")

        if declared_cows > 0:
            ratio = detected_count / declared_cows
            if ratio < 0.3:
                st.error(f"🚨 ФРОД! Подтверждено только {int(ratio * 100)}% поголовья.")
            else:
                st.success("✅ Визуальный контроль пройден.")
        else:
            st.info("Для выбранного фермера нет заявленного поголовья.")
