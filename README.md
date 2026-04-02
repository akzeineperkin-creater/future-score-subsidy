# 🚜 FutureScore: AI-Driven Smart Subsidy Ecosystem

**FutureScore** — это инновационная экосистема интеллектуальной оценки и анти-фрод контроля сельскохозяйственных субсидий. Проект разработан для Министерства сельского хозяйства РК в рамках хакатона **Decentrathon 5.0**.

---

## 🎯 Миссия проекта
Переход от неэффективной очереди «First-Come, First-Served» к **Merit-based** подходу. Мы внедрили систему «Тройного фильтра», которая гарантирует, что господдержку получат только реальные, законные и эффективные хозяйства.

---

## 🛠 Ключевые модули системы (The "Triple-Check" System)

### 1. 🧠 ML Scoring Engine (XGBoost & SHAP)
* **Алгоритм:** Градиентный бустинг на базе **XGBoost**, обученный на 33,000+ записей.
* **Explainable AI (XAI):** Использование **SHAP Waterfall** диаграмм для расшифровки каждого балла. Система объясняет комиссару: «Почему у этого фермера 75 баллов?».
* **Факторы:** Учитываются исторические показатели района, климатические риски и технологичность (селекция/племя).

### 2. ⚖️ Legal AI Auditor (Order №108 Compliance)
* **Суть:** NLP-модуль, в который «вшита» логика **Приказа МСХ РК №108**.
* **Функционал:** Автоматическая сверка заявки с нормативами. ИИ выявляет логические ошибки (например, запрос на селекцию без племенного статуса) и превышение лимитов поголовья.
* **Результат:** Снижение коррупционных рисков за счет исключения человеческого фактора при первичной проверке документов.

### 3. 📸 Optical Anti-Fraud Control (Computer Vision)
* **Технология:** Нейросеть **YOLOv8 (Ultralytics)**.
* **Решаемая задача:** Борьба с «бумажным скотом».
* **Логика:** Система сопоставляет количество голов, заявленное в документах, с реальным количеством объектов (коров, овец, лошадей), обнаруженных на фотоотчетах или спутниковых снимках. При критическом расхождении заявка блокируется.

---

## 🚀 Инновационные признаки (Feature Engineering)

Мы обогатили государственные данные уникальными метриками:
* **🌍 Climate Risk Index:** Интеграция данных по засушливости и деградации почв. Система лояльнее к фермерам в зонах экстремального климата.
* **📊 Amount-to-Norm Ratio:** Аналитическая метрика адекватности запрошенных средств относительно рыночных нормативов.
* **🧬 Breeding & Selection Priority:** Повышающий коэффициент для хозяйств, работающих над генофондом страны.

---

## 🏗 Технологический стек

* **Frontend:** Streamlit (UI/UX для госслужащих).
* **ML:** Python, XGBoost, Scikit-learn, Pandas.
* **Computer Vision:** YOLOv8 (Object Detection).
* **Explainability:** SHAP (Waterfall plots), Plotly.
* **NLP & Logic:** Custom Rule-based Engine (Legal compliance).

---

## 📁 Структура проекта

* `app.py` — Ядро приложения и интерфейс.
* `final_dataset_pro.csv` — Обработанный датасет (33k+ строк).
* `yolov8n.pt` — Веса нейросети для распознавания скота.
* `futurescore_model_pro.pkl` — Обученная модель скоринга.
* `requirements.txt` — Список зависимостей.

---

## 🛠 Инструкция по установке

```bash
# Клонирование
git clone https://github.com/saylaaur/future-score-subsidy.git

# Установка зависимостей (требуется Python 3.9+)
pip install -r requirements.txt

# Запуск локально
streamlit run app.py
```

---

## 🔗 Ссылки
* **Live Web App:** [FutureScore Online](https://future-score-subsidy-x6lshrlsiuq3frnn5mvdhn.streamlit.app/)
* **Source Code:** [GitHub Repository](https://github.com/saylaaur/future-score-subsidy)

---

## 👥 Команда (Decentrathon 5.0)

* **Zhaniya Nurlankyzy** — Data Strategy, Data Cleaning, EDA & Legal Logic Mapping.
* **Zhangirkhan Aigarayev** — ML Engineering (XGBoost), CV Integration (YOLOv8) & Explainable AI (SHAP).
* **Akzeinep Erkin** — UI/UX Lead, Cloud Deployment & Frontend Development.

---
*Developed with 💡 and 🚜 for a more transparent agriculture in Kazakhstan. April 2026.*

---
