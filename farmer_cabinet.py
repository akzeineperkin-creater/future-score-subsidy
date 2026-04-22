"""
=============================================================
  УЧАСТНИК №3 — Личный кабинет фермера
  Streamlit-страница: подача заявки, XGBoost скоринг, SHAP
=============================================================
Зависимости:
    pip install streamlit xgboost shap matplotlib pandas numpy scikit-learn
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import shap
import datetime
import re
import random

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder


# ─────────────────────────────────────────────
#  ДЕМО-МОДЕЛЬ XGBoost
# ─────────────────────────────────────────────
@st.cache_resource
def load_model():
    rng = np.random.default_rng(42)
    n = 500

    livestock   = rng.integers(10, 500, n).astype(float)
    deaths      = rng.integers(0, 50, n).astype(float)
    death_rate  = deaths / (livestock + 1)
    amount      = rng.uniform(50_000, 5_000_000, n)
    region_enc  = rng.integers(0, 17, n).astype(float)
    years_work  = rng.integers(1, 30, n).astype(float)

    X = pd.DataFrame({
        "поголовье":      livestock,
        "падёж":          deaths,
        "доля_падежа":    death_rate,
        "сумма_субсидии": amount,
        "регион":         region_enc,
        "стаж_лет":       years_work,
    })

    y = ((death_rate < 0.1) & (amount < 2_000_000)).astype(int)

    model = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    model.fit(X, y)

    explainer = shap.TreeExplainer(model)
    return model, explainer, list(X.columns)


MODEL, EXPLAINER, FEATURE_NAMES = load_model()

REGIONS = [
    "Алматинская", "Акмолинская", "Актюбинская", "Атырауская",
    "Восточно-Казахстанская", "Жамбылская", "Западно-Казахстанская",
    "Карагандинская", "Костанайская", "Кызылординская", "Мангистауская",
    "Павлодарская", "Северо-Казахстанская", "Туркестанская",
    "г. Алматы", "г. Астана", "г. Шымкент",
]
REGION_MAP = {r: i for i, r in enumerate(REGIONS)}


# ─────────────────────────────────────────────
#  ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ─────────────────────────────────────────────
def validate_bin(bin_str: str) -> bool:
    return bool(re.fullmatch(r"\d{12}", bin_str.strip()))


def is_duplicate(bin_str: str) -> bool:
    for app in st.session_state.db_apps:
        if app["bin"] == bin_str and app["status"] == "pending":
            return True
    return False


def compute_score(livestock, deaths, amount, region_idx, years_work=5):
    death_rate = deaths / (livestock + 1)
    X = pd.DataFrame([{
        "поголовье":      livestock,
        "падёж":          deaths,
        "доля_падежа":    death_rate,
        "сумма_субсидии": amount,
        "регион":         region_idx,
        "стаж_лет":       years_work,
    }])
    prob      = MODEL.predict_proba(X)[0][1]
    score     = int(round(prob * 100))
    shap_vals = EXPLAINER.shap_values(X)
    return score, shap_vals, X


def shap_figure(shap_values, X_row):
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.set_facecolor("#f8faf9")
    fig.patch.set_facecolor("#f8faf9")

    vals   = shap_values[0]
    names  = FEATURE_NAMES
    colors = ["#c0392b" if v < 0 else "#27ae60" for v in vals]

    y_pos = range(len(names))
    bars  = ax.barh(y_pos, vals, color=colors, edgecolor="none", height=0.55)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(names, fontsize=10)
    ax.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Вклад в решение (SHAP)", fontsize=9)
    ax.set_title("Почему такой балл? (SHAP-объяснение)", fontsize=11, fontweight="bold", pad=8)

    for bar, val in zip(bars, vals):
        ax.text(
            val + (0.003 if val >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{val:+.3f}",
            va="center",
            ha="left" if val >= 0 else "right",
            fontsize=8,
            color="#333",
        )

    plt.tight_layout()
    return fig


def score_label(score: int):
    if score >= 65:
        return "score-high", "✅ Высокая вероятность одобрения"
    elif score >= 40:
        return "score-mid",  "⚠️ Средняя вероятность — требуется проверка"
    else:
        return "score-low",  "❌ Низкая вероятность одобрения"


def status_badge(status: str) -> str:
    mapping = {
        "pending":  '<span class="status-pending">⏳ На рассмотрении</span>',
        "approved": '<span class="status-approved">✅ Одобрена</span>',
        "rejected": '<span class="status-rejected">❌ Отклонена</span>',
    }
    return mapping.get(status, status)


# ─────────────────────────────────────────────
#  ГЛАВНАЯ ФУНКЦИЯ — вызывается из main.py
# ─────────────────────────────────────────────
def main():

    # ── CSS ──────────────────────────────────
    st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,500;1,9..40,300&family=Space+Mono:wght@400;700&display=swap');
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #f7f9f4;
    color: #1a2e0d;
  }
  .cabinet-header {
    background: #3B6D11; color: #EAF3DE;
    padding: 1.8rem 2.4rem; border-radius: 14px;
    margin-bottom: 1.5rem; position: relative; overflow: hidden;
  }
  .cabinet-header::before {
    content: ''; position: absolute;
    top: -40px; right: -40px; width: 160px; height: 160px;
    background: rgba(160,220,100,0.08); border-radius: 50%;
  }
  .cabinet-header h1 { margin: 0; font-size: 1.5rem; font-weight: 500; color: #EAF3DE; }
  .cabinet-header p  { margin: 0.3rem 0 0; opacity: 0.7; font-size: 0.85rem; font-family: 'Space Mono', monospace; }
  .score-block { text-align: center; padding: 1.5rem; border-radius: 12px; font-weight: 500; }
  .score-high  { background: #EAF3DE; color: #27500A; border: 1px solid #C0DD97; }
  .score-mid   { background: #FAEEDA; color: #633806; border: 1px solid #FAC775; }
  .score-low   { background: #FCEBEB; color: #A32D2D; border: 1px solid #F7C1C1; }
  .status-pending  { color: #633806; background: #FAEEDA; padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 500; font-family: 'Space Mono', monospace; }
  .status-approved { color: #27500A; background: #EAF3DE; padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 500; font-family: 'Space Mono', monospace; }
  .status-rejected { color: #A32D2D; background: #FCEBEB; padding: 3px 10px; border-radius: 6px; font-size: 0.75rem; font-weight: 500; font-family: 'Space Mono', monospace; }
</style>
""", unsafe_allow_html=True)

    # ── SESSION STATE ─────────────────────────
    if "db_apps" not in st.session_state:
        st.session_state.db_apps = []
    if "current_farmer_bin" not in st.session_state:
        st.session_state.current_farmer_bin = None
    if "last_result" not in st.session_state:
        st.session_state.last_result = None

    # ── Синхронизация статусов из CSV → db_apps ──────────────────────────────
    # Аудитор меняет статус в CSV; фермер должен видеть актуальный статус.
    try:
        import os as _os, pandas as _pd
        _CSV = "data/applications.csv"
        if _os.path.exists(_CSV):
            _csv_df = _pd.read_csv(_CSV)
            for _app in st.session_state.db_apps:
                _mask = (
                    (_csv_df["bin"].astype(str) == str(_app["bin"])) &
                    (_csv_df["submitted_at"].astype(str) == str(_app["submitted_at"]))
                )
                if _mask.any():
                    _row = _csv_df[_mask].iloc[0]
                    _app["status"]         = str(_row.get("status", _app["status"]))
                    _app["reviewed_by"]    = _row.get("reviewed_by", None)
                    _app["reviewed_at"]    = _row.get("reviewed_at", None)
                    _app["review_comment"] = _row.get("review_comment", None)
    except Exception:
        pass

    # ── ЗАГОЛОВОК ─────────────────────────────
    st.markdown("""
<div class="cabinet-header">
  <h1>🌾 Личный кабинет фермера</h1>
  <p>Подача заявки на субсидию · Автоматический скоринг · Прозрачное решение</p>
</div>
""", unsafe_allow_html=True)

    tab_form, tab_apps, tab_help = st.tabs(["📋 Подать заявку", "📂 Мои заявки", "❓ Помощь"])

    # ══════════════════════════════════════════
    #  ВКЛАДКА 1 — ФОРМА ПОДАЧИ
    # ══════════════════════════════════════════
    with tab_form:
        with st.form("farmer_application_form"):
            with st.expander("📌 Реквизиты хозяйства", expanded=True):
                # Строка 1: название, БИН, регион
                col1, col2, col3 = st.columns([2, 1.5, 1.5])
                with col1:
                    farm_name = st.text_input("Название хозяйства", placeholder='ТОО "АгроПром"...', key="f_name")
                with col2:
                    bin_input = st.text_input("БИН (12 цифр)", max_chars=12, key="f_bin")
                with col3:
                    region = st.selectbox("Регион", REGIONS, key="f_reg")

                # Строка 2: ИИН, email, телефон
                col4, col5, col6 = st.columns([1.5, 2, 1.5])
                with col4:
                    iin_input = st.text_input("ИИН руководителя (12 цифр)", max_chars=12, key="f_iin")
                with col5:
                    email_input = st.text_input("Электронная почта", placeholder="example@mail.kz", key="f_email")
                with col6:
                    phone_input = st.text_input("Номер телефона", placeholder="+7", key="f_phone")

            with st.expander("📊 Производственные показатели", expanded=True):
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    livestock = st.number_input("Общее поголовье (голов)", min_value=1, value=100, key="f_live")
                with c2:
                    deaths = st.number_input("Падеж за период (головы)", min_value=0, value=2, key="f_death")
                with c3:
                    years_work = st.number_input("Стаж работы (лет)", min_value=0, value=5, key="f_years")
                with c4:
                    hectares = st.number_input("Площадь угодий (гектар)", min_value=0.1, step=0.1, value=10.0, key="f_hectares")

            with st.expander("💰 Запрос субсидий", expanded=True):
                requested_amount = st.number_input("Сумма субсидии (₸)", min_value=100000, step=100000, key="f_amount")

            submitted = st.form_submit_button("🚀 Рассчитать Future Score и отправить заявку")

        if submitted:
            errors = []
            if not farm_name.strip():
                errors.append("Укажите название хозяйства.")
            if not validate_bin(bin_input):
                errors.append("БИН должен содержать ровно 12 цифр.")
            if not re.fullmatch(r"\d{12}", iin_input.strip()):
                errors.append("ИИН должен содержать ровно 12 цифр.")
            if "@" not in email_input.strip():
                errors.append("Email должен содержать символ «@».")
            if not phone_input.strip():
                errors.append("Укажите номер телефона.")
            if hectares <= 0:
                errors.append("Площадь угодий должна быть больше 0 гектар.")
            if deaths > livestock:
                errors.append("Количество падежа не может превышать поголовье.")
            if requested_amount <= 0:
                errors.append("Запрашиваемая сумма должна быть больше 0.")
            if is_duplicate(bin_input.strip()):
                errors.append(
                    f"По БИН {bin_input} уже есть активная заявка. "
                    "Дождитесь решения перед повторной подачей."
                )

            if errors:
                for err in errors:
                    st.error(f"❌ {err}")
            else:
                with st.spinner("⚙️ Рассчитываем скоринговый балл..."):
                    region_idx = REGION_MAP[region]
                    score, shap_vals, X_row = compute_score(
                        livestock  = float(livestock),
                        deaths     = float(deaths),
                        amount     = float(requested_amount),
                        region_idx = float(region_idx),
                        years_work = float(years_work),
                    )

                st.success("✅ Заявка принята и отправлена на рассмотрение!")
                st.balloons()
                st.markdown("---")
                st.markdown("### 📊 Ваш скоринговый результат")

                css_class, label_text = score_label(score)
                st.markdown(
                    f'<div class="score-block {css_class}">'
                    f'<div style="font-size:3rem;margin-bottom:0.3rem">{score}</div>'
                    f'<div style="font-size:1rem">{label_text}</div>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

                st.markdown("#### 🔍 Объяснение решения (SHAP)")
                st.caption(
                    "Зелёные столбцы — факторы, которые **повышают** балл. "
                    "Красные — **снижают**. Длина = сила влияния."
                )
                fig = shap_figure(shap_vals, X_row)
                st.pyplot(fig)
                plt.close(fig)

                app_record = {
                    "farm_name":        farm_name.strip(),
                    "bin":              bin_input.strip(),
                    "iin":              iin_input.strip(),
                    "email":            email_input.strip(),
                    "phone":            phone_input.strip(),
                    "region":           region,
                    "livestock":        int(livestock),
                    "hectares":         round(float(hectares), 2),
                    "deaths":           int(deaths),
                    "death_rate":       round(deaths / max(livestock, 1), 4),
                    "years_work":       int(years_work),
                    "requested_amount": int(requested_amount),
                    "score":            score,
                    "shap_values":      shap_vals[0].tolist(),
                    "feature_names":    FEATURE_NAMES,
                    "status":           "pending",
                    "submitted_at":     datetime.datetime.now().isoformat(),
                    "reviewed_by":      None,
                    "reviewed_at":      None,
                    "review_comment":   None,
                }

                st.session_state.db_apps.append(app_record)
                st.session_state.current_farmer_bin = bin_input.strip()
                st.session_state.last_result = app_record

                # ── Синхронизация с CSV (для кабинета аудитора) ────────────
                try:
                    import os as _os, pandas as _pd
                    _CSV = "data/applications.csv"
                    _os.makedirs("data", exist_ok=True)
                    _cols = [
                        "farm_name","bin","iin","email","phone",
                        "region","livestock","hectares","deaths","death_rate",
                        "years_work","requested_amount","score","shap_values","feature_names",
                        "status","submitted_at","reviewed_by","reviewed_at","review_comment",
                    ]
                    _row = {k: app_record.get(k, "") for k in _cols}
                    if _os.path.exists(_CSV):
                        _df = _pd.read_csv(_CSV)
                    else:
                        _df = _pd.DataFrame(columns=_cols)
                    _df = _pd.concat([_df, _pd.DataFrame([_row])], ignore_index=True)
                    _df.to_csv(_CSV, index=False)
                except Exception as _e:
                    st.warning(f"⚠️ Не удалось сохранить в CSV: {_e}")

                with st.expander("📄 Сводка заявки (для печати)", expanded=False):
                    summary_df = pd.DataFrame([{
                        "Хозяйство":       farm_name,
                        "БИН":             bin_input,
                        "ИИН":             iin_input,
                        "Email":           email_input,
                        "Телефон":         phone_input,
                        "Регион":          region,
                        "Поголовье":       livestock,
                        "Площадь (га)":    hectares,
                        "Падёж":           deaths,
                        "Сумма (₸)":       f"{requested_amount:,}",
                        "Балл":            score,
                        "Статус":          "На рассмотрении",
                        "Дата подачи":     datetime.datetime.now().strftime("%d.%m.%Y %H:%M"),
                    }]).T
                    summary_df.columns = ["Значение"]
                    st.dataframe(summary_df, use_container_width=True)

    # ══════════════════════════════════════════
    #  ВКЛАДКА 2 — МОИ ЗАЯВКИ
    # ══════════════════════════════════════════
    with tab_apps:
        st.markdown("### 📂 История заявок")

        filter_bin = st.text_input(
            "Введите ваш БИН для просмотра заявок",
            value=st.session_state.current_farmer_bin or "",
            placeholder="123456789012",
            max_chars=12,
        )

        my_apps = (
            [a for a in st.session_state.db_apps if a["bin"] == filter_bin.strip()]
            if filter_bin.strip() else []
        )

        if not my_apps:
            st.info("ℹ️ Заявок не найдено. Подайте заявку на вкладке «Подать заявку».")
        else:
            st.markdown(f"**Найдено заявок: {len(my_apps)}**")

            for i, app in enumerate(reversed(my_apps)):
                idx = len(my_apps) - i

                with st.container():
                    c1, c2, c3, c4 = st.columns([2, 1, 1, 1.5])
                    with c1:
                        st.markdown(f"**{app['farm_name']}**")
                        st.caption(f"Подано: {app['submitted_at'][:16].replace('T', ' ')}")
                    with c2:
                        st.metric("Балл", app["score"])
                    with c3:
                        st.metric("Сумма", f"{app['requested_amount']:,} ₸")
                    with c4:
                        badge_map = {"pending": "⏳ На рассмотрении", "approved": "✅ Одобрена", "rejected": "❌ Отклонена"}
                        st.markdown(f'**Статус:** {badge_map.get(app["status"], app["status"])}', unsafe_allow_html=True)
                        if app.get("review_comment"):
                            st.caption(f"Комментарий: {app['review_comment']}")

                    with st.expander(f"🔍 Детали скоринга заявки #{idx}"):
                        col_d1, col_d2 = st.columns(2)
                        with col_d1:
                            details = {
                                "Поголовье":   app["livestock"],
                                "Падёж":       app["deaths"],
                                "Доля падежа": f"{app['death_rate']*100:.1f}%",
                                "Регион":      app["region"],
                                "Стаж":        f"{app.get('years_work', '—')} лет",
                            }
                            st.table(pd.Series(details).rename("Значение"))
                        with col_d2:
                            if app.get("shap_values") and app.get("feature_names"):
                                shap_arr = np.array(app["shap_values"])
                                fig2, ax2 = plt.subplots(figsize=(5, 3))
                                ax2.set_facecolor("#f8faf9")
                                fig2.patch.set_facecolor("#f8faf9")
                                colors2 = ["#c0392b" if v < 0 else "#27ae60" for v in shap_arr]
                                ax2.barh(app["feature_names"], shap_arr, color=colors2, edgecolor="none", height=0.5)
                                ax2.axvline(0, color="#888", lw=0.8, ls="--")
                                ax2.set_title("SHAP", fontsize=9)
                                plt.tight_layout()
                                st.pyplot(fig2)
                                plt.close(fig2)

                    st.divider()

    # ══════════════════════════════════════════
    #  ВКЛАДКА 3 — ПОМОЩЬ
    # ══════════════════════════════════════════
    with tab_help:
        st.markdown("### ❓ Часто задаваемые вопросы")

        faq = {
            "Что такое скоринговый балл?": (
                "Балл (0–100) рассчитывается моделью XGBoost на основе данных хозяйства. "
                "Балл ≥ 65 — высокая вероятность одобрения. Окончательное решение принимает инспектор."
            ),
            "Что такое SHAP-график?": (
                "SHAP объясняет решение ИИ: зелёные столбцы повышают балл, красные — снижают."
            ),
            "Что такое 'Кол-во падежа'?": (
                "Количество голов, погибших за период. Падёж >10–15% снижает балл. Укажите 0, если не было."
            ),
            "Можно ли подать несколько заявок?": (
                "Нет. Одна активная заявка на БИН. Повторная подача — после получения решения."
            ),
            "Когда придёт решение?": (
                "Инспектор рассматривает заявку в течение 5 рабочих дней."
            ),
        }

        for question, answer in faq.items():
            with st.expander(question):
                st.write(answer)

        st.divider()
        st.markdown(
            "📞 **Служба поддержки:** +7 (7172) 00-00-00  \n"
            "📧 **Email:** support@agrosubsidy.kz  \n"
            "🕐 Пн–Пт, 09:00–18:00 (Астана)"
        )

    # ── САЙДБАР ───────────────────────────────
    with st.sidebar:
        st.markdown("## 🌾 АгроСубсидия")
        st.divider()

        total    = len(st.session_state.db_apps)
        pending  = sum(1 for a in st.session_state.db_apps if a["status"] == "pending")
        approved = sum(1 for a in st.session_state.db_apps if a["status"] == "approved")

        st.metric("Всего заявок (в системе)", total)
        st.metric("На рассмотрении",           pending)
        st.metric("Одобрено",                  approved)

        if st.session_state.last_result:
            st.divider()
            lr = st.session_state.last_result
            st.markdown("**Последняя заявка:**")
            st.write(f"🏠 {lr['farm_name']}")
            st.write(f"📊 Балл: **{lr['score']}**")
            st.write(f"💰 {lr['requested_amount']:,} ₸")

        st.divider()
       

        if st.button("🗑️ Сбросить все заявки (DEV)", type="secondary"):
            st.session_state.db_apps            = []
            st.session_state.current_farmer_bin = None
            st.session_state.last_result        = None
            st.rerun()
