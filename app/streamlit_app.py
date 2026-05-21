import streamlit as st
import requests

st.set_page_config(
    page_title="Horse Racing Predictor",
    page_icon="🐎",
    layout="wide"
)

API_URL = "http://127.0.0.1:8000/predict"
st.markdown("""
<style>
    /* Заголовок */
    .main-header {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 50%, #48dbfb 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 {
        color: #1a1a2e;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .main-header p {
        color: #2c3e50;
        margin-top: 0.3rem;
        font-weight: 600;
    }

    /* Плашки секций — ЧУТЬ БОЛЕЕ ЖЁЛТЫЕ */
    .section-header {
        background: linear-gradient(135deg, #ffe082 0%, #ffcc80 100%);
        padding: 0.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        text-align: center;
        border: none;
    }
    .section-header h3 {
        color: #1a1a2e !important;
        margin: 0;
        font-weight: bold;
    }

    /* Карточки результатов */
    .result-card-top3 {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-card-not-top3 {
        background: linear-gradient(135deg, #eb3349 0%, #f45c43 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .result-card-top3 h2, .result-card-not-top3 h2 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
    }
    .result-card-top3 p, .result-card-not-top3 p {
        color: #ffffff;
        margin: 0;
        font-weight: 500;
    }

    /* Вероятность и достоверность */
    .probability-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .probability-card h2 {
        color: #ffffff;
        margin: 0;
        font-size: 1.8rem;
    }
    .probability-card p {
        color: #ffffff;
        margin: 0;
        font-weight: 500;
    }
    .confidence-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 1rem;
        border-radius: 12px;
        text-align: center;
    }
    .confidence-card h2 {
        color: #ffffff;
        margin: 0;
        font-size: 1.5rem;
    }
    .confidence-card p {
        color: #ffffff;
        margin: 0;
        font-weight: 500;
    }

    .message-box {
        background: linear-gradient(135deg, #a8edea 0%, #fed6e3 100%);
        padding: 1rem;
        border-radius: 10px;
        margin-top: 1rem;
        text-align: center;
    }
    .message-box p {
        color: #2c3e50;
        margin: 0;
        font-weight: bold;
    }

    footer {
        text-align: center;
        margin-top: 2rem;
        padding: 1rem;
        color: #7f8c8d;
        font-size: 0.8rem;
        border-top: 1px solid #ecf0f1;
    }

    .stButton button {
        background: linear-gradient(135deg, #ff6b6b 0%, #feca57 100%);
        color: #1a1a2e !important;
        font-weight: bold;
        font-size: 1.2rem;
        border: none;
        border-radius: 30px;
        padding: 0.5rem 2rem;
    }
    .stButton button:hover {
        background: linear-gradient(135deg, #feca57 0%, #ff6b6b 100%);
        transform: scale(1.02);
        color: #1a1a2e !important;
    }
    label {
        font-weight: 600;
        color: #2c3e50;
    }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown("""
<div class="main-header">
    <h1>🏇 Horse Racing Predictor</h1>
    <p>Прогнозирование попадания лошади в топ-3 на скачках</p>
</div>
""", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    st.markdown('<div class="section-header"><h3>🏇 Лошадь и скачки</h3></div>', unsafe_allow_html=True)
    horse_age = st.number_input("Возраст лошади (лет)", min_value=2.0, max_value=12.0, value=7.0, step=0.5)
    horse_id = st.number_input("ID лошади", min_value=1, value=1736)
    track = st.selectbox("Ипподром", ["Sha Tin", "Happy Valley"])
    race_number = st.number_input("Номер заезда", min_value=1, value=10)
    distance = st.number_input("Дистанция (метры)", min_value=1000, max_value=2400, value=1400, step=100)
    surface = st.selectbox("Тип покрытия", ["Gress", "Dirt"])
    prize_money = st.number_input("Призовой фонд", min_value=660000, value=1310000, step=50000)
    race_type = st.selectbox("Тип заезда", ["Handicap", "Other"])

with col_right:
    st.markdown('<div class="section-header"><h3>👨‍🦱 Жокей и тренер</h3></div>', unsafe_allow_html=True)
    jockey_weight = st.number_input("Вес жокея (кг)", min_value=47, max_value=63, value=52)
    jockey_id = st.number_input("ID жокея", min_value=1, value=8656)
    country = st.selectbox("Страна жокея", ["Sverige", "Other"])
    trainer_name = st.text_input("Имя тренера", "CH Yip")
    trainer_id = st.number_input("ID тренера", min_value=1, value=6687)

    st.markdown('<div class="section-header"><h3>📊 Стартовые данные</h3></div>', unsafe_allow_html=True)
    starting_position = st.number_input("Стартовая позиция", min_value=1, max_value=14, value=6)
    odds = st.number_input("Коэффициент ставок (Odds)", min_value=0.1, value=22.0, step=0.5)

# Кнопка
col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
with col_btn2:
    predict_clicked = st.button("🔮 ПРЕДСКАЗАТЬ РЕЗУЛЬТАТ", type="primary", use_container_width=True)

if predict_clicked:
    payload = {
        "race_date": "2024-06-15",
        "track": track,
        "race_number": race_number,
        "distance": distance,
        "surface": surface,
        "prize_money": prize_money,
        "starting_position": starting_position,
        "jockey_weight": jockey_weight,
        "country": country,
        "trainer_name": trainer_name,
        "odds": odds,
        "race_type": race_type,
        "horse_id": horse_id,
        "jockey_id": jockey_id,
        "trainer_id": trainer_id,
        "horse_age": horse_age
    }

    try:
        response = requests.post(API_URL, json=payload, timeout=10)

        if response.status_code == 200:
            result = response.json()

            st.markdown("---")
            st.markdown("## 📊 Результат предсказания")

            col_r1, col_r2, col_r3 = st.columns(3)

            with col_r1:
                if result["prediction"] == 1:
                    st.markdown("""
                    <div class="result-card-top3">
                        <h2>🏆 ТОП-3</h2>
                        <p>Попадание в тройку</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div class="result-card-not-top3">
                        <h2>❌ НЕ ТОП-3</h2>
                        <p>Вне тройки лидеров</p>
                    </div>
                    """, unsafe_allow_html=True)

            with col_r2:
                prob_percent = result["probability"] * 100
                st.markdown(f"""
                <div class="probability-card">
                    <h2>{prob_percent:.1f}%</h2>
                    <p>Вероятность ТОП-3</p>
                </div>
                """, unsafe_allow_html=True)
                st.progress(result["probability"])

            with col_r3:
                confidence_text = {"high": "Высокая", "medium": "Средняя", "low": "Низкая"}
                st.markdown(f"""
                <div class="confidence-card">
                    <h2>{confidence_text[result['confidence']]}</h2>
                    <p>Достоверность</p>
                </div>
                """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="message-box">
                <p>📝 {result['message']}</p>
            </div>
            """, unsafe_allow_html=True)

        else:
            st.error(f"Ошибка API: {response.status_code}")

    except requests.exceptions.ConnectionError:
        st.error("❌ FastAPI не запущен! Запустите: python -m app.main")
    except Exception as e:
        st.error(f"Ошибка: {e}")

st.markdown("""
<footer>
    <p>🚀 Модель: Gradient Boosting | ROC-AUC: 0.78</p>
</footer>
""", unsafe_allow_html=True)