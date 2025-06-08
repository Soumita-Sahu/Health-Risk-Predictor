import streamlit as st
import numpy as np
import joblib

# Load models
model_lung = joblib.load('final_random_forest_model.pkl')
model_heart = joblib.load('final_logistic_regression_model.pkl')

# Page config
st.set_page_config(
    page_title="💖 Health Risk Predictor",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="❤️"
)

# --- Dark Mode Toggle ---
dark_mode = st.sidebar.checkbox("🌙 Enable Dark Mode")
if dark_mode:
    st.markdown(
        """
        <style>
            body { background-color: #121212; color: #e0e0e0; }
            .stButton>button { background-color: #333; color: #fff; }
            .css-1d391kg { background-color: #222 !important; }
            .stTextInput>div>input { background-color: #333; color: #fff; }
        </style>
        """,
        unsafe_allow_html=True
    )

# --- Custom CSS ---
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto+Slab&display=swap');
body { font-family: 'Roboto Slab', serif; }
h1, h2, h3 { color: #d6336c; letter-spacing: 0.05em; }
.section-heading { font-size: 1.8rem; color: #d6336c; margin-top: 40px; margin-bottom: 20px; }
.footer-section { text-align: center; padding: 20px 10px; color: #999; font-size: 0.9rem; }
.footer-links a { color: #999; text-decoration: none; margin: 0 15px; }
.footer-links a:hover { color: #d6336c; }
.chat-box { border: 1px solid #d6336c; border-radius: 8px; padding: 10px; max-height: 250px; overflow-y: auto; background-color: #fff0f5; }
.chat-message-user { color: #d6336c; font-weight: bold; }
.chat-message-bot { color: #6a1b4d; font-style: italic; }
.header-title { font-size: 3rem; font-weight: 700; color: #d6336c; text-align: center; margin-bottom: 5px; }
.header-subtitle { font-size: 1.2rem; text-align: center; color: #555; margin-bottom: 30px; }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: About Us ---
st.sidebar.header("ℹ️ About Us")
st.sidebar.markdown("""
We are final-year students from the **Computer Science and Engineering Department** of **Techno Main Salt Lake**.

### 👥 Team Members:
- **Soumita Sahu** (13000121050)
- **Arghyadeep Mondal** (13000121130)
- **Souvik Mondal** (13000121132)
- **Soumyadeep Das** (13000121136)

**Mission:** Empower individuals with simple tools for early health risk detection.

**Disclaimer:** This tool offers predictions based on statistical models. It is **not** a substitute for medical advice. Always consult a healthcare provider for professional diagnosis.
""")

# --- HEADER ---
st.markdown('<div class="header-title">💖 Health Risk Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Early detection made simple and accessible for everyone</div>', unsafe_allow_html=True)

# Initialize chat history session state if not present (once outside tabs)
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# --- Tabs ---
tab_names = [
    "💉 Lung Cancer Prediction",
    "💔 Heart Disease Prediction",
    "🩺 Health Tips",
    "❓ FAQ",
    "📬 Contact & Feedback",
    "🤖 Health Assistant Chat Bot"
]
tabs = st.tabs(tab_names)

# -------- Tab 1: Lung Cancer Prediction --------
with tabs[0]:
    st.subheader("🌍 Lung Cancer Prediction")
    age = st.slider("Age", 18, 100, 30)
    smoking = st.radio("Do you smoke?", ("Yes", "No"))
    yellow_fingers = st.radio("Do you have yellow fingers?", ("Yes", "No"))
    anxiety = st.radio("Do you suffer from anxiety?", ("Yes", "No"))
    peer_pressure = st.radio("Are you under peer pressure?", ("Yes", "No"))
    chronic_disease = st.radio("Do you have any chronic disease?", ("Yes", "No"))
    fatigue = st.radio("Do you feel fatigued?", ("Yes", "No"))
    allergy = st.radio("Do you have allergies?", ("Yes", "No"))
    wheezing = st.radio("Do you experience wheezing?", ("Yes", "No"))
    alcohol = st.radio("Do you consume alcohol?", ("Yes", "No"))
    coughing = st.radio("Do you have a persistent cough?", ("Yes", "No"))
    shortness = st.radio("Do you feel shortness of breath?", ("Yes", "No"))
    swallowing = st.radio("Do you face difficulty in swallowing?", ("Yes", "No"))
    chest_pain = st.radio("Do you experience chest pain?", ("Yes", "No"))
    weight_loss = st.radio("Have you experienced unexplained weight loss?", ("Yes", "No"))

    if st.button("Predict Lung Cancer Risk"):
        input_features = np.array([
            age,
            1 if smoking == "Yes" else 0,
            1 if yellow_fingers == "Yes" else 0,
            1 if anxiety == "Yes" else 0,
            1 if peer_pressure == "Yes" else 0,
            1 if chronic_disease == "Yes" else 0,
            1 if fatigue == "Yes" else 0,
            1 if allergy == "Yes" else 0,
            1 if wheezing == "Yes" else 0,
            1 if alcohol == "Yes" else 0,
            1 if coughing == "Yes" else 0,
            1 if shortness == "Yes" else 0,
            1 if swallowing == "Yes" else 0,
            1 if chest_pain == "Yes" else 0,
            1 if weight_loss == "Yes" else 0
        ]).reshape(1, -1)

        result = model_lung.predict(input_features)
        if result[0] == 1:
            st.error("Warning: High risk of Lung Cancer. Please consult a doctor immediately.")
        else:
            st.success("Low risk of Lung Cancer. Keep monitoring your health.")

# -------- Tab 2: Heart Disease Prediction --------
with tabs[1]:
    st.subheader("💔 Heart Disease Prediction")
    age = st.slider("Age", 20, 100, 45)
    sex = st.selectbox("Gender", ("Male", "Female"))
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mm Hg)", 90, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dl)", 100, 400, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ("Yes", "No"))
    restecg = st.selectbox("Resting ECG Result", ["Normal", "Having ST-T wave abnormality", "Showing probable/definite LV hypertrophy"])
    thalach = st.number_input("Maximum Heart Rate Achieved", 60, 220, 150)
    exang = st.radio("Exercise Induced Angina", ("Yes", "No"))
    oldpeak = st.slider("Oldpeak", 0.0, 6.0, 1.0)
    slope = st.selectbox("Slope of Peak Exercise ST Segment", ["Upsloping", "Flat", "Downsloping"])
    ca = st.slider("Number of Major Vessels Colored by Fluoroscopy", 0, 4, 0)
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

    if st.button("Predict Heart Disease Risk"):
        sex_val = 1 if sex == "Male" else 0
        cp_val = ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"].index(cp)
        fbs_val = 1 if fbs == "Yes" else 0
        restecg_val = ["Normal", "Having ST-T wave abnormality", "Showing probable/definite LV hypertrophy"].index(restecg)
        exang_val = 1 if exang == "Yes" else 0
        slope_val = ["Upsloping", "Flat", "Downsloping"].index(slope)
        thal_val = ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)

        input_data = np.array([
            age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
            thalach, exang_val, oldpeak, slope_val, ca, thal_val
        ]).reshape(1, -1)

        result = model_heart.predict(input_data)
        if result[0] == 1:
            st.error("Warning: High risk of Heart Disease. Please consult a doctor immediately.")
        else:
            st.success("Low risk of Heart Disease. Maintain a healthy lifestyle.")

# -------- Tab 3: Health Tips --------
with tabs[2]:
    st.markdown('<div class="section-heading">🩺 Health Tips</div>', unsafe_allow_html=True)
    st.markdown("""
    - 🚭 **Avoid Smoking**: Major risk factor for lung and heart diseases.
    - 🥗 **Eat Balanced Diet**: Rich in fiber, low in saturated fats.
    - 🏃‍♀️ **Exercise Regularly**: At least 30 minutes most days.
    - 🛌 **Sleep Well**: Aim for 7–8 hours of restful sleep.
    - 😌 **Manage Stress**: Try meditation or yoga.
    """)

# -------- Tab 4: FAQ --------
with tabs[3]:
    st.markdown('<div class="section-heading">❓ Frequently Asked Questions</div>', unsafe_allow_html=True)
    st.markdown("""
    **Q1:** Is this tool accurate?  
    > This tool uses trained ML models, but it's not a replacement for professional medical advice.

    **Q2:** Can I use this report for medical treatment?  
    > No. It is for awareness and should be shown to a doctor for further advice.

    **Q3:** How often should I check?  
    > Once every 6–12 months or as recommended by a doctor.
    """)

# -------- Tab 5: Contact & Feedback --------
with tabs[4]:
    st.markdown('<div class="section-heading">📬 Contact & Feedback</div>', unsafe_allow_html=True)
    contact_name = st.text_input("Your Name")
    contact_email = st.text_input("Your Email")
    contact_msg = st.text_area("Your Feedback or Message")
    if st.button("📨 Submit Feedback"):
        if contact_name and contact_email and contact_msg:
            st.success("Thank you for your feedback! We'll get back to you soon.")
        else:
            st.error("Please fill out all fields before submitting.")

# -------- Tab 6: Health Assistant Chat Bot --------
with tabs[5]:
    st.markdown('<div class="section-heading">🤖 Health Assistant Chat Bot</div>', unsafe_allow_html=True)

    # Fix chat_history entries if necessary
    fixed_history = []
    for entry in st.session_state.chat_history:
        if isinstance(entry, dict):
            if "user" in entry:
                fixed_history.append(("user", entry["user"]))
            elif "bot" in entry:
                fixed_history.append(("bot", entry["bot"]))
        else:
            fixed_history.append(entry)
    st.session_state.chat_history = fixed_history

    # Display chat history
    chat_box = st.container()
    with chat_box:
        for sender, message in st.session_state.chat_history:
            if sender == "user":
                st.markdown(f'<p class="chat-message-user">You: {message}</p>', unsafe_allow_html=True)
            else:
                st.markdown(f'<p class="chat-message-bot">Bot: {message}</p>', unsafe_allow_html=True)

    # Chat input form
    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_input("Type your message here:", key="chat_input")
        submitted = st.form_submit_button("Send")

    if submitted and user_input:
        st.session_state.chat_history.append(("user", user_input))

        msg_lower = user_input.lower()
        if "lung" in msg_lower:
            bot_reply = "For lung health, avoid smoking and get regular checkups."
        elif "heart" in msg_lower:
            bot_reply = "Maintain a balanced diet and exercise regularly for a healthy heart."
        elif "hello" in msg_lower or "hi" in msg_lower:
            bot_reply = "Hello! How can I assist you with your health concerns today?"
        else:
            bot_reply = "I'm here to help! Please ask about lung or heart health."

        st.session_state.chat_history.append(("bot", bot_reply))

# --- Footer ---
st.markdown("""
<div class="footer-section">
    <div class="footer-links">
        <a href="#">About</a> | <a href="#">Contact</a> | <a href="#">Privacy Policy</a>
    </div>
    <p>© 2025 Health Risk Predictor Team. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)
