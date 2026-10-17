import streamlit as st
import joblib
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# PAGE CONFIGURATION
# -------------------------------
st.set_page_config(
    page_title="AI Spam Detector",
    page_icon="🛡️",
    layout="wide"
)

# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------
# -------------------------------
# LOAD TRAINED MODEL
# -------------------------------

MODEL_PATH = "models/spam_model.pkl"

@st.cache_resource(show_spinner=False)
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    import time

    start = time.time()
    loaded_model = joblib.load(MODEL_PATH)

    print(
        f"Model loaded in {time.time() - start:.2f} seconds"
    )

    return loaded_model


model = load_model()

# -------------------------------
# CUSTOM CSS
# -------------------------------
st.markdown("""
<style>
    .main {
        background-color: #f5f7fb;
    }

    .hero {
        padding: 28px;
        border-radius: 18px;
        background: linear-gradient(120deg, #102a43, #2563eb);
        color: white;
        margin-bottom: 24px;
    }

    .hero h1 {
        color: white;
        font-size: 36px;
    }

    .hero p {
        color: #e2e8f0;
        font-size: 17px;
    }

    .result-box {
        padding: 22px;
        border-radius: 14px;
        margin-top: 18px;
        background: #ffffff;
        border: 1px solid #dbe3ef;
    }

    .safe {
        color: #15803d;
        font-size: 27px;
        font-weight: bold;
    }

    .danger {
        color: #dc2626;
        font-size: 27px;
        font-weight: bold;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 25px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------
# HEADER
# -------------------------------
st.markdown("""
<div class="hero">
    <h1>🛡️ AI-Powered SMS Spam Detector</h1>
    <p>
        Intelligent message classification using Natural Language
        Processing and Machine Learning.
    </p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# CHECK MODEL
# -------------------------------
if model is None:
    st.error(
        "Trained model not found. "
        "Please run: python train_model.py"
    )
    st.stop()

# -------------------------------
# SIDEBAR
# -------------------------------
with st.sidebar:
    st.title("⚙️ About Project")

    st.write(
        "This application classifies SMS messages "
        "as Spam or Normal using a trained ML model."
    )

    st.divider()

    st.subheader("Technologies")
    st.write("🐍 Python")
    st.write("🧠 Scikit-learn")
    st.write("📝 TF-IDF")
    st.write("🌐 Streamlit")

    st.divider()

    st.caption(
        "Educational project. Predictions may be incorrect."
    )

# -------------------------------
# SESSION STATE
# -------------------------------
if "message_input" not in st.session_state:
    st.session_state.message_input = ""

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

if "spam_score" not in st.session_state:
    st.session_state.spam_score = None

if "selected_sample" not in st.session_state:
    st.session_state.selected_sample = "Choose a sample..."

# -------------------------------
# MAIN LAYOUT
# -------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("📩 Analyze Your Message")

    sample_options = [
        "Choose a sample...",
        "Normal: Hey, are we meeting at college tomorrow?",
        "Spam: Congratulations! You have won a free prize. Call now!",
        "Spam: WIN CASH NOW! Claim your reward by clicking here!",
        "Normal: Please send me the class notes."
    ]

    sample_messages = {
        sample_options[1]: "Hey, are we meeting at college tomorrow?",
        sample_options[2]: "Congratulations! You have won a free prize. Call now!",
        sample_options[3]: "WIN CASH NOW! Claim your reward by clicking here!",
        sample_options[4]: "Please send me the class notes."
    }

    def clear_dashboard():
        st.session_state.message_input = ""
        st.session_state.prediction_result = None
        st.session_state.spam_score = None
        st.session_state.selected_sample = "Choose a sample..."

    st.selectbox(
        "Try an example",
        sample_options,
        key="selected_sample",
        on_change=lambda: st.session_state.update(
            message_input=sample_messages.get(
                st.session_state.selected_sample, ""
            ),
            prediction_result=None,
            spam_score=None
        )
    )

    st.text_area(
        "Enter SMS message",
        key="message_input",
        height=150,
        placeholder="Paste your SMS message here..."
    )

    col1, col2 = st.columns(2)

    with col1:
        analyze = st.button(
            "🔍 Analyze Message",
            type="primary",
            use_container_width=True
        )

    with col2:
        st.button(
            "🔄 Reset / Clear",
            on_click=clear_dashboard,
            use_container_width=True
        )

with right:
    st.subheader("📊 How It Works")

    st.info(
        "1. Enter an SMS message.\n\n"
        "2. TF-IDF converts text into numerical features.\n\n"
        "3. The trained Naive Bayes model classifies it.\n\n"
        "4. The application displays the prediction."
    )

    st.warning(
        "Do not rely on this result alone to determine "
        "whether a message is safe."
    )

# -------------------------------
# PREDICTION
# -------------------------------
if analyze:
    message = st.session_state.message_input.strip()

    if not message:
        st.session_state.prediction_result = None
        st.session_state.spam_score = None
        st.warning("Please enter a message first.")

    else:
        prediction = int(model.predict([message])[0])

        st.session_state.prediction_result = prediction

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba([message])[0]
            st.session_state.spam_score = float(probabilities[1])
        else:
            st.session_state.spam_score = None

# -------------------------------
# DISPLAY STORED RESULT
# -------------------------------
if st.session_state.prediction_result is not None:
    prediction = st.session_state.prediction_result

    st.divider()
    st.subheader("🎯 Prediction Result")

    if prediction == 1:
        st.error("🚨 SPAM MESSAGE")
        st.write("This message has been classified as spam.")
    else:
        st.success("✅ NORMAL MESSAGE")
        st.write("This message has been classified as normal.")

    if st.session_state.spam_score is not None:
        score = st.session_state.spam_score

        st.write("### Model Spam Score")
        st.progress(score)
        st.write(f"Spam-class probability: {score:.1%}")

        st.caption(
            "This is the model's estimated class probability, "
            "not a guarantee that the message is spam."
        )


            # -------------------------------
# MODEL EVALUATION DASHBOARD
# -------------------------------
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.divider()
st.header("📊 Model Evaluation Dashboard")

METRICS_PATH = "models/metrics.pkl"

if os.path.exists(METRICS_PATH):
    metrics = joblib.load(METRICS_PATH)

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
    c2.metric("Precision", f"{metrics['precision']:.2%}")
    c3.metric("Recall", f"{metrics['recall']:.2%}")
    c4.metric("F1-Score", f"{metrics['f1_score']:.2%}")

    st.info(
        "Metrics are calculated on the held-out test set. "
        "Results may vary with the dataset and train-test split."
    )
else:
    st.warning(
        "Evaluation metrics not found. "
        "Run python train_model.py first."
    )
# -------------------------------
# CONFUSION MATRIX VISUALIZATION
# -------------------------------

st.subheader("🔍 Confusion Matrix")

CM_PATH = "models/confusion_matrix.pkl"

if os.path.exists(CM_PATH):
    cm = joblib.load(CM_PATH)

    fig, ax = plt.subplots(figsize=(6, 4))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Spam"],
        yticklabels=["Normal", "Spam"],
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("SMS Spam Classification")

    st.pyplot(fig)
    plt.close(fig)

else:
    st.warning(
        "Confusion matrix not found. "
        "Run python train_model.py first."
    )
# -------------------------------
# FOOTER
# -------------------------------
st.markdown("""
<div class="footer">
    AI Spam Detector | NLP + Machine Learning |
    Built with Python and Streamlit
</div>
""", unsafe_allow_html=True)







# python -m streamlit run app.py