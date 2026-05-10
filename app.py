import streamlit as st
import pickle
import numpy as np

# ------------------ Page Configuration ------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ------------------ Custom CSS Styling ------------------
st.markdown("""
<style>
body {
    background-color: #f4f6f9;
}
.title {
    text-align: center;
    color: #2c3e50;
}
.result-real {
    color: green;
    font-size: 22px;
    font-weight: bold;
}
.result-fake {
    color: red;
    font-size: 22px;
    font-weight: bold;
}
.confidence {
    color: #34495e;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model & Vectorizer ------------------
@st.cache_resource
def load_model():
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("tfidf_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

# ------------------ App Title ------------------
st.markdown("<h1 class='title'>📰 Fake News Detection System</h1>", unsafe_allow_html=True)
st.write("### Enter a news article to check whether it is **Fake** or **Real**")

# ------------------ Text Input ------------------
news_text = st.text_area(
    "📝 News Content:",
    height=200,
    placeholder="Paste news article text here..."
)

# ------------------ Prediction Button ------------------
if st.button("🔍 Predict"):
    if news_text.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        # Vectorize input
        transformed_text = vectorizer.transform([news_text])

        # Prediction
        prediction = model.predict(transformed_text)[0]
        probabilities = model.predict_proba(transformed_text)[0]

        confidence = np.max(probabilities) * 100

        # ------------------ Display Result ------------------
        if prediction == 1:
            st.markdown("<p class='result-real'>✅ Result: REAL News</p>", unsafe_allow_html=True)
        else:
            st.markdown("<p class='result-fake'>❌ Result: FAKE News</p>", unsafe_allow_html=True)

        st.markdown(
            f"<p class='confidence'>📊 Prediction Confidence: <b>{confidence:.2f}%</b></p>",
            unsafe_allow_html=True
        )

# ------------------ Footer ------------------
st.markdown("---")
st.markdown(
    "<center>Developed for NLP Final Project | Fake News Detection</center>",
    unsafe_allow_html=True
)
