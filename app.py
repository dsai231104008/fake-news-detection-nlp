import streamlit as st
import pickle
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ------------------ Page Config ------------------
st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered"
)

# ------------------ Custom CSS ------------------
st.markdown("""
<style>
body {
    background: linear-gradient(135deg, #1e3c72, #2a5298);
}
.main-title {
    font-size: 44px;
    font-weight: bold;
    color: #ffffff;
    text-align: center;
}
.sub-title {
    font-size: 18px;
    color: #e0e0e0;
    text-align: center;
}
.card {
    background: rgba(255,255,255,0.15);
    backdrop-filter: blur(12px);
    padding: 30px;
    border-radius: 20px;
}
.result-real {
    color: #00ff9c;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}
.result-fake {
    color: #ff4b4b;
    font-size: 28px;
    font-weight: bold;
    text-align: center;
}
.footer {
    text-align: center;
    color: #cccccc;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ------------------ Load Model ------------------
model = pickle.load(open("fake_news_model.pkl", "rb"))
vectorizer = pickle.load(open("tfidf_vectorizer.pkl", "rb"))

# ------------------ Clean Text ------------------
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text

# ------------------ Header ------------------
st.markdown('<div class="main-title">📰 Fake News Detection System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-title">NLP-based Fake & Real News Classification</div><br>', unsafe_allow_html=True)

# ------------------ Input Card ------------------
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)
    news_text = st.text_area(
        "📝 Enter News Article:",
        height=200,
        placeholder="Paste news content here..."
    )
    col1, col2 = st.columns(2)
    with col1:
        predict_btn = st.button("🔍 Analyze News")
    with col2:
        reset_btn = st.button("🔄 Reset")
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ Reset (FIXED) ------------------
if reset_btn:
    st.rerun()   # ✅ correct function in latest Streamlit

# ------------------ Prediction ------------------
if predict_btn:
    if news_text.strip() == "":
        st.warning("⚠️ Please enter news text!")
    else:
        cleaned = clean_text(news_text)
        vectorized = vectorizer.transform([cleaned])
        prediction = model.predict(vectorized)
        probability = model.predict_proba(vectorized)[0]

        st.markdown("---")

        if prediction[0] == 1:
            st.markdown('<div class="result-fake">🚨 FAKE NEWS DETECTED</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="result-real">✅ REAL NEWS DETECTED</div>', unsafe_allow_html=True)

        confidence = max(probability) * 100
        st.markdown(f"### 📊 Model Confidence: **{confidence:.2f}%**")
        st.progress(int(confidence))

        # ------------------ Probability Bar Chart ------------------
        st.markdown("### 📈 Prediction Probability Chart")
        fig, ax = plt.subplots()
        ax.bar(["Real News", "Fake News"], probability)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1)
        st.pyplot(fig)

        # ------------------ Word Cloud ------------------
        st.markdown("### ☁️ Word Cloud of Entered News")
        wc = WordCloud(
            width=800,
            height=400,
            background_color="white"
        ).generate(cleaned)

        fig2, ax2 = plt.subplots()
        ax2.imshow(wc, interpolation="bilinear")
        ax2.axis("off")
        st.pyplot(fig2)

# ------------------ Footer ------------------
st.markdown("<br><div class='footer'>NLP Final Project | Fake News Detection</div>", unsafe_allow_html=True)