import streamlit as st
from sentiment_analysis import SentimentAnalyzer, DEMO_DATA

st.set_page_config(page_title="Sentiment AI", page_icon="🤖")
st.title("🤖 Sentiment Analysis AI")
st.write("Type any sentence below, and the AI will guess if it is POSITIVE or NEGATIVE.")

@st.cache_resource
def load_model():
    texts = [t for t, _ in DEMO_DATA]
    labels = [l for _, l in DEMO_DATA]
    model = SentimentAnalyzer().fit(texts, labels)
    return model

model = load_model()
user_input = st.text_input("Enter your text here:")

if st.button("Analyze Sentiment"):
    if user_input:
        result = model.predict_one(user_input)
        sentiment = result['sentiment'].upper()
        confidence = result['confidence'] * 100
        
        if sentiment == "POSITIVE":
            st.success(f"**Result: {sentiment}** (Confidence: {confidence:.2f}%)")
        else:
            st.error(f"**Result: {sentiment}** (Confidence: {confidence:.2f}%)")
        st.write(f"**Cleaned Text:** {result['cleaned']}")
    else:
        st.warning("Please type something first!")