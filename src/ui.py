import streamlit as st
import requests
import os

# 1. Page Config
st.set_page_config(page_title="Sentiment Analyzer", page_icon="🎬")

st.title("🎬 Movie Review Sentiment Analyzer")
st.markdown("Enter a movie review below to see if it's **Positive** or **Negative**.")

# 2. Input Area
user_input = st.text_area("Review Text:", placeholder="The cinematography was brilliant, but the plot was thin...")

# 3. API URL (Use your Railway URL here!)
# For local testing, use "http://localhost:8000/predict"
API_URL = os.getenv("API_URL", "http://localhost:8000/predict")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text first!")
    else:
        with st.spinner("Analyzing..."):
            try:
                # Send request to our FastAPI backend
                response = requests.post(API_URL, json={"text": user_input})
                response.raise_for_status()
                data = response.json()

                # 4. Display Results
                sentiment = data["sentiment"]
                prob = data["probability"]

                if sentiment == "positive":
                    st.success(f"**Positive** (Confidence: {prob:.2%})")
                    st.balloons()
                else:
                    st.error(f"**Negative** (Confidence: {prob:.2%})")
            
            except Exception as e:
                st.error(f"Connection Error: {e}")