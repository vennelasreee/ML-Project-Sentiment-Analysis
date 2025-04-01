import streamlit as st
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model, vectorizer, and label encoder
best_model = joblib.load('best_model.pkl')  # Ensure this is the model with best performance
vectorizer = joblib.load('vectorizer.pkl')
label_encoder = joblib.load('label_encoder.pkl')

# Title of the web app
st.title("Sentiment Analysis App")

# Description
st.markdown("This is a simple app for sentiment analysis. Enter a text, and click **Predict** to see the result.")

# User input for sentiment analysis
user_input = st.text_area("Enter your text:", "")

# Add a button for prediction
if st.button("Predict"):
    if user_input.strip():  # Check if the user has entered valid text
        # Preprocess and vectorize the input text
        user_input_tfidf = vectorizer.transform([user_input])

        # Predict sentiment
        prediction = best_model.predict(user_input_tfidf)
        predicted_sentiment = label_encoder.inverse_transform(prediction)

        # Display the prediction result
        if predicted_sentiment[0] == 0:
            st.write(f"**Predicted Sentiment for the input text**: Negative")
        else:
            st.write(f"**Predicted Sentiment for the input text**: Positive")
    else:
        st.warning("Please enter some text to analyze.")
