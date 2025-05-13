import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Embedding, SimpleRNN, Dense
from tensorflow.keras.preprocessing import sequence
from sklearn.model_selection import train_test_split

# Load the IMDB dataset
word_index = imdb.get_word_index()
reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

model = load_model('imdb_rnn_model_1.h5')

def decode_review(encoded_review):
    return ' '.join(
        [reverse_word_index.get(i - 3, '?') for i in encoded_review]
    )
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word, 2) + 3 for word in words]
    return sequence.pad_sequences([encoded_review], maxlen=500)
def process_review(text):
    encoded_review = preprocess_text(text)
    prediction = model.predict(encoded_review)
    sentiment = 'positive' if prediction[0][0] > 0.5 else 'negative'
    return sentiment, prediction[0][0]

st.title("Movie Review Sentiment Analysis")
st.write("Enter a movie review to analyze its sentiment.")

user_input = st.text_input("Enter a movie review:")
if st.button("Analayze"):
    if user_input:
        sentiment, score = process_review(user_input)
        st.write(f"Sentiment: {sentiment}")
        st.write(f"Score: {score:.4f}")
    else:
        st.write("Please enter a review.")