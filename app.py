import streamlit as st
import pickle
import string
import nltk
import pandas as pd
import joblib  # For loading .sav files
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer


nltk.download('punkt')
nltk.download("stopwords")

# Load the trained RandomForest model
model_path = "randomforrest_spam_model.sav"
clf = joblib.load(model_path)

# Load dataset to fit vectorizer again
df = pd.read_csv("spam_ham_dataset.csv")

# Text Preprocessing
stemmer = PorterStemmer()
stopwords_set = set(stopwords.words("english"))

def preprocess_text(text):
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    text = " ".join([stemmer.stem(word) for word in text.split() if word not in stopwords_set])
    return text

df["processed_text"] = df["text"].apply(preprocess_text)

# Fit vectorizer again (since we didn't save it)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df["processed_text"])

# Streamlit UI
st.title("Spam Detection App")
st.write("Enter a message to check if it's Spam or Ham.")

user_input = st.text_area("Enter a message:", "")

if user_input:
    processed_input = preprocess_text(user_input)
    vectorized_input = vectorizer.transform([processed_input])
    prediction = clf.predict(vectorized_input)[0]
    st.write("Prediction:", "Spam" if prediction == 1 else "Ham")
