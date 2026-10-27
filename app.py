import streamlit as st
import joblib
import re

#----------------------------------------------
#Loading the trained NLP and TF-IDF vectorizer
#----------------------------------------------

model = joblib.load("sgd_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

#----------------------------------------------
#Text denoising function - matched during pre-
#processing used during model training
#----------------------------------------------

def denoise(text):

    #Converting to Lowercase
    text = text.lower()

    #Removing any URL'S
    text = re.sub(r'http\S+|www\S+', '', text)

    #Removing email addresses
    text = re.sub(r'\S+@\S+', '', text)

    #Removing HTML tags
    text = re.sub(r'<.*?>', '', text)

    #Removing remaing special text
    #while keeping letters, numbers, and spaces
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)

    #Remving any spaces
    text = re.sub(r'\s+', ' ', text)

    #Removing any white space from start to end
    text = text.strip()

    return text

#-----------------------------------------------
#Web Interface
#-----------------------------------------------

st.title("Researchly Research Abstract Classifier")

st.write(
    "This app classifies research as either AI-related or classical research."
)

#-----------------------------------------------
#User text input
#-----------------------------------------------

abstract = st.text_area(
    "Enter research abstract:",
    height=250,
    placeholder="Type in or copy and paste a research abstract here..."
)

#-----------------------------------------------
#Classidy button
#-----------------------------------------------

if st.button("Classify Abstract"):

    #Verify user input
    if abstract.strip() == "":
        st.warning("Please enter a research abstract for classifying.")

    else:
        #Apply thr same denoising preprocessing
        cleaned_abstract = denoise(abstract)

        #Convert the cleaned text into TF-IDF features
        abstract_tfidf = tfidf.transform([cleaned_abstract])

        #Making the prediction
        prediction = model.predict(abstract_tfidf)[0]

#-----------------------------------------------
#Displaying the presiction
#-----------------------------------------------

        st.subheader("Prediction:")

        if prediction == "ai":
            st.success("This abstract is classified as: Artificial Intelligence")

        else:
            st.info(
                "The abstract is classified as: Classical Research"
            )