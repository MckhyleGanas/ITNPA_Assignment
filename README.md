# ITNPA_Assignment
Natural Language Processing with Python
Research Abstract Classifier Web Application.
Within this README file, I will be detailing the Streamlit application as well as the preprocessing techniques that is used in the interface. This repository contains a Streamlit-based Python web application - app.py - which is designed to classify unseen research abstracts using a machine learning model. The application is developed inside a Jupyter Notebook environment and deployed locally via the Jupyter Terminal.

Project Overview:
The best performing classifier from the model development phase conducted in Jupyter file "Question 1&2" is deployed here to provide a clean, web-based interface. The system classifies submitted research abstract into one of two categories:
-Artificial Intelligence
-Classical Research

How the model pipeline works:
1. User input: A user enters or pastes a research abstract into the application's text box and clicks "Classify Abstract".
2. Text denoising: The submitted text is automatically processed using the same denoising preprocessing applied during the model training.
3. Feature Transformation: The application reuses a saved, already-fitted TF-IDF vectorizer stored as Python object (tfidf_vectorizer.pkl). Reusing this fitted vectorizer ensures new user input is represented using the exact same vocabulary and feature space learned during model training.
4. Classification: The transformed features are passed into the trained classifier (also stored as a .pkl file "sgd_model.pkl) to generate and display the final prediction.

How to run the application:
Option A: Using the Jupyter Terminal
Due to development taken place within Jupyter Notebook, you can use the built in Jupyter Terminal to launch the Streamlit server.
1. Open the Jupyter Terminal
   -Go to your Jupyter Notebook dashboard
   -Click New in the top-right corner
   -Select Terminal from the drop down menu
2. Navigate to your Project Directory
   -cd path\to\your\project (In the terminal, change to the folder containing your application files
3. Run the Streamlit Application
   -streamlit run app.py (Execute the following command to start the web interface)
4. Access the interface
   -Once launched, Streamlit usually provides two URL's in the terminal:
   ~Local Url: Usually http:
   ~Network URL: This is used for accessing the application from another device on the same local network

Option B: Using system terminals (Alternative for non-Jupyter Notebook users)
If a user wants to test this application and does not have Jupyter Notebook running, they can use their computers' standard terminal application such as PowerShell, Command Prompt on Windows or Terminal on macOS/Linux)
1. Open PowerShell (Windows) or your operating systems native Terminal.
2. Activate the specific Python environment where your dependencies are installed (if using a virtual environment or Conda):
   -Example for Conda: conda activate your_env_name
3. Navigate to the project repository:
   -cd "C:\path\to\your\project"
4. Run the Streamlit application
   -streamlit run app.py

Prerequisites
Before running the application, make sure you have installed the required libraries in your environment:
-pip install streamlit scikit-learn

Once the server is running, simply type in or paste your abstracts into the web interface to instantly check its classification.
