import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('student_model.pkl', 'rb'))

# App title
st.title("🎓 Student Score Prediction App")
st.write("Predict student exam scores based on study habits and attendance.")

# Input fields
hours = st.number_input("📘 Hours Studied", min_value=0, max_value=15, value=5)
attendance = st.number_input("📅 Attendance (%)", min_value=0, max_value=100, value=85)
assignments = st.number_input("📝 Assignments Submitted", min_value=0, max_value=10, value=8)

# Prediction button
if st.button("🔮 Predict Score"):
    features = np.array([[hours, attendance, assignments]])
    prediction = model.predict(features)
    st.success(f"Predicted Score: **{prediction[0]:.2f}** / 100")

# Footer
st.caption("Built with ❤️ using Streamlit and Machine Learning")
