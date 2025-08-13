import streamlit as st
import pandas as pd
import joblib

# Load model and label encoders
model = joblib.load("career_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Attendance mapping
attendance_map = {'Never': 0, 'Sometimes': 1, 'Always': 2}

# Helper to normalize inputs
def normalize_input(value, col):
    value = value.strip()
    if col in label_encoders:
        for cls in label_encoders[col].classes_:
            if value.lower() == cls.lower():
                return cls
        return label_encoders[col].classes_[0]
    return value

# Streamlit App
st.title("🎓 Career Suggestion Predictor")

# User inputs
student_age = st.number_input("Enter Student Age", min_value=5, max_value=25, value=15)
sex = st.selectbox("Enter Sex", ["Male", "Female"])
high_school_type = st.selectbox("High School Type", ["Private", "State", "Other"])
scholarship = st.text_input("Scholarship (e.g., 50%)", "50")
additional_work = st.selectbox("Additional Work?", ["Yes", "No"])
sports_activity = st.selectbox("Sports Activity?", ["Yes", "No"])
transportation = st.selectbox("Transportation", ["Bus", "Walk", "Car", "Other"])
weekly_study_hours = st.number_input("Weekly Study Hours", min_value=0, max_value=100, value=3)
attendance = st.selectbox("Attendance", ["Never", "Sometimes", "Always"])
reading = st.selectbox("Reading?", ["Yes", "No"])
notes = st.selectbox("Notes?", ["Yes", "No"])
listening_in_class = st.selectbox("Listening in Class?", ["Yes", "No"])
project_work = st.selectbox("Project Work?", ["Yes", "No"])

# Predict button
if st.button("Predict Career Suggestion"):
    # Convert Scholarship
    scholarship_int = int(scholarship.replace("%", "").strip())
    
    # Prepare DataFrame
    input_df = pd.DataFrame([[
        student_age, sex, high_school_type, scholarship_int,
        additional_work, sports_activity, transportation,
        weekly_study_hours, attendance, reading, notes,
        listening_in_class, project_work
    ]], columns=[
        'Student_Age', 'Sex', 'High_School_Type', 'Scholarship',
        'Additional_Work', 'Sports_activity', 'Transportation',
        'Weekly_Study_Hours', 'Attendance',
        'Reading', 'Notes', 'Listening_in_Class', 'Project_work'
    ])
    
    # Normalize inputs
    for col in label_encoders.keys():
        input_df[col] = input_df[col].apply(lambda x: normalize_input(x, col))
        input_df[col] = label_encoders[col].transform(input_df[col])
    
    # Map Attendance
    input_df['Attendance'] = attendance_map.get(attendance, 0)
    
    # Prediction
    prediction = model.predict(input_df)[0]
    
    # Display result
    if prediction == 1:
        st.success("🎯 Career Suggestion: HIGH performance potential")
    else:
        st.warning("📘 Career Suggestion: LOW performance — focus on skill-building")
