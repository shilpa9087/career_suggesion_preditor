# 🎓 Career Suggestion Predictor

A **machine learning-based tool** that predicts students’ career performance potential (HIGH or LOW) based on academic, behavioral, and demographic data. Built using **Random Forest**, **Label Encoding**, and deployed with **Streamlit** for an interactive web interface.

---

## 🧠 Objective

To provide personalized insights on students’ career readiness and guide them on areas to focus on for skill-building and improvement.

---

## 📂 Project Structure

Project1/
│
├── career_model.pkl # Trained Random Forest model
├── label_encoders.pkl # Encoders for categorical variables
├── student_career.py # Python script for console prediction
├── career_app.py # Streamlit web app
└── student.csv # Sample dataset 


---

## ⚙️ Technologies & Libraries Used

- Python 3.x  
- Pandas  
- Scikit-learn (Random Forest, LabelEncoder)  
- Joblib (model serialization)  
- Streamlit (web interface)  

---

## 📝 Features

- Takes inputs like age, sex, high school type, scholarship, extracurriculars, study hours, attendance, and project work.  
- Normalizes and encodes user inputs automatically.  
- Provides **HIGH/LOW career suggestion** with clear output.  
- Interactive Streamlit app for user-friendly experience.  

---

## 🚀 How to Run

### Console Version

```bash
python student_career.py

### Streamlit Web App

```bash
streamlit run career_app.py 

## Dataset
student.csv contains features like:

Student_Age, Sex, High_School_Type, Scholarship

Additional_Work, Sports_activity, Transportation

Weekly_Study_Hours, Attendance

Reading, Notes, Listening_in_Class, Project_work

Target variable: Grade (converted to HIGH/LOW)

