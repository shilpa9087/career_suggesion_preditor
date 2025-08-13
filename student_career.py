import joblib
import pandas as pd

# Load model and encoders
model = joblib.load("career_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")

# Attendance mapping
attendance_map = {'Never': 0, 'Sometimes': 1, 'Always': 2}

# Helper to normalize yes/no and categorical inputs
def normalize_input(value, col):
    value = value.strip()
    if col in label_encoders:
        # match case-insensitive to encoder classes
        for cls in label_encoders[col].classes_:
            if value.lower() == cls.lower():
                return cls
        # if no match, return first class as default
        return label_encoders[col].classes_[0]
    return value

# Collect user input
student_age = int(input("Enter Student Age: "))
sex = input("Enter Sex (Male/Female): ")
high_school_type = input("Enter High School Type (Private/State/Other): ")
scholarship = input("Enter Scholarship (e.g., 50%): ").replace('%','')
scholarship = int(scholarship)
additional_work = input("Additional Work? (yes/no): ")
sports_activity = input("Sports Activity? (yes/no): ")
transportation = input("Transportation (bus/walk/car): ")
weekly_study_hours = int(input("Enter Weekly Study Hours: "))
attendance = input("Attendance (Never/Sometimes/Always): ")
reading = input("Reading? (yes/no): ")
notes = input("Notes? (yes/no): ")
listening_in_class = input("Listening in Class? (yes/no): ")
project_work = input("Project Work? (yes/no): ")

# Prepare DataFrame
input_df = pd.DataFrame([[
    student_age, sex, high_school_type, scholarship,
    additional_work, sports_activity, transportation,
    weekly_study_hours, attendance,
    reading, notes, listening_in_class, project_work
]], columns=[
    'Student_Age', 'Sex', 'High_School_Type', 'Scholarship',
    'Additional_Work', 'Sports_activity', 'Transportation',
    'Weekly_Study_Hours', 'Attendance',
    'Reading', 'Notes', 'Listening_in_Class', 'Project_work'
])

# Normalize inputs for LabelEncoder
for col in label_encoders.keys():
    input_df[col] = input_df[col].apply(lambda x: normalize_input(x, col))
    input_df[col] = label_encoders[col].transform(input_df[col])

# Map Attendance
input_df['Attendance'] = attendance_map.get(attendance.capitalize(), 0)

# Predict
prediction = model.predict(input_df)[0]

# Output
if prediction == 1:
    print("\n🎯 Career Suggestion: HIGH performance potential")
else:
    print("\n📘 Career Suggestion: LOW performance — focus on skill-building")

