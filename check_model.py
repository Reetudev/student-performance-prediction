import joblib

model = joblib.load("student_prediction_model.pkl")

print(type(model))
print("Model loaded successfully!")
