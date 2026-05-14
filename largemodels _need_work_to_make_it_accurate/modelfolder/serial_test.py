import joblib

le = joblib.load("Models/labels.pkl")
print(le.classes_)