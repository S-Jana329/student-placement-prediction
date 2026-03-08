from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open("model.pkl","rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict",methods=["POST"])
def predict():

    cgpa = float(request.form["cgpa"])
    internships = int(request.form["internships"])
    projects = int(request.form["projects"])
    workshops = int(request.form["workshops"])

    features = np.array([[cgpa,internships,projects,workshops]])

    prediction = model.predict(features)

    if prediction == 1:
        result = "Student is likely to be Placed"
    else:
        result = "Student may not get placed"

    return render_template("index.html",prediction_text=result)

if __name__ == "__main__":
    app.run(debug=True)