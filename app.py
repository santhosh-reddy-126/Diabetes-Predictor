from flask import Flask, request, jsonify, render_template
import joblib
app=Flask(__name__)

model = joblib.load('sugar_pred.pkl')
sc = joblib.load('sc.pkl')
@app.route('/')
def Home():
    return render_template("app.html")

@app.route('/predict', methods=['POST'])
def prediction():
    try:
        features = [float(x) for x in request.form.values()]
    except ValueError as e:
        return render_template("app.html",error="Error: Please fill all the Details")
    pred = model.predict(sc.transform([features]))
    
    if pred[0]==0:
        return render_template("app.html",pred="You are Safe!.We are 78.4% Accurate")
    else:
        return render_template("app.html",pred="Diabetes Alert! Consult a Doctor.We are 78.4% Accurate")
if __name__=="__main__":
    app.run()