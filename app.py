from flask import Flask, render_template, request
import numpy as np
import joblib
from keras.models import load_model

app = Flask(__name__)

# Load the trained model and scaler
model = load_model('diabetes_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form values and convert to float
        input_data = [float(request.form[key]) for key in request.form]
        
        # Convert to numpy array and scale it
        input_array = np.array([input_data])
        scaled_input = scaler.transform(input_array)

        # Make prediction
        prediction = model.predict(scaled_input)[0][0]
        confidence = round(prediction * 100, 2)
        
        if prediction >= 0.5:
            result_status = 'Diabetes Risk Detected'
            result_class = 'high-risk'
            tips = [
                "Consult a healthcare professional immediately",
                "Monitor blood sugar levels regularly",
                "Follow a balanced, low-sugar diet",
                "Engage in regular physical activity",
                "Take prescribed medications as directed",
                "Maintain a healthy weight"
            ]
        else:
            result_status = 'Low Diabetes Risk'
            result_class = 'low-risk'
            tips = [
                "Maintain a healthy lifestyle",
                "Exercise regularly (30 min/day)",
                "Eat a balanced diet with whole foods",
                "Stay hydrated and get adequate sleep",
                "Schedule regular health check-ups",
                "Avoid excessive sugar and processed foods"
            ]
        
        return render_template('index.html', 
                             result=result_status,
                             confidence=confidence,
                             result_class=result_class,
                             tips=tips)
    
    except Exception as e:
        return render_template('index.html', 
                             result="Error in prediction",
                             confidence=0,
                             result_class='error',
                             tips=["Please check your input values and try again"])

if __name__ == '__main__':
    app.run(debug=True)
