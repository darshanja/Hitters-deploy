import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	int_features = [int(x) for x in request.form.values()]
	final_features = [np.array(int_features)]
	prediction = model.predict(final_features)
	prediction = prediction**2.72
	output = round(prediction[0], 2)
	return render_template('index.html', prediction_text='Salary of the player should be $ {}'.format(output))

@app.route('/predict_api',methods=['POST'])
def predict_api():
	data = request.get_json(force=True)
	prediction = model.predict([np.array(list(data.values()))])
	output = prediction**2.72
	return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)