from flask import Flask, request, jsonify
from model import brain
import numpy as np

app = Flask(__name__)

@app.route('/')
def home():
    return "MCLAREN OS 1: Online. Created by MCLARENXCODE Team."

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json['input'] # Expecting something like [0, 1]
    prediction = brain.think(np.array(data))
    return jsonify({
        "status": "Success",
        "output": prediction.tolist(),
        "engine": "MCLAREN OS 1"
    })

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
