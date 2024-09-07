import os

from flask import Flask, request, send_file, jsonify

from predict import prediction

app = Flask(__name__)


@app.route('/')
def hello_world():
    return '''Welcome to AutoForecast! Try running a POST request to /predict...
with the command curl -X POST -F 'file=@path-to-csv-file' http://127.0.0.1:5000/predict -o predictions.csv'''


@app.route('/predict', methods=['POST'])
def predict_route():
    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    file_path = os.path.join(os.getcwd(), file.filename)
    file.save(file_path)

    _prediction = prediction(file_path)
    try:
        return send_file(_prediction, as_attachment=True)
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)
        if os.path.exists(_prediction):
            os.remove(_prediction)


if __name__ == '__main__':
    app.run()
