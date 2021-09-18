from flask import Flask, request
import classifier
app = Flask(__name__)


@app.route("/predict", methods=['POST'])
def generate_prediction():
    prediction = classifier.generate_prediction(request.get_json()['image_base64'])
    print(prediction)
    return prediction


if __name__ == '__main__':
    app.run()

