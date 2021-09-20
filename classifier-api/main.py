from flask import Flask, request
from flasgger import Swagger, swag_from
import classifier

app = Flask(__name__)
app.config["SWAGGER"] = {"title": "Swagger-UI", "uiversion": 2}
swagger_config = {
    "headers": [],
    "specs": [
        {
            "endpoint":  "apispec_1",
            "route": "/apispec_1.json",
            "rule_filter": lambda rule: True,
            "model_filter": lambda tag: True
        }
    ],
    "static_url_path": "/flasgger_static",
    "swagger_ui": True,
    "specs_route": "/swagger/",
}

swagger = Swagger(app, config=swagger_config)


@app.route("/predict", methods=['POST'])
@swag_from("swagger_config.yml")
def generate_prediction():
    prediction = classifier.generate_prediction(request.get_json()['image_base64'])
    print(prediction)
    return prediction


if __name__ == '__main__':
    app.run()

