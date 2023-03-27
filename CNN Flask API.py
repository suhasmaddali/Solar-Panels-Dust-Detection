import flask
from tensorflow.keras.models import load_model

model = load_model("Models/Mobilenet.h5")
app = flask.Flask(__name__)

@app.route("/")
def home():
    return flask.render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    # Get the input data from the request
    input_data = flask.request.json["input_data"]

    # Make predictions using the loaded model
    predictions = model.predict(input_data)

    # Return the predictions as a JSON response
    return flask.jsonify(predictions.tolist())

if __name__ == '__main__':
    app.run(debug = True)
