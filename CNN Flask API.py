import os
import flask
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model

model = load_model("Models/Mobilenet.h5")
app = flask.Flask(__name__)

@app.route("/")
def index():
    # Render the home page template with the image upload form
    return flask.render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    if flask.request.method == "POST":
        # Get the uploaded file from the request
        file = flask.request.files["file"]

        # Save the file to a temporary directory
        file_path = os.path.join(app.root_path, "temp", file.filename)
        file.save(file_path)

        # Load the image file and preprocess it
        img = image.load_img(file_path, target_size = (224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis = 0)

        # Make predictions using the loaded model
        predictions = model.predict(x)

        # Get the predicted class label
        if predictions[0][0] > 0.25:
            label = "Dusty"
        else:
            label = "Clean"

        # Render the result template with the predicted class label
        return flask.render_template("result.html", label = label)

if __name__ == "__main__":
    app.run(debug=True)
