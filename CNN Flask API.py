import os
import flask
import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("Models/Mobilenet.h5")
app = flask.Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def home():
    if flask.request.method == "POST":
        # Get the uploaded file from the request
        file = flask.request.files["file"]

        # Save the file to a temporary directory
        file_path = os.path.join(app.root_path, "temp", file.filename)
        file.save(file_path)

        # Load the image file and preprocess it
        img = cv2.imread(file_path, target_size=(224, 224))
        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)
        x = preprocess_input(x)

        # Make predictions using the loaded model
        predictions = model.predict(x)

        # Get the predicted class label
        label = np.argmax(predictions[0])

        # Return the predicted class label as a response
        return flask.render_template("result.html", label=label)

    else:
        # Render the home page template with the image upload form
        return flask.render_template("home.html")



if __name__ == '__main__':
    app.run(debug = True)
