import os
import io
import sys
import logging
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import img_to_array

img_width, img_height = 224, 224

PATH_UPLOAD_FOLDER = 'upload/'

app = Flask(__name__)
app.config['PATH_UPLOAD_FOLDER'] = PATH_UPLOAD_FOLDER
if True:
  app.logger.addHandler(logging.StreamHandler(sys.stdout))
  app.logger.setLevel(logging.ERROR)
model_NP = load_model("static/models/NormalvsPneumonia-model_xray1.h5")
model_NP.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print("---------------First model loaded---------------")
model_BV = load_model("static/models/model_xray2.h5")
model_BV.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print("---------------Second model loaded---------------")
graph = tf.compat.v1.get_default_graph()
@app.route("/")
def initial():
  app.logger.debug("Loading initial route")
  data = {"msg": "Before prediction."}
  return render_template("index.html", data=data)
@app.route("/", methods = ['POST'])
def upload_file():
   data = {"msg": "file uploaded"}
   if request.method == 'POST':
     if request.files.get("img"):
       input_img = request.files["img"]
       image=secure_filename(input_img.filename)
       input_img.save(os.path.join(PATH_UPLOAD_FOLDER, image))
       return jsonify(data),200
@app.route("/predict/NP", methods=["POST"])
def predictNP():
  data = {
    "class": "No Class Predicted",
    "confidence": 0.0,
    "error": "",
    "msg": "Prediction 1 Pending",
    "prediction": -1,
    "success": False
  }
  if request.method == "POST":
    if request.files.get("img"):

      
      try:

       
        input_image = request.files["img"].read()
        fn = request.files["img"].filename

        
        input_img = Image.open(io.BytesIO(input_image))
        input_image = img_to_array(input_img)
        input_image = np.resize(input_image, (img_width, img_height, 3))
        input_image = np.expand_dims(input_image, axis=0)
        final_image = np.vstack([input_image])
        
        with graph.as_default():
          model_NP = load_model("static/models/NormalvsPneumonia-model_xray1.h5")
          preds = model_NP.predict(final_image)
          print(preds)
          predClass = int(np.argmax(preds))
          print(predClass)
          data["msg"] = "Prediction done"
          if predClass == 0:
            data["success"] = True
            data["class"] = "Normal"
            data["prediction"] = 0
            app.logger.debug(data);
            return jsonify(data), 200
          elif predClass == 1:
            data["success"] = True
            data["class"] = "Pneumonia"
            data["prediction"] = 1
            app.logger.debug(data);
            return jsonify(data), 200
      except Exception as e:
        data["error"] = "Some Server Error Occurred."
        app.logger.error(str(e))
        return jsonify(data), 500
  else:
    app.logger.debug("Non POST request at /predict")
    data["msg"] = "Not a POST request"
    data["error"] = "Forbidden"
    return jsonify(data), 403

 
@app.route("/predict/BV", methods=["POST"])
def predictBV():
  data = {
    "success": False,
    "class": "No Class Predicted",
    "msg": "Prediction 2 Pending",
    "prediction": -1,
    "error": ""
  }
  if request.method == "POST":
    if request.files.get("img"):

      try:
        input_image = request.files["img"].read()
        input_image = Image.open(io.BytesIO(input_image))
        fn = request.files["img"].filename
        input_image = img_to_array(input_image)
        input_image = np.resize(input_image, (img_width, img_height, 3))
        input_image = np.expand_dims(input_image, axis=0)
        final_image = np.vstack([input_image])
        with graph.as_default():
          model_BV = load_model("static/models/model_xray2.h5")
          preds = model_BV.predict(final_image)
          print(preds)
          predClass = int(np.argmax(preds))
          print(predClass)
          data["msg"] = "Prediction done"
          if predClass == 0:
            data["class"] = "Bacterial"
            data["success"] = True
            data["prediction"] = 0
            app.logger.debug(data);
            return jsonify(data), 200
          elif predClass == 1:
            data["class"] = "Viral"
            data["success"] = True
            data["prediction"] = 1
            app.logger.debug(data);
            return jsonify(data), 200
      except Exception as e:
        data["error"] = "Some Server Error Occurred in /predict/BV"
        app.logger.error(str(e))
        return jsonify(data), 500
  else:
    app.logger.debug("Non POST request at /predict/BV")
    data["msg"] = "Not a POST request"
    data["error"] = "Forbidden"
    return jsonify(data), 403

if __name__ == "__main__":
  app.logger.debug("Loading the Pneumonia Keras models and starting Flask server, please wait...")
  app.run(host='0.0.0.0', 
               port=9001, 
               debug=True)



  input_img = request.files["img"]
  image=secure_filename(input_img.filename)
  input_img.save(os.path.join(PATH_UPLOAD_FOLDER, image))



  
