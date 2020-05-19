import os
import io
from werkzeug.utils import secure_filename
from flask import Flask, render_template, request, jsonify
from PIL import Image
from tensorflow.keras.preprocessing import image
img_width, img_height = 224, 224

PATH_UPLOAD_FOLDER = 'upload/'

app = Flask(__name__)
app.config['PATH_UPLOAD_FOLDER'] = PATH_UPLOAD_FOLDER
@app.route("/")
def initial():
  app.logger.debug("Loading initial route")
  data = {"msg": "Before prediction."}
  return render_template("index.html", data=data)
@app.route("/predict/NP", methods=["POST"])
def predictNP():

  if request.method == "POST":
    if request.files.get("img"):
        input_img = request.files["img"]

        
        image=secure_filename(input_img.filename)

        input_img.save(os.path.join(PATH_UPLOAD_FOLDER, image))


if __name__ == "__main__":
       app.run(host='0.0.0.0', 
               port=9001, 
               debug=True)


