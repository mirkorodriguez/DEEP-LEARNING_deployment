# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# --------------------------------------------
# Exponiendo el servicio Web en el puerto 5000
# --------------------------------------------

#Import Flask
from flask import Flask, request, jsonify
from flask_cors import CORS

#Import Tensorflow
import tensorflow as tf

#Import libraries
import numpy as np
import os
from werkzeug.utils import secure_filename
from model_loader import cargarModeloH5

UPLOAD_FOLDER = '../images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

port = int(os.getenv('PORT', 5000))
print ("Port recognized: ", port)

#Initialize the application service
app = Flask(__name__)
CORS(app)
global loaded_model
loaded_model = cargarModeloH5()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Funciones
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a default route
@app.route('/')
def main_page():
	return '¡REST service is active via Flask!'

# Model route
@app.route('/model/predict/',methods=['POST'])
def predict():
    data = {"success": False}
    if request.method == "POST":
        # check if the post request has the file part
        if 'file' not in request.files:
            print('No file part')
        file = request.files['file']
        # if user does not select file, browser also submit a empty part without filename
        if file.filename == '':
            print('No selected file')
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            #loading image
            filename = UPLOAD_FOLDER + '/' + filename
            print("\nfilename:",filename)

            image_to_predict = tf.keras.preprocessing.image.load_img(filename, target_size=(224, 224))
            test_image = tf.keras.preprocessing.image.img_to_array(image_to_predict)
            test_image = np.expand_dims(test_image, axis = 0)
            test_image = test_image.astype('float32')
            test_image /= 255.0

            predictions = loaded_model.predict(test_image)[0]
            index = np.argmax(predictions)
            CLASSES = ['Daisy', 'Dandelion', 'Rosa', 'Girasol', 'Tulipán']
            ClassPred = CLASSES[index]
            ClassProb = predictions[index]

            print("Classes:", CLASSES)
            print("Predictions",predictions)
            print("Predicción Index:", index)
            print("Predicción Label:", ClassPred)
            print("Predicción Prob: {:.2%}".format(ClassProb))

            #Results as Json
            data["predictions"] = []
            r = {"label": ClassPred, "score": float(ClassProb)}
            data["predictions"].append(r)

            #Success
            data["success"] = True

    return jsonify(data)

# Run de application
app.run(host='0.0.0.0',port=port, threaded=False)
