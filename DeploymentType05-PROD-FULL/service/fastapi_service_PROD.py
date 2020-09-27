# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ------------------------
# REST service via FastAPI
# ------------------------

#Import FastAPI libraries
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from typing import List
from werkzeug.utils import secure_filename

import json
import numpy as np
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Main definition for FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a default route
@app.get('/')
def main_page():
    return 'REST service is active via FastAPI'

@app.get("/form")
async def main():
    content = """
<body>
<form action="/model/predict/" enctype="multipart/form-data" method="post">
<input name="files" type="file" multiple accept=".jpg,.jpeg,.png">
<input type="submit">
</form>
</body>
    """
    return HTMLResponse(content=content)

@app.post("/model/predict/")
async def predict(files: List[UploadFile] = File(...)):
    data = {"success": False}

    #Validate and save received image files
    tmpfiles=[]
    for file in files:
        filename = file.filename
        if file and allowed_file(filename):
            print("\nFilename received:",filename)
            contents = await file.read()
            filename = secure_filename(filename)
            tmpfile = ''.join([UPLOAD_FOLDER ,'/',filename])
            with open(tmpfile, 'wb') as f:
                f.write(contents)
            print("Filename stored:",tmpfile)
            tmpfiles.append(tmpfile)

    #If we have at least one valid image
    if(len(tmpfiles) > 0):
        #model
        model_name='flowers'
        model_version='1'
        port_gRPC='9500'

        predictions_list = predict_via_gRPC_batch(tmpfiles,model_name,model_version,port_gRPC)

        CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

        data["predictions"] = []
        for i in range(len(predictions_list)):
            predictions = predictions_list[i]
            index = predictions.index(max(predictions))
            ClassPred = CLASSES[index]
            ClassProb = predictions[index]

            print("\nImage:", tmpfiles[i])
            print("predictions:", predictions)
            print("Index:", index)
            print("Pred:", ClassPred)
            print("Prob:", ClassProb)

            #Results as Json
            r = {"image": tmpfiles[i].split('/')[-1],"label": ClassPred, "score": float(ClassProb)}
            data["predictions"].append(r)

            #Success
            data["success"] = True

    return data

def predict_via_gRPC_batch(images_to_predict,model_name,model_version,port):
    import grpc
    import tensorflow as tf
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2_grpc

    print("\nImages to predict:",images_to_predict)
    print("Model:",model_name)
    print("Model version:",model_version)
    print("Port:",port)

    host = "127.0.0.1"
    server = host + ':' + port
    model_version = int(model_version)
    request_timeout = float(10)

    #Loading images
    image_data=[]
    for image_filepath in images_to_predict:
      print("image_path:",image_filepath)
      test_image = image.load_img(image_filepath, target_size=(224, 224))
      test_image = image.img_to_array(test_image)
      test_image = test_image.astype('float32')
      test_image = test_image / 255.0
      image_data.append(test_image)

    image_data_batch = np.array(image_data).astype(np.float32)

    # Create gRPC client and request
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.version.value = model_version
    request.model_spec.signature_name = "serving_default"
    request.inputs['vgg16_input'].CopyFrom(tf.make_tensor_proto(image_data_batch,shape=image_data_batch.shape)) # shape=[num_imgs, 224, 224, 3]

    # Send request
    result_predict = str(stub.Predict(request, request_timeout))
    print("\nresult_predict:",result_predict)

    num_classes = 5
    values = result_predict.split('float_val:')[1:len(image_data)*num_classes + 1]

    predictions = []
    for element in values:
      value = element.split('\n')[0]
      print("value:",value)
      predictions.append(float("{:.8f}".format(float(value))))
    # print("\npredictions:", predictions)

    predictions_list = list(divide_chunks(predictions, num_classes))

    return predictions_list


def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]
