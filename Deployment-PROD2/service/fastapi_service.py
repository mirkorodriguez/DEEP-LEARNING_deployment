# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ------------------------
# REST service via FastAPI
# ------------------------

#Import FastAPI libraries
from fastapi import FastAPI, File, UploadFile
from starlette.middleware.cors import CORSMiddleware
from werkzeug.utils import secure_filename

import json
import numpy as np
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = 'uploads/images'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Main definition for FastAPI
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=['*'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a default route
@app.get('/')
def main_page():
    return 'REST service is active via FastAPI'

@app.post("/model/predict/")
async def predict(file: UploadFile = File(...)):
    data = {"success": False}
    filename = file.filename
    if file and allowed_file(filename):
        print("\nFilename received:",filename)
        contents = await file.read()
        filename = secure_filename(filename)
        tmpfile = ''.join([UPLOAD_FOLDER ,'/',filename])
        with open(tmpfile, 'wb') as f:
            f.write(contents)
        print("\nFilename stored:",tmpfile)

        #model
        model_name='flowers'
        model_version='1'
        port_HTTP='9501'
        port_gRPC='9500'

        predictions = predict_via_HTTP(tmpfile,model_name,model_version,port_HTTP)
        #predictions = predict_via_gRPC(tmpfile,model_name,model_version,port_gRPC)

        index = np.argmax(predictions)
        CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
        ClassPred = CLASSES[index]
        ClassProb = predictions[index]

        print("Index:", index)
        print("Pred:", ClassPred)
        print("Prob:", ClassProb)

        #Results as Json
        data["predictions"] = []
        r = {"label": ClassPred, "score": float(ClassProb)}
        data["predictions"].append(r)

        #Success
        data["success"] = True

    return data

def predict_via_HTTP(image_to_predict,model_name,model_version,port):
    import requests

    print("\nImage:",image_to_predict)
    print("Model:",model_name)
    print("Model version:",model_version)
    print("Port:",port)

    test_image = image.load_img(image_to_predict, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image.astype('float32')
    test_image /= 255.0

    data = json.dumps({"signature_name": "serving_default", "instances": test_image.tolist()})
    headers = {"content-type": "application/json"}
    uri = ''.join(['http://127.0.0.1:',port,'/v',model_version,'/models/',model_name,':predict'])
    print("URI:",uri)

    json_response = requests.post(uri, data=data, headers=headers)
    predictions = json.loads(json_response.text)['predictions'][0]
    print("\npredictions:",predictions)

    return predictions


def predict_via_gRPC(image_to_predict,model_name,model_version,port):
    import grpc
    from tensorflow.python.framework import tensor_util
    from tensorflow_serving.apis import predict_pb2
    from tensorflow_serving.apis import prediction_service_pb2_grpc

    print("\nImage:",image_to_predict)
    print("Model:",model_name)
    print("Model version:",model_version)
    print("Port:",port)

    host = "127.0.0.1"
    server = host + ':' + port
    model_version = int(model_version)
    request_timeout = float(10)

    test_image = image.load_img(image_to_predict, target_size=(224, 224))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis = 0)
    test_image = test_image.astype('float32')
    test_image /= 255.0

    image_data = np.array(test_image).astype(np.float32)

    # Create gRPC client and request
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.version.value = model_version
    request.model_spec.signature_name = "serving_default"
    request.inputs['vgg16_input'].CopyFrom(tensor_util.make_tensor_proto(image_data,shape=list(image_data.shape)))

    # Send request
    result_predict = str(stub.Predict(request, request_timeout))
    # print("\nresult_predict:",result_predict)

    num_classes = 5
    values = result_predict.split('float_val:')[1:num_classes + 1]

    predictions = []
    for element in values:
      value = element.split('\n')[0]
      predictions.append(float("{:.8f}".format(float(value))))
    print("\npredictions:", predictions)

    return predictions
