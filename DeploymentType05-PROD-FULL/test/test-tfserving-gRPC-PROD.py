# -----------------------------------
# Testing TensorFlow Serving via gRPC
# Author: Mirko Rodriguez
# -----------------------------------

import grpc
import argparse
import tensorflow as tf

from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.keras.preprocessing import image

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-is", "--images", required=True, help="Images' PATH as list are required.")
ap.add_argument("-m", "--model", required=True, help="Model NAME is required.")
ap.add_argument("-v", "--version", required=True, help="Model VERSION is required.")
ap.add_argument("-p", "--port", required=True, help="Model PORT number is required.")
args = vars(ap.parse_args())

images_path = args['images']
model_name = args['model']
model_version = args['version']
port = args['port']

print("\nModel:",model_name)
print("Model version:",model_version)
print("Images:",images_path)
print("Port:",port)

host = "127.0.0.1"
port = port
server = host + ':' + port
model_name = model_name
model_version = int(model_version)
request_timeout = float(10)
image_filepaths = images_path.split(',')

# Loading image
image_data=[]
for image_filepath in image_filepaths:
  print("image_path:",image_filepath)
  test_image = image.load_img(image_filepath, target_size=(224, 224))
  test_image = image.img_to_array(test_image)
  test_image = test_image.astype('float32')
  test_image = test_image / 255.0
  image_data.append(test_image)

import numpy as np
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

CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
values = result_predict.split('float_val:')[1:len(image_data)*len(CLASSES) + 1]

predictions = []
for element in values:
  value = element.split('\n')[0]
  print("value:",value)
  predictions.append(float("{:.8f}".format(float(value))))
# print("\npredictions:", predictions)

def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        yield l[i:i + n]

preditions_list = list(divide_chunks(predictions, len(CLASSES)))

for i in range(len(preditions_list)):
    predictions = preditions_list[i]
    index = predictions.index(max(predictions))
    ClassPred = CLASSES[index]
    ClassProb = predictions[index]

    print("\nImage:", image_filepaths[i])
    print("predictions:", predictions)
    print("Index:", index)
    print("Pred:", ClassPred)
    print("Prob:", ClassProb)
