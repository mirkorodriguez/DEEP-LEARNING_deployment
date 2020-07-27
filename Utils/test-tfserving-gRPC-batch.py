# -----------------------------------
# Testing TensorFlow Serving via gRPC
# Author: Mirko Rodriguez
# -----------------------------------

import grpc
import argparse

from tensorflow.python.framework import tensor_util
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

from tensorflow.keras.preprocessing import image

# Args
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image PATH is required.")
ap.add_argument("-m", "--model", required=True, help="Model NAME is required.")
ap.add_argument("-v", "--version", required=True, help="Model VERSION is required.")
ap.add_argument("-p", "--port", required=True, help="Model PORT number is required.")
args = vars(ap.parse_args())

image_path = args['image']
model_name = args['model']
model_version = args['version']
port = args['port']

print("\nModel:",model_name)
print("Model version:",model_version)
print("Image:",image_path)
print("Port:",port)

host = "127.0.0.1"
port = port
server = host + ':' + port
model_name = model_name
model_version = int(model_version)
request_timeout = float(10)
image_filepaths = [image_path]

# Loading image
test_image = image.load_img(image_filepaths[0], target_size=(224, 224))
test_image = image.img_to_array(test_image)
test_image = test_image.astype('float32')
test_image = test_image / 255.0

# Create gRPC client and request
channel = grpc.insecure_channel(server)
stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
request = predict_pb2.PredictRequest()
request.model_spec.name = model_name
request.model_spec.version.value = model_version
request.model_spec.signature_name = "serving_default"
request.inputs['vgg16_input'].CopyFrom(tensor_util.make_tensor_proto(test_image, shape=[1] + list(test_image.shape)))
# https://gitmemory.com/issue/tensorflow/serving/1267/472128977
# # https://sudonull.com/post/30758-How-We-Increased-Tensorflow-Serving-Productivity-by-70
# host = "127.0.0.1"
# port = port
# server = host + ':' + port
# model_name = model_name
# model_version = int(model_version)
# request_timeout = float(10)
# #image_filepaths = [image_path]
# image_filepaths = ['/home/mirko_stem/DEEP-LEARNING_deployment/DeploymentType02/images/test/img01.jpg','/home/mirko_stem/DEEP-LEARNING_deployment/DeploymentType02/images/test/img02.jpg','/home/mirko_stem/DEEP-LEARNING_deployment/DeploymentType02/images/test/img03.jpg']
#
# # Loading image
# image_data=[]
# for image_filepath in image_filepaths:
#   print("image_path",image_filepath)
#   test_image = image.load_img(image_filepath, target_size=(224, 224))
#   test_image = image.img_to_array(test_image)
#   test_image = test_image.astype('float32')
#   test_image = test_image / 255.0
#   image_data.append(test_image)
# #  with open(image_filepath,'rb') as f:
# #    image_data.append(f.read())
#
# # Create gRPC client and request
# channel = grpc.insecure_channel(server)
# stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# request = predict_pb2.PredictRequest()
# request.model_spec.name = model_name
# request.model_spec.version.value = model_version
# request.model_spec.signature_name = "serving_default"
# request.inputs['vgg16_input'].CopyFrom(tf.make_tensor_proto(image_data, shape=[3, 224, 224, 3]))

# Send request
result_predict = str(stub.Predict(request, request_timeout))
# print("\nresult_predict:",result_predict)

CLASSES = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']
values = result_predict.split('float_val:')[1:len(CLASSES) + 1]

predictions = []
for element in values:
  value = element.split('\n')[0]
  print("value:",value)
  predictions.append(float("{:.8f}".format(float(value))))
print("\npredictions:", predictions)

index = predictions.index(max(predictions))
ClassPred = CLASSES[index]
ClassProb = predictions[index]


print("Index:", index)
print("Pred:", ClassPred)
print("Prob:", ClassProb)
