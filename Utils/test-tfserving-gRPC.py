import argparse
import numpy as np
import grpc

import tensorflow as tf
from tensorflow.python.framework import tensor_util
# from grpc.beta import implementations
from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2
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

def main():
  # host = "127.0.0.1"
  # port = port
  # server = host + ':' + port
  # model_name = model_name
  # model_version = model_version
  # request_timeout = float(10)
  # image_filepaths = [image_path]
  host = "127.0.0.1"
  port = "8500"
  server = host +':'+port
  model_name = "flowers"
  model_version = 1
  request_timeout = 10.0
  image_filepaths = ["/home/mirko_stem/DEEP-LEARNING_deployment/DeploymentType02/images/test/img01.jpg"]

  for index, image_filepath in enumerate(image_filepaths):
    image_ndarray = image.img_to_array(image.load_img(image_filepaths[0], target_size=(224, 224)))
    image_ndarray = image_ndarray / 255.

  # Create gRPC client and request
  # channel = implementations.insecure_channel(host, port)
  channel = grpc.insecure_channel(server)
  # stub = prediction_service_pb2.beta_create_PredictionService_stub(channel)
  stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.version.value = model_version
  request.model_spec.signature_name = "serving_default"
  request.inputs['vgg16_input'].CopyFrom(tensor_util.make_tensor_proto(image_ndarray, shape=[1] + list(image_ndarray.shape)))

  # Send request
  predictions = str(stub.Predict(request, request_timeout))
  print(predictions)

  CLASSES = ['Daisy', 'Dandelion', 'Rosa', 'Sunflower', 'Tulip']
  mylist = predictions.split('float_val:')[1:len(CLASSES) + 1]

  values = []
  for element in mylist:
      values.append(float("{:.6f}".format(float(element))))

  index = values.index(max(values))
  ClassPred = CLASSES[index]
  ClassProb = values[index]

  print(values)
  print(ClassPred)
  print(ClassProb)

if __name__ == '__main__':
  main()
