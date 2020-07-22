import argparse
import json
import numpy as np
import requests
from tensorflow.keras.preprocessing import image

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image PATH is needed.")
ap.add_argument("-m", "--model", required=True, help="Model NAME is needed.")
ap.add_argument("-p", "--port", required=True, help="Model PORT number is needed.")
args = vars(ap.parse_args())

image_path = args['image']
model_name = args['model']
port = args['port']

print("\nModel:",model_name)
print("Image:",image_path)
print("Puerto:",port)


# Image processing
test_image = image.load_img(image_path,target_size = (224, 224))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image.astype('float32')
test_image /= 255.0

data = json.dumps({"signature_name": "serving_default", "instances": test_image.tolist()})
headers = {"content-type": "application/json"}
uri = ''.join(['http://127.0.0.1:',port,'/v1/models/',model_name,':predict'])
print("URI:",uri)

json_response = requests.post(uri, data=data, headers=headers)
predictions = json.loads(json_response.text)['predictions'][0]
print(predictions)
index = np.argmax(predictions)

CLASSES = ['Daisy', 'Dandelion', 'Rosa', 'Girasol', 'Tulipán']
ClassPred = CLASSES[index]
ClassProb = predictions[index]

print("Index:", index)
print("Pedicción:", ClassPred)
print("Prob:", ClassProb)
