import argparse
import json
import numpy as np
import requests
from tensorflow.keras.preprocessing import image

# Argumentos
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Image path is needed.")
ap.add_argument("-m", "--model", required=True, help="Model name is needed.")
ap.add_argument("-p", "--port", required=True, help="Model PORT number is needed.")
args = vars(ap.parse_args())

image_path = args['image']
model_name = args['model']
port = args['port']

print("\nModel:",model_name)
print("Image:",image_path)
print("Puerto:",port)


# Preprocesar imagen
image = image.img_to_array(image.load_img(image_path, target_size=(224, 224)))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
test_image = test_image.astype('float32')
test_image /= 255.0
payload = {"instances": [{'input_image': img.tolist()}]}

# URI
uri = ''.join(['http://127.0.0.1:',port,'/v1/models/',model_name,':predict'])
print("URI:",uri)

# Request al modelo desplegado en TensorFlow Serving
r = requests.post(uri, json=payload)
pred = json.loads(r.content.decode('utf-8'))

# Decodificando decoder util
predictions = decode_predictions(np.array(pred['predictions']),top=1)
print("Predictions:\n",predictions)
print("Class:",predictions[0][0][1])
print("Score",predictions[0][0][2])
