# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ------------------------
# REST service via FastAPI
# ------------------------

#Import FastAPI libraries
from fastapi import FastAPI, File, UploadFile
from typing import Optional
from werkzeug.utils import secure_filename

#Import Tensorflow image
from tensorflow.keras.preprocessing import image

UPLOAD_FOLDER = '../images/uploads'
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

#Main definition for FastAPI
app = FastAPI()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

#Define a default route
@app.get('/')
def main_page():
    return 'REST service is active via FastAPI'

@app.post("/model/predict/")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    if file and allowed_file(filename):
        print("\nFilename received:",filename)
        contents = await file.read()
        filename = secure_filename(filename)
        tmpfile = ''.join([UPLOAD_FOLDER ,'/',filename])
        with open(tmpfile, 'wb') as f:
            f.write(contents)
        print("\nFilename stored:",tmpfile)

        #loading image
        image_to_predict = image.load_img(tmpfile, target_size=(224, 224))

    return {"filename received": file.filename}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
