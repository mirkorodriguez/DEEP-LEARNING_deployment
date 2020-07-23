# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ------------------------
# REST service via FastAPI
# ------------------------

from typing import Optional
from fastapi import FastAPI, File, UploadFile

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
async def create_upload_file(file: UploadFile = File(...)):
    if file and allowed_file(file.filename):
        print("\nfilename:",file.filename)
        contents = await file.read()
        print("\ncontents:",contents)
    return {"filename": file.filename}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
