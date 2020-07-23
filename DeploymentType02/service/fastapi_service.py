# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ------------------------
# REST service via FastAPI
# ------------------------

from typing import Optional
from fastapi import FastAPI

app = FastAPI()


#Define a default route
@app.get('/')
def main_page():
    return 'REST service is active via FastAPI'

@app.post("/model/predict/")
async def create_upload_file(file: UploadFile = File(...)):
    return {"\nfilename": file.filename}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
