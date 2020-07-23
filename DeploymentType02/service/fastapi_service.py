# Developed by Mirko J. Rodríguez mirko.rodriguezm@gmail.com

# ----------------------------
# REST service under port 9000
# ----------------------------

from typing import Optional
from fastapi import FastAPI

app = FastAPI()


#Define a default route
@app.route('/')
def main_page():
    return '¡REST service is active via FastAPI (version: '+fastapi.__version__+')!'

@app.get("/items/{item_id}")
def read_item(item_id: int, q: Optional[str] = None):
    return {"item_id": item_id, "q": q}
