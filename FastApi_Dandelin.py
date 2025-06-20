# !pip install fastapi Pillow python-multipart torch transformers uvicorn nest-asyncio pyngrok

import os
from dotenv import load_dotenv

# ✅ Load environment variables from .env
load_dotenv()

# ✅ Ngrok authentication
from pyngrok import ngrok
ngrok.set_auth_token(os.getenv("NGROK_AUTH_TOKEN"))

# ✅ Hugging Face authentication
from huggingface_hub import login
login(os.getenv("HF_TOKEN"))

# ✅ Continue with model loading
from transformers import ViltProcessor, ViltForQuestionAnswering
import requests
from PIL import Image

processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
model = ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")



# prepare image + question
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)

text = "What are the colors of the cats?"

# prepare inputs
encoding = processor(image, text, return_tensors="pt")



# forward pass
outputs = model(**encoding)
logits = outputs.logits
idx = logits.argmax(-1).item()


print("Predicted answer:", model.config.id2label[idx])

# TODO: put above code into a function that accepts image and text as input

def model_pipeline(text: str, image: Image):
    # prepare inputs
    encoding = processor(image, text, return_tensors="pt")

    # forward pass
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()

    return  model.config.id2label[idx]

from typing import Union
from fastapi import FastAPI, UploadFile
import io
from PIL import Image

app = FastAPI()


@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/ask")
def ask(text: str, image: UploadFile):
    content = image.file.read()

    image = Image.open(io.BytesIO(content))
    image = Image.open(image.file)

    result = model_pipeline(text, image)
    return {"answer": result}








import nest_asyncio
from pyngrok import ngrok
import uvicorn


# ngrok.set_auth_token("2iu15NKJ9dV750sK2gVGciIG8rF_3JTn7dcWzN6c277V77RoU")
ngrok.set_auth_token("28QuBbclrQ7cCyOfxGpBokBbnEF_6R6iV6xHdqkiDwPoqTv2u")
ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)











# Convert the Colab Notebook to a Python script

# !jupyter nbconvert --to script FastApi_Dandelin.ipynb

# !ls




