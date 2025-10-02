from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import transforms
from PIL import Image
import io
import json
from fastapi import Request
from fastapi.responses import JSONResponse
# --------------------
# CONFIG
# --------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMAGE_SIZE = 224
MODEL_PATH = "models/ensemble_traced.pt"
CLASSES_PATH = "models/classes.json"

# --------------------
# FASTAPI APP
# --------------------
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],  
)

# --------------------
# RESPONSE MODEL
# --------------------
class PredictionResponse(BaseModel):
    prediction: str

# --------------------
# LOAD MODEL & CLASSES ONCE
# --------------------
ensemble = torch.jit.load(MODEL_PATH, map_location=DEVICE)
ensemble.eval()

with open(CLASSES_PATH, "r") as f:
    classes = json.load(f)

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# --------------------
# PREDICTION FUNCTION
# --------------------
def predict_image(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    x = transform(img).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = ensemble(x)
        pred_idx = torch.argmax(logits, dim=1).item()
    return classes[pred_idx]

# --------------------
# ENDPOINT
# --------------------
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    prediction = predict_image(image_data)
    return PredictionResponse(prediction=prediction)

@app.get("/")
async def root(request: Request):
    # Handle HEAD requests gracefully
    if request.method == "HEAD":
        return JSONResponse(content=None)
    return {"message": "Welcome to the Herb-Testing API! Send images to /predict"}
