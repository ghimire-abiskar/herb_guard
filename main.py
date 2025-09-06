from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
import random  # To simulate a dummy prediction
from fastapi.middleware.cors import CORSMiddleware

# Enable CORS for all origins (useful for development)

# Initialize FastAPI app
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Define the response model for prediction results
class PredictionResponse(BaseModel):
    prediction: str  # Dummy prediction result

# Endpoint to handle image upload and prediction
@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    # Read the uploaded image file into memory (for now, we're ignoring the actual file contents)
    image_data = await file.read()

    # Simulate a prediction result. This can later be replaced with actual ML model logic.
    prediction = random.choice(["Cat", "Dog", "Bird"])

    # Return the prediction result
    return PredictionResponse(prediction=prediction)

# Optional: Root endpoint to test if the server is running
@app.get("/")
async def root():
    return {"message": "Welcome to the ML API! Send images to /predict"}
