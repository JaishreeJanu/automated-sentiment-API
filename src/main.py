from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
from fastapi.middleware.cors import CORSMiddleware

# Define the request body schema
class SentimentRequest(BaseModel):
    text: str

# Define the response schema
class SentimentResponse(BaseModel):
    sentiment: str
    probability: float

# Initialize FastAPI
app = FastAPI(title="Automated Sentiment API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://automated-sentiment-api-production.up.railway.app/predict", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model at startup to save latency
MODEL_PATH = "models/model.pkl"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

@app.get("/")
def read_root():
    return {"message": "Sentiment API is live! Use the /predict endpoint."}

@app.post("/predict", response_model=SentimentResponse)
def predict_sentiment(request: SentimentRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Check training pipeline.")
    
    try:
        # Logistic Regression returns [0] or [1]
        prediction = model.predict([request.text])[0]
        # Get probability scores
        probs = model.predict_proba([request.text])[0]
        
        sentiment_label = "positive" if prediction == 1 else "negative"
        confidence = float(max(probs))

        return SentimentResponse(
            sentiment=sentiment_label,
            probability=round(confidence, 4)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))