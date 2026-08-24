from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
import torch
from ml_engine.stgnn_model import SpatioTemporalGNN, generate_adjacency_matrix

app = FastAPI(
    title="HealthAI Edge-To-Cloud API Framework",
    description="Asynchronous RESTful API for real-time biomedical signal inference and anomaly detection.",
    version="1.0.0"
)

security = HTTPBearer()
SECRET_API_TOKEN = "healthai-research-secure-key"

def verify_bearer_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    if credentials.credentials != SECRET_API_TOKEN:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or unauthorized Security Bearer Token."
        )
    return credentials.credentials

class SignalPayload(BaseModel):
    num_channels: int = Field(..., example=19)
    sequence_length: int = Field(..., example=256)
    data: list = Field(..., description="2D matrix of shape [num_channels, sequence_length]")

# Initialize PyTorch Model and Adjacency Matrix
ADJ_MATRIX = generate_adjacency_matrix(num_nodes=19)
MODEL = SpatioTemporalGNN()
MODEL.eval()

@app.get("/")
def health_check():
    return {"status": "online", "system": "HealthAI Telemetry Microservice", "version": "1.0.0"}

@app.post("/api/v1/predict", dependencies=[Depends(verify_bearer_token)])
async def predict_signal(payload: SignalPayload):
    try:
        raw_tensor = torch.tensor(payload.data, dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            logits = MODEL(raw_tensor, ADJ_MATRIX)
            probs = torch.softmax(logits, dim=1)
            predicted_class = torch.argmax(probs, dim=1).item()
            
        return {
            "status": "success",
            "prediction_class": predicted_class,
            "confidence_score": round(probs[0][predicted_class].item(), 4),
            "anomaly_detected": bool(predicted_class == 1)
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Inference Processing Error: {str(e)}")
