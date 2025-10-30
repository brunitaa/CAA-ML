from fastapi import FastAPI
from pydantic import BaseModel
import torch
from backend.load_model import model, vectorizer, label_encoder, personal_libraries, global_library

app = FastAPI(title="CAA-ML Predictor")

class PredictRequest(BaseModel):
    text: str
    speaker_id: int

@app.post("/predict")
def predict(request: PredictRequest):
    X = vectorizer.transform([request.text]).toarray()
    X_tensor = torch.tensor(X, dtype=torch.float32)

    with torch.no_grad():
        y_pred = model(X_tensor)
        topk = torch.topk(y_pred, k=5)
        top_indices = topk.indices[0].tolist()
        top_labels = label_encoder.inverse_transform(top_indices).tolist()

    # Filtrar según librerías
    personal_lib = personal_libraries.get(request.speaker_id, set())
    final_labels = [p for p in top_labels if p in personal_lib or p in global_library][:5]

    return {"predictions": final_labels}
