import torch
import pickle
from backend.model import PictogramMLP

with open("artifacts/vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open("artifacts/label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open("artifacts/personal_libraries.pkl", "rb") as f:
    personal_libraries = pickle.load(f)

with open("artifacts/global_library.pkl", "rb") as f:
    global_library = pickle.load(f)

input_dim = len(vectorizer.vocabulary_)
output_dim = len(label_encoder.classes_)

model = PictogramMLP(input_dim, output_dim)
model.load_state_dict(torch.load("artifacts/best_model.pth"))
model.eval()
