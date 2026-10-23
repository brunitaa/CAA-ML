import requests
import torch
import pickle
import os
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder
from backend.model import PictogramMLP

BACKEND_URL = os.environ["CAA_AUTO_TRAIN_URL"]

resp = requests.get(BACKEND_URL, timeout=30)
resp.raise_for_status()
data = resp.json()
pictograms = data.get("global", []) + data.get("personal", [])

texts = [p['example_text'] for p in pictograms]
ids = [p['id'] for p in pictograms]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(texts).toarray()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(ids)

input_dim = X.shape[1]
output_dim = len(label_encoder.classes_)
model = PictogramMLP(input_dim, output_dim)

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(100):
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()

print("Reentrenamiento finalizado con loss:", loss.item())

torch.save(model.state_dict(), "artifacts/best_model.pth")
pickle.dump(vectorizer, open("artifacts/vectorizer.pkl", "wb"))
pickle.dump(label_encoder, open("artifacts/label_encoder.pkl", "wb"))
