import random
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from itertools import chain
from backend.model import PictogramMLP
import numpy as np

# ------------------------------
# 1. Dataset de ejemplo
# ------------------------------
dataset = [
    {"input_text": "yo quiero jugar", "pictogram_ids": ["1119","1133","1182"], "speaker_id": "speaker1"},
    {"input_text": "tú quieres beber agua", "pictogram_ids": ["1120","1133","1146","1159"], "speaker_id": "speaker2"},
    # Agrega todo tu dataset real aquí...
]

# ------------------------------
# 2. Librerías personalizadas/globales
# ------------------------------
personal_libraries = {}
for d in dataset:
    speaker = d["speaker_id"]
    if speaker not in personal_libraries:
        personal_libraries[speaker] = set()
    personal_libraries[speaker].update(d["pictogram_ids"])

global_library = set(chain.from_iterable([d["pictogram_ids"] for d in dataset]))

# ------------------------------
# 3. Vectorización y codificación
# ------------------------------
all_words = list(set(chain.from_iterable([d['input_text'].split() for d in dataset])))
vectorizer = CountVectorizer(vocabulary=all_words)

X = vectorizer.transform([d['input_text'] for d in dataset]).toarray()
X = torch.tensor(X, dtype=torch.float32)

all_pictograms = list(set(chain.from_iterable([d['pictogram_ids'] for d in dataset])))
label_encoder = LabelEncoder()
label_encoder.fit(all_pictograms)

def multilabel_to_tensor(data_list):
    y_list = []
    for d in data_list:
        labels = label_encoder.transform(d['pictogram_ids'])
        multi_hot = [0]*len(all_pictograms)
        for l in labels:
            multi_hot[l] = 1
        y_list.append(multi_hot)
    return torch.tensor(y_list, dtype=torch.float32)

y = multilabel_to_tensor(dataset)

# ------------------------------
# 4. Modelo y entrenamiento
# ------------------------------
model = PictogramMLP(input_dim=X.shape[1], output_dim=len(all_pictograms))

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

split_val = int(0.9 * len(X))
X_tr, X_val = X[:split_val], X[split_val:]
y_tr, y_val = y[:split_val], y[split_val:]

best_val_loss = float('inf')
patience = 10
counter = 0
num_epochs = 100

for epoch in range(num_epochs):
    model.train()
    perm = torch.randperm(X_tr.size(0))
    X_tr, y_tr = X_tr[perm], y_tr[perm]

    optimizer.zero_grad()
    outputs = model(X_tr)
    loss = criterion(outputs, y_tr)
    loss.backward()
    optimizer.step()

    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        counter = 0
        torch.save(model.state_dict(), "artifacts/best_model.pth")
    else:
        counter += 1

    if counter >= patience:
        print(f"Early stopping en epoch {epoch+1}")
        break

    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")


with open("artifacts/vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open("artifacts/label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)

with open("artifacts/personal_libraries.pkl", "wb") as f:
    pickle.dump(personal_libraries, f)

with open("artifacts/global_library.pkl", "wb") as f:
    pickle.dump(global_library, f)

print("Entrenamiento finalizado y artefactos guardados en /artifacts")
