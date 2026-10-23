import requests
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from backend.model import PictogramMLP
import os

BACKEND_URL = os.environ["CAA_INCREMENTAL_TRAIN_URL"]
ARTIFACTS_PATH = "artifacts/"
SPEAKER_ID_DEFAULT = "all"

os.makedirs(ARTIFACTS_PATH, exist_ok=True)

try:
    with open(f"{ARTIFACTS_PATH}vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
except:
    vectorizer = None

try:
    with open(f"{ARTIFACTS_PATH}label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
except:
    label_encoder = None

try:
    with open(f"{ARTIFACTS_PATH}personal_libraries.pkl", "rb") as f:
        personal_libraries = pickle.load(f)
except:
    personal_libraries = {}

try:
    with open(f"{ARTIFACTS_PATH}global_library.pkl", "rb") as f:
        global_library = pickle.load(f)
except:
    global_library = set()

resp = requests.get(BACKEND_URL, timeout=30)
resp.raise_for_status()
data = resp.json()

if not data:
    print("No hay datos nuevos para entrenar. Fin del script.")
    exit()

new_texts = []
new_pictos = []
new_speakers = []

for d in data:
    pictos_ids = [int(p) for p in d.get('pictogram_ids', [])]
    unseen = any([p not in global_library for p in pictos_ids])
    if unseen or vectorizer is None or label_encoder is None:
        new_texts.append(d.get('input_text', ''))
        new_pictos.append(pictos_ids)
        new_speakers.append(d.get('speaker_id', SPEAKER_ID_DEFAULT))

if not new_texts:
    print("No hay frases nuevas para entrenar. Fin del script.")
    exit()

new_words = list(set(chain.from_iterable([text.split() for text in new_texts])))
if vectorizer:
    all_words = list(vectorizer.vocabulary_.keys()) + [w for w in new_words if w not in vectorizer.vocabulary_]
else:
    all_words = new_words

vectorizer = CountVectorizer(vocabulary=all_words)
X = vectorizer.fit_transform(new_texts).toarray()

all_labels = list(set(chain.from_iterable(new_pictos)))
if label_encoder:
    all_labels = list(label_encoder.classes_) + [l for l in all_labels if l not in label_encoder.classes_]

label_encoder = LabelEncoder()
label_encoder.fit(all_labels)

y_tensor = torch.zeros((len(new_texts), len(label_encoder.classes_)), dtype=torch.float32)
for i, pictos in enumerate(new_pictos):
    indices = label_encoder.transform(pictos)
    for idx in indices:
        y_tensor[i, idx] = 1.0

X_tensor = torch.tensor(X, dtype=torch.float32)

input_dim = X.shape[1]
output_dim = len(label_encoder.classes_)

old_model = None
if os.path.exists(f"{ARTIFACTS_PATH}best_model.pth") and vectorizer and label_encoder:
    try:
        old_model = PictogramMLP(input_dim=input_dim, output_dim=output_dim)
        old_model.load_state_dict(torch.load(f"{ARTIFACTS_PATH}best_model.pth", map_location="cpu"))
        old_model.eval()
    except RuntimeError:
        old_model = None  # si mismatch, ignoramos y entrenamos desde cero

model = PictogramMLP(input_dim=input_dim, output_dim=output_dim)

if old_model:
    with torch.no_grad():
        old_in_features = old_model.fc1.in_features
        model.fc1.weight[:, :old_in_features] = old_model.fc1.weight
        model.fc1.bias = old_model.fc1.bias

        model.bn1.weight = old_model.bn1.weight
        model.bn1.bias = old_model.bn1.bias

        old_out_features_fc2 = old_model.fc2.out_features
        model.fc2.weight[:old_out_features_fc2, :] = old_model.fc2.weight
        model.fc2.bias[:old_out_features_fc2] = old_model.fc2.bias

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 20

for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_tensor)
    loss = criterion(outputs, y_tensor)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}")

global_library.update(all_labels)
for speaker, pictos in zip(new_speakers, new_pictos):
    if speaker in personal_libraries:
        personal_libraries[speaker].update(pictos)
    else:
        personal_libraries[speaker] = set(pictos)

torch.save(model.state_dict(), f"{ARTIFACTS_PATH}best_model.pth")
with open(f"{ARTIFACTS_PATH}vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)
with open(f"{ARTIFACTS_PATH}label_encoder.pkl", "wb") as f:
    pickle.dump(label_encoder, f)
with open(f"{ARTIFACTS_PATH}personal_libraries.pkl", "wb") as f:
    pickle.dump(personal_libraries, f)
with open(f"{ARTIFACTS_PATH}global_library.pkl", "wb") as f:
    pickle.dump(global_library, f)

print("Entrenamiento incremental continuo completado con nuevas frases.")
