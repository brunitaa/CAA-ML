import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from backend.model import PictogramMLP
from itertools import chain
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer

ARTIFACTS_PATH = "artifacts/"

with open(f"{ARTIFACTS_PATH}vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

with open(f"{ARTIFACTS_PATH}label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

with open(f"{ARTIFACTS_PATH}personal_libraries.pkl", "rb") as f:
    personal_libraries = pickle.load(f)

with open(f"{ARTIFACTS_PATH}global_library.pkl", "rb") as f:
    global_library = pickle.load(f)

# Cada elemento contiene input_text, pictogram_ids y speaker_id.
new_data = [
    {"input_text": "nuevo pictograma", "pictogram_ids": ["2001"], "speaker_id": "speaker1"},
]

new_words = list(set(chain.from_iterable([d['input_text'].split() for d in new_data])))
all_words = list(vectorizer.vocabulary_.keys()) + [w for w in new_words if w not in vectorizer.vocabulary_]
vectorizer = CountVectorizer(vocabulary=all_words)

new_pictos = list(set(chain.from_iterable([d['pictogram_ids'] for d in new_data])))
all_labels = list(label_encoder.classes_) + [p for p in new_pictos if p not in label_encoder.classes_]

new_label_encoder = LabelEncoder()
new_label_encoder.fit(all_labels)

input_dim = len(vectorizer.vocabulary_)
output_dim = len(new_label_encoder.classes_)

old_model = PictogramMLP(input_dim=input_dim, output_dim=len(label_encoder.classes_))
old_model.load_state_dict(torch.load(f"{ARTIFACTS_PATH}best_model.pth", map_location="cpu"))
old_model.eval()

model = PictogramMLP(input_dim=input_dim, output_dim=output_dim)

with torch.no_grad():
    model.fc1.weight[:, :old_model.fc1.in_features] = old_model.fc1.weight
    model.fc1.bias = old_model.fc1.bias
    model.bn1.weight = old_model.bn1.weight
    model.bn1.bias = old_model.bn1.bias
    model.fc2.weight[:len(label_encoder.classes_)] = old_model.fc2.weight
    model.fc2.bias[:len(label_encoder.classes_)] = old_model.fc2.bias

def multilabel_to_tensor(data_list):
    y_list = []
    for d in data_list:
        labels = new_label_encoder.transform(d['pictogram_ids'])
        multi_hot = [0]*len(new_label_encoder.classes_)
        for l in labels:
            multi_hot[l] = 1
        y_list.append(multi_hot)
    return torch.tensor(y_list, dtype=torch.float32)

X_train = vectorizer.transform([d['input_text'] for d in new_data]).toarray()
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = multilabel_to_tensor(new_data)

criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

num_epochs = 30  # puedes ajustar
for epoch in range(num_epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

torch.save(model.state_dict(), f"{ARTIFACTS_PATH}best_model.pth")

with open(f"{ARTIFACTS_PATH}vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

with open(f"{ARTIFACTS_PATH}label_encoder.pkl", "wb") as f:
    pickle.dump(new_label_encoder, f)

print("Entrenamiento incremental completado. Artefactos actualizados.")
