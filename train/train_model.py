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

dataset = [
    {"input_text": "yo quiero jugar", "pictogram_ids": ["1119","1133","1182"], "speaker_id": "speaker1"},
{"input_text": "tú quieres beber agua", "pictogram_ids": ["1120","1133","1146","1159"], "speaker_id": "speaker2"},
{"input_text": "él va al parque", "pictogram_ids": ["1121","1135","1196"], "speaker_id": "speaker3"},
{"input_text": "ella lee un libro", "pictogram_ids": ["1122","1151","1171"], "speaker_id": "speaker4"},
{"input_text": "nosotros comemos pan", "pictogram_ids": ["1123","1145","1166"], "speaker_id": "speaker1"},
{"input_text": "nosotras bebemos leche", "pictogram_ids": ["1124","1146","1161"], "speaker_id": "speaker2"},
{"input_text": "ellos juegan con la pelota", "pictogram_ids": ["1125","1148","1169"], "speaker_id": "speaker3"},
{"input_text": "ellas corren en el jardín", "pictogram_ids": ["1126","1149","1201"], "speaker_id": "speaker4"},
{"input_text": "mi papá trabaja mucho", "pictogram_ids": ["1128","1156"], "speaker_id": "speaker4"},
{"input_text": "mi amigo estudia", "pictogram_ids": ["1129","1157"], "speaker_id": "speaker2"},
{"input_text": "mi profesora enseña", "pictogram_ids": ["1130","1158"], "speaker_id": "speaker2"},
{"input_text": "mi hermano juega fútbol", "pictogram_ids": ["1131","1148"], "speaker_id": "speaker2"},
{"input_text": "mi perro corre", "pictogram_ids": ["1132","1149"], "speaker_id": "speaker2"},
{"input_text": "quiero beber jugo", "pictogram_ids": ["1133","1146","1162"], "speaker_id": "speaker1"},
{"input_text": "no quiero comer galleta", "pictogram_ids": ["1134","1145","1164"], "speaker_id": "speaker1"},
{"input_text": "voy a la playa", "pictogram_ids": ["1135","1136","1208"], "speaker_id": "speaker2"},
{"input_text": "me gusta la música", "pictogram_ids": ["1137","1181"], "speaker_id": "speaker1"},
{"input_text": "necesito agua", "pictogram_ids": ["1138","1159"], "speaker_id": "speaker1"},
{"input_text": "tengo hambre", "pictogram_ids": ["1139","1160"], "speaker_id": "speaker2"},
{"input_text": "veo la televisión", "pictogram_ids": ["1140","1172"], "speaker_id": "speaker1"},
{"input_text": "escucho música", "pictogram_ids": ["1141","1181"], "speaker_id": "speaker3"},
{"input_text": "siento frío", "pictogram_ids": ["1142","1216"], "speaker_id": "speaker3"},
{"input_text": "pienso en la escuela", "pictogram_ids": ["1143","1194"], "speaker_id": "speaker2"},
{"input_text": "digo la verdad", "pictogram_ids": ["1144"], "speaker_id": "speaker1"},
{"input_text": "como pan y fruta", "pictogram_ids": ["1145","1166","1168"], "speaker_id": "speaker1"},
{"input_text": "bebo leche", "pictogram_ids": ["1146","1161"], "speaker_id": "speaker4"},
{"input_text": "uso computadora", "pictogram_ids": ["1147","1174"], "speaker_id": "speaker4"},
{"input_text": "juego con la muñeca", "pictogram_ids": ["1148","1170"], "speaker_id": "speaker1"},
{"input_text": "corro en el parque", "pictogram_ids": ["1149","1196"], "speaker_id": "speaker3"},
{"input_text": "duermo en la cama", "pictogram_ids": ["1150","1184"], "speaker_id": "speaker2"},
{"input_text": "leo un libro", "pictogram_ids": ["1151","1171"], "speaker_id": "speaker2"},
{"input_text": "pinto con colores", "pictogram_ids": ["1152"], "speaker_id": "speaker3"},
{"input_text": "ayudo a mi amigo", "pictogram_ids": ["1153","1129"], "speaker_id": "speaker1"},
{"input_text": "camino en la calle", "pictogram_ids": ["1154","1200"], "speaker_id": "speaker3"},
{"input_text": "salto la cuerda", "pictogram_ids": ["1155"], "speaker_id": "speaker1"},
{"input_text": "trabajo en la computadora", "pictogram_ids": ["1156","1174"], "speaker_id": "speaker3"},
{"input_text": "estudio en la biblioteca", "pictogram_ids": ["1157","1206"], "speaker_id": "speaker4"},
{"input_text": "aprendo a leer", "pictogram_ids": ["1158","1234"], "speaker_id": "speaker1"},
{"input_text": "bebo agua", "pictogram_ids": ["1146","1159"], "speaker_id": "speaker4"},
{"input_text": "como arroz", "pictogram_ids": ["1145","1167"], "speaker_id": "speaker1"},
{"input_text": "tengo hambre", "pictogram_ids": ["1139","1160"], "speaker_id": "speaker3"},
{"input_text": "veo el sol", "pictogram_ids": ["1140","1241"], "speaker_id": "speaker3"},
{"input_text": "escucho al pájaro", "pictogram_ids": ["1141","1220"], "speaker_id": "speaker1"},
{"input_text": "siento feliz", "pictogram_ids": ["1142","1210"], "speaker_id": "speaker1"},
{"input_text": "pienso en el mar", "pictogram_ids": ["1143","1233"], "speaker_id": "speaker3"},
{"input_text": "digo hola", "pictogram_ids": ["1144"], "speaker_id": "speaker1"},
{"input_text": "como pollo", "pictogram_ids": ["1145","1169"], "speaker_id": "speaker4"},
{"input_text": "bebo jugo", "pictogram_ids": ["1146","1162"], "speaker_id": "speaker1"},
{"input_text": "uso la computadora portátil", "pictogram_ids": ["1147","1238"], "speaker_id": "speaker1"},
{"input_text": "juego con el lego", "pictogram_ids": ["1148","1268"], "speaker_id": "speaker4"},
{"input_text": "corro rápido", "pictogram_ids": ["1149"], "speaker_id": "speaker1"},
{"input_text": "duermo en la habitación", "pictogram_ids": ["1150","1205"], "speaker_id": "speaker1"},
{"input_text": "leo la revista", "pictogram_ids": ["1151"], "speaker_id": "speaker4"},
{"input_text": "pinto un dibujo", "pictogram_ids": ["1152"], "speaker_id": "speaker1"},
{"input_text": "ayudo a mi mamá", "pictogram_ids": ["1153","1127"], "speaker_id": "speaker1"},
{"input_text": "camino al bosque", "pictogram_ids": ["1154","1297"], "speaker_id": "speaker3"},
{"input_text": "salto en la plaza", "pictogram_ids": ["1155","1296"], "speaker_id": "speaker1"},
{"input_text": "trabajo en la tienda", "pictogram_ids": ["1156","1291"], "speaker_id": "speaker4"},
{"input_text": "estudio en la escuela", "pictogram_ids": ["1157","1194"], "speaker_id": "speaker1"},
{"input_text": "aprendo a escribir", "pictogram_ids": ["1158","1235"], "speaker_id": "speaker1"},
{"input_text": "bebo leche y jugo", "pictogram_ids": ["1146","1161","1162"], "speaker_id": "speaker1"},
{"input_text": "como pan y galleta", "pictogram_ids": ["1145","1166","1164"], "speaker_id": "speaker2"},
{"input_text": "tengo sed", "pictogram_ids": ["1139"], "speaker_id": "speaker2"},
{"input_text": "veo la luna", "pictogram_ids": ["1140","1242"], "speaker_id": "speaker1"},
{"input_text": "escucho música y canto", "pictogram_ids": ["1141","1181"], "speaker_id": "speaker1"},
{"input_text": "siento cansado", "pictogram_ids": ["1142","1214"], "speaker_id": "speaker1"},
{"input_text": "pienso en la montaña", "pictogram_ids": ["1143","1232"], "speaker_id": "speaker1"},
{"input_text": "digo adiós", "pictogram_ids": ["1144"], "speaker_id": "speaker1"},
{"input_text": "como fruta y pollo", "pictogram_ids": ["1145","1168","1169"], "speaker_id": "speaker1"},
{"input_text": "bebo agua y leche", "pictogram_ids": ["1146","1159","1161"], "speaker_id": "speaker1"},
{"input_text": "uso el teléfono", "pictogram_ids": ["1147","1173"], "speaker_id": "speaker1"},
{"input_text": "juego con la pelota y el títere", "pictogram_ids": ["1148","1169","1272"], "speaker_id": "speaker1"},
{"input_text": "corro en la calle", "pictogram_ids": ["1149","1200"], "speaker_id": "speaker1"},
{"input_text": "duermo en la cama y la habitación", "pictogram_ids": ["1150","1184","1205"], "speaker_id": "speaker1"},
{"input_text": "leo un libro y escribo", "pictogram_ids": ["1151","1171","1235"], "speaker_id": "speaker1"},
{"input_text": "pinto un dibujo y aprendo", "pictogram_ids": ["1152","1158"], "speaker_id": "speaker1"},
{"input_text": "ayudo a mi hermano", "pictogram_ids": ["1153","1131"], "speaker_id": "speaker1"},
{"input_text": "camino por la calle y el jardín", "pictogram_ids": ["1154","1200","1201"], "speaker_id": "speaker1"},
    {"input_text": "yo veo la luna", "pictogram_ids": ["1119","1140","1242"], "speaker_id": "speaker1"},
    {"input_text": "tú escuchas música", "pictogram_ids": ["1120","1141","1181"], "speaker_id": "speaker1"},
    {"input_text": "él corre en la plaza", "pictogram_ids": ["1121","1149","1204"], "speaker_id": "speaker1"},
    {"input_text": "ella juega con el lego", "pictogram_ids": ["1122","1148","1268"], "speaker_id": "speaker1"},
    {"input_text": "nosotros bebemos jugo", "pictogram_ids": ["1123","1146","1162"], "speaker_id": "speaker1"},
    {"input_text": "nosotras comemos fruta", "pictogram_ids": ["1124","1145","1165"], "speaker_id": "speaker1"},
    {"input_text": "ellos duermen en la cama", "pictogram_ids": ["1125","1150","1184"], "speaker_id": "speaker1"},
    {"input_text": "ellas leen un libro", "pictogram_ids": ["1126","1151","1171"], "speaker_id": "speaker1"},
    {"input_text": "mi papá trabaja en la tienda", "pictogram_ids": ["1128","1156","1197"], "speaker_id": "speaker1"},
    {"input_text": "mi mamá ayuda a mi amigo", "pictogram_ids": ["1127","1153","1129"], "speaker_id": "speaker1"},
    {"input_text": "mi amigo estudia en la biblioteca", "pictogram_ids": ["1129","1157","1206"], "speaker_id": "speaker1"},
    {"input_text": "mi profesor enseña en la escuela", "pictogram_ids": ["1130","1158","1194"], "speaker_id": "speaker1"},
    {"input_text": "mi hermano juega con la pelota", "pictogram_ids": ["1131","1148","1169"], "speaker_id": "speaker1"},
    {"input_text": "mi perro corre en el jardín", "pictogram_ids": ["1132","1149","1201"], "speaker_id": "speaker1"},
    {"input_text": "quiero beber leche", "pictogram_ids": ["1133","1146","1161"], "speaker_id": "speaker1"},
    {"input_text": "no quiero comer galleta", "pictogram_ids": ["1134","1145","1164"], "speaker_id": "speaker1"},
    {"input_text": "voy a la playa", "pictogram_ids": ["1135","1136","1208"], "speaker_id": "speaker1"},
    {"input_text": "me gusta la música", "pictogram_ids": ["1137","1181"], "speaker_id": "speaker1"},
    {"input_text": "necesito agua", "pictogram_ids": ["1138","1159"], "speaker_id": "speaker1"},
    {"input_text": "tengo pan y jugo", "pictogram_ids": ["1139","1163","1162"], "speaker_id": "speaker1"},
    {"input_text": "veo la televisión", "pictogram_ids": ["1140","1172"], "speaker_id": "speaker1"},
    {"input_text": "escucho música y canto", "pictogram_ids": ["1141","1181"], "speaker_id": "speaker1"},
    {"input_text": "siento cansado", "pictogram_ids": ["1142","1214"], "speaker_id": "speaker1"},
    {"input_text": "pienso en la montaña", "pictogram_ids": ["1143","1232"], "speaker_id": "speaker1"},
    {"input_text": "digo hola", "pictogram_ids": ["1144"], "speaker_id": "speaker2"},
    {"input_text": "como pollo y arroz", "pictogram_ids": ["1145","1166","1167"], "speaker_id": "speaker1"},
    {"input_text": "bebo agua y jugo", "pictogram_ids": ["1146","1159","1162"], "speaker_id": "speaker3"},
    {"input_text": "uso la computadora portátil", "pictogram_ids": ["1147","1238"], "speaker_id": "speaker1"},
    {"input_text": "juego con la muñeca y el títere", "pictogram_ids": ["1148","1170","1272"], "speaker_id": "speaker1"},
    {"input_text": "corro en la calle", "pictogram_ids": ["1149","1200"], "speaker_id": "speaker1"},
    {"input_text": "duermo en la cama y habitación", "pictogram_ids": ["1150","1184","1205"], "speaker_id": "speaker1"},
    {"input_text": "leo un libro y escribo", "pictogram_ids": ["1151","1171","1235"], "speaker_id": "speaker3"},
    {"input_text": "pinto un dibujo y aprendo", "pictogram_ids": ["1152","1158"], "speaker_id": "speaker4"},
    {"input_text": "ayudo a mi hermano", "pictogram_ids": ["1153","1131"], "speaker_id": "speaker4"},
    {"input_text": "camino por la calle y el jardín", "pictogram_ids": ["1154","1200","1201"], "speaker_id": "speaker1"},
    {"input_text": "salto en la plaza", "pictogram_ids": ["1155","1296"], "speaker_id": "speaker1"},
    {"input_text": "trabajo en la computadora", "pictogram_ids": ["1156","1174"], "speaker_id": "speaker1"},
    {"input_text": "estudio en la escuela", "pictogram_ids": ["1157","1194"], "speaker_id": "speaker1"},
    {"input_text": "aprendo a leer", "pictogram_ids": ["1158","1234"], "speaker_id": "speaker1"},
    {"input_text": "bebo leche y jugo", "pictogram_ids": ["1146","1161","1162"], "speaker_id": "speaker1"},
    {"input_text": "como pan y galleta", "pictogram_ids": ["1145","1163","1164"], "speaker_id": "speaker1"},
    {"input_text": "tengo sed", "pictogram_ids": ["1139"], "speaker_id": "speaker2"},
    {"input_text": "veo la luna y la estrella", "pictogram_ids": ["1140","1242","1240"], "speaker_id": "speaker4"},
    {"input_text": "escucho al pájaro y al perro", "pictogram_ids": ["1141","1220","1218"], "speaker_id": "speaker1"},
    {"input_text": "siento feliz y emocionado", "pictogram_ids": ["1142","1210","1215"], "speaker_id": "speaker4"},
    {"input_text": "pienso en el mar y la montaña", "pictogram_ids": ["1143","1233","1232"], "speaker_id": "speaker1"},
    {"input_text": "digo adiós", "pictogram_ids": ["1144"], "speaker_id": "speaker4"},
    {"input_text": "como fruta y pollo", "pictogram_ids": ["1145","1165","1166"], "speaker_id": "speaker4"},
    {"input_text": "bebo agua y leche", "pictogram_ids": ["1146","1159","1161"], "speaker_id": "speaker2"},
    {"input_text": "uso el teléfono y la computadora", "pictogram_ids": ["1147","1173","1174"], "speaker_id": "speaker1"},
    {"input_text": "juego con la pelota", "pictogram_ids": ["1148","1169"], "speaker_id": "speaker1"},
    {"input_text": "corro en el parque", "pictogram_ids": ["1149","1196"], "speaker_id": "speaker1"},
    {"input_text": "duermo en la habitación", "pictogram_ids": ["1150","1205"], "speaker_id": "speaker4"},
    {"input_text": "leo la revista", "pictogram_ids": ["1151"], "speaker_id": "speaker2"},
    {"input_text": "pinto un dibujo", "pictogram_ids": ["1152"], "speaker_id": "speaker4"},
    {"input_text": "ayudo a mi mamá", "pictogram_ids": ["1153","1127"], "speaker_id": "speaker1"},
    {"input_text": "camino al bosque", "pictogram_ids": ["1154","1297"], "speaker_id": "speaker3"},
    {"input_text": "salto la cuerda en la plaza", "pictogram_ids": ["1155","1296"], "speaker_id": "speaker3"},
    {"input_text": "trabajo en la tienda", "pictogram_ids": ["1156","1197"], "speaker_id": "speaker1"},
    {"input_text": "estudio en la biblioteca", "pictogram_ids": ["1157","1206"], "speaker_id": "speaker2"},
    {"input_text": "aprendo a escribir", "pictogram_ids": ["1158","1235"], "speaker_id": "speaker2"},
    {"input_text": "bebo agua, leche y jugo", "pictogram_ids": ["1146","1159","1161","1162"], "speaker_id": "speaker1"},
    {"input_text": "como pan, galleta y arroz", "pictogram_ids": ["1145","1164","1167"], "speaker_id": "speaker3"},
    {"input_text": "tengo hambre y sed", "pictogram_ids": ["1139","1160"], "speaker_id": "speaker3"},
    {"input_text": "veo el sol y la luna", "pictogram_ids": ["1140","1241","1242"], "speaker_id": "speaker3"},
    {"input_text": "escucho música y canto", "pictogram_ids": ["1141","1181"], "speaker_id": "speaker4"},
    {"input_text": "siento cansado y feliz", "pictogram_ids": ["1142","1214","1210"], "speaker_id": "speaker4"},
    {"input_text": "pienso en la escuela y el mar", "pictogram_ids": ["1143","1194","1233"], "speaker_id": "speaker3"},
    {"input_text": "digo hola y adiós", "pictogram_ids": ["1144"], "speaker_id": "speaker2"},
    {"input_text": "como pollo, arroz y fruta", "pictogram_ids": ["1145","1166","1167","1165"], "speaker_id": "speaker2"},
    {"input_text": "bebo agua, leche y jugo", "pictogram_ids": ["1146","1159","1161","1162"], "speaker_id": "speaker1"},
    {"input_text": "uso la computadora portátil", "pictogram_ids": ["1147","1238"], "speaker_id": "speaker4"},
    {"input_text": "juego con la pelota y el títere", "pictogram_ids": ["1148","1169","1272"], "speaker_id": "speaker3"},
    {"input_text": "corro en la calle", "pictogram_ids": ["1149","1200"], "speaker_id": "speaker3"},
    {"input_text": "duermo en la habitación", "pictogram_ids": ["1150","1205"], "speaker_id": "speaker2"},
   ]




personal_libraries = {}
for d in dataset:
    speaker = d["speaker_id"]
    if speaker not in personal_libraries:
        personal_libraries[speaker] = set()
    personal_libraries[speaker].update(d["pictogram_ids"])

global_library = set(chain.from_iterable([d["pictogram_ids"] for d in dataset]))

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
