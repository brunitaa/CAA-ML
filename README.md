# CAA-ML

Prototipo de aprendizaje automático para predecir pictogramas de comunicación aumentativa y alternativa (CAA) a partir de una frase y del identificador de una persona usuaria. El proyecto incluye una API HTTP con FastAPI, un modelo PyTorch y scripts de entrenamiento.

> **Datos y estado del proyecto:** el dataset de ejemplo está incluido directamente en `train/train_model.py` como una lista de 153 registros; no se almacena en un archivo independiente. Los scripts automáticos también pueden obtener datos de un backend externo. Los artefactos incluidos no deben considerarse un modelo validado para uso clínico o de producción.

## Cómo funciona

1. `CountVectorizer` convierte el texto de entrada en un vector de frecuencias de palabras.
2. Una red neuronal multicapa (MLP) implementada en PyTorch produce puntuaciones para los pictogramas conocidos.
3. La API selecciona hasta cinco pictogramas con mayor puntuación y conserva los que están en la biblioteca global o en la biblioteca personal de la persona indicada.

El modelo está definido en [`backend/model.py`](backend/model.py). La carga del modelo y de sus artefactos se realiza en [`backend/load_model.py`](backend/load_model.py), y el endpoint está en [`backend/predictor.py`](backend/predictor.py).

## Requisitos

- Una instalación de Python compatible con las versiones de PyTorch y scikit-learn que elijas.
- PyTorch, scikit-learn, FastAPI y Uvicorn.
- Para los scripts que consultan otro servicio: `requests`.

Instala las dependencias desde la raíz del repositorio:

```bash
python -m pip install -r requirements.txt
```

La instalación de PyTorch puede variar según el sistema operativo y si se desea usar GPU; consulta las instrucciones de instalación de PyTorch para elegir la variante adecuada.

## Ejecutar la API

Desde la raíz del repositorio, inicia el servidor:

```bash
python -m uvicorn backend.predictor:app --reload --host 127.0.0.1 --port 8000
```

La API carga los archivos de `artifacts/` al iniciar. Es necesario ejecutar el comando desde la raíz del proyecto y tener disponibles los cinco artefactos:

- `best_model.pth`: pesos del modelo.
- `vectorizer.pkl`: vocabulario y configuración del vectorizador.
- `label_encoder.pkl`: correspondencia entre etiquetas e identificadores de pictogramas.
- `personal_libraries.pkl`: pictogramas disponibles por persona.
- `global_library.pkl`: pictogramas globales disponibles.

### Endpoint de predicción

`POST /predict`

Solicitud JSON:

```json
{
  "text": "quiero beber agua",
  "speaker_id": 18
}
```

Respuesta:

```json
{
  "predictions": ["1120", "1146"]
}
```

La lista puede contener hasta cinco etiquetas y puede estar vacía si ninguna de las cinco etiquetas con mayor puntuación pertenece a la biblioteca aplicable. La documentación interactiva de FastAPI está disponible en `http://127.0.0.1:8000/docs` mientras el servidor está activo.

Ejemplo de llamada:

```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"text":"quiero beber agua","speaker_id":18}'
```

## Entrenamiento

### Entrenamiento inicial

El punto de entrada es [`train/train_model.py`](train/train_model.py):

```bash
python -m train.train_model
```

El script construye el vectorizador, el codificador de etiquetas y las bibliotecas personal/global a partir de una lista `dataset` definida en el propio archivo, y guarda los artefactos en `artifacts/`. Puedes reemplazar o ampliar esa lista con tu dataset. El directorio `artifacts/` debe existir.

El dataset de ejemplo está definido directamente en el archivo de entrenamiento. Revisa su tamaño y distribución antes de entrenar; el script reserva el 10 % de los datos para validación y usa una capa `BatchNorm1d`, por lo que se requiere un conjunto de entrenamiento con suficientes ejemplos.

Cada elemento del dataset tiene esta forma:

```python
{
    "input_text": "quiero beber agua",
    "pictogram_ids": ["1120", "1146"],
    "speaker_id": "18"
}
```

`pictogram_ids` es una lista porque el entrenamiento inicial trata cada frase como una selección multietiqueta. Mantén coherentes los tipos y valores de los identificadores de persona y pictograma entre los datos, los artefactos y las solicitudes a la API.

### Scripts adicionales

- [`scripts/train_incremental.py`](scripts/train_incremental.py): ejemplo de entrenamiento incremental con datos editados directamente en el script.
- [`scripts/auto_incremental_train.py`](scripts/auto_incremental_train.py): obtiene frases de una URL configurada mediante la variable `CAA_INCREMENTAL_TRAIN_URL` y actualiza los artefactos. La respuesta debe ser una lista de registros con `input_text`, `pictogram_ids` y, opcionalmente, `speaker_id`.
- [`scripts/auto_train.py`](scripts/auto_train.py): obtiene pictogramas de la URL configurada mediante `CAA_AUTO_TRAIN_URL`. Esa URL debe apuntar al endpoint que devuelve un objeto con las listas `global` y `personal`; cada elemento debe incluir `example_text` e `id`.

Configura las variables de entorno con las direcciones de tu servicio antes de ejecutar estos scripts. No se incluyen direcciones de backend en el código. En PowerShell, por ejemplo:

```powershell
$env:CAA_AUTO_TRAIN_URL = "https://tu-servidor/endpoint"
python -m scripts.auto_train

$env:CAA_INCREMENTAL_TRAIN_URL = "https://tu-servidor/endpoint"
python -m scripts.auto_incremental_train
```

Los scripts incrementales y automáticos son prototipos y no forman un flujo de entrenamiento de producción. En particular, los automáticos requieren que exista y responda el servicio externo con el formato esperado por el código. Revisa y valida los datos, las dimensiones del modelo y los artefactos resultantes antes de utilizarlos con la API.

## Estructura del proyecto

```text
.
├── artifacts/                 # Modelo y objetos serializados usados en inferencia
├── backend/
│   ├── load_model.py          # Carga de los artefactos
│   ├── model.py               # Definición de la red neuronal
│   └── predictor.py           # API FastAPI y endpoint /predict
├── scripts/                   # Ejemplos de entrenamiento y actualización
├── train/
│   └── train_model.py         # Entrenamiento inicial de ejemplo
├── requirements.txt           # Dependencias de Python
└── README.md
```

## Consideraciones

- El cargador usa rutas relativas a la raíz del repositorio; inicia el servidor desde esa ubicación.
- Los archivos `.pkl` se cargan con `pickle`. Carga únicamente artefactos de una fuente confiable: deserializar archivos pickle no confiables puede ejecutar código.
- El dataset incluido es de ejemplo y no demuestra la calidad ni la generalización del modelo. Evalúa el modelo con datos representativos antes de cualquier uso real.
- El repositorio no declara una versión mínima de Python; verifica la compatibilidad de las dependencias con la versión que elijas.
- Este repositorio no incluye pruebas automatizadas, autenticación ni configuración de despliegue.
