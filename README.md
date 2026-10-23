# CAA-ML

Prototipo de aprendizaje automático para predecir pictogramas de comunicación aumentativa y alternativa (CAA) a partir de una frase y del identificador de una persona usuaria. El proyecto incluye una API HTTP con FastAPI, un modelo PyTorch y scripts de entrenamiento.

> **Estado del proyecto:** el código de entrenamiento de ejemplo contiene solo dos frases ficticias y no hay un dataset de producción ni un archivo de dependencias. Los artefactos incluidos permiten probar el flujo con el entorno y los datos con los que fueron generados, pero no deben considerarse un modelo validado para uso clínico o de producción.

## Cómo funciona

1. `CountVectorizer` convierte el texto de entrada en un vector de frecuencias de palabras.
2. Una red neuronal multicapa (MLP) implementada en PyTorch produce puntuaciones para los pictogramas conocidos.
3. La API selecciona hasta cinco pictogramas con mayor puntuación y conserva los que están en la biblioteca global o en la biblioteca personal de la persona indicada.

El modelo está definido en [`backend/model.py`](backend/model.py). La carga del modelo y de sus artefactos se realiza en [`backend/load_model.py`](backend/load_model.py), y el endpoint está en [`backend/predictor.py`](backend/predictor.py).

## Requisitos

- Una instalación de Python compatible con las versiones de PyTorch y scikit-learn que elijas.
- PyTorch, scikit-learn, FastAPI y Uvicorn.
- Para los scripts que consultan otro servicio: `requests`.

No se incluye `requirements.txt` ni otra configuración de dependencias. En un entorno virtual, instala las dependencias necesarias:

```bash
python -m pip install torch scikit-learn fastapi uvicorn requests
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

El script construye el vectorizador, el codificador de etiquetas y las bibliotecas personal/global a partir de una lista `dataset` definida en el propio archivo, y guarda los artefactos en `artifacts/`. Sustituye los ejemplos por datos representativos antes de entrenar. El directorio `artifacts/` debe existir.

El dataset de ejemplo tiene solo dos filas; con la división 90/10 actual queda una sola fila de entrenamiento, que no es suficiente para la capa `BatchNorm1d`. Por tanto, el ejemplo no debe considerarse ejecutable o entrenado de forma útil sin ampliar y validar el dataset.

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
- [`scripts/auto_incremental_train.py`](scripts/auto_incremental_train.py): intenta obtener frases de `http://localhost:4000/api/sentences/all` y actualizar los artefactos.
- [`scripts/auto_train.py`](scripts/auto_train.py): ejemplo de reentrenamiento desde el servicio externo configurado en `http://localhost:4000/api/sentences/model`.
- [`scripts/update_artifacts.py`](scripts/update_artifacts.py): archivo auxiliar actualmente vacío.

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
└── README.md
```

## Consideraciones

- El cargador usa rutas relativas a la raíz del repositorio; inicia el servidor desde esa ubicación.
- Los archivos `.pkl` se cargan con `pickle`. Carga únicamente artefactos de una fuente confiable: deserializar archivos pickle no confiables puede ejecutar código.
- El conjunto de datos de ejemplo es demasiado pequeño para evaluar la calidad o generalización del modelo. Evalúa el modelo con datos representativos antes de cualquier uso real.
- El repositorio no declara una versión mínima de Python; verifica la compatibilidad de las dependencias con la versión que elijas.
- Este repositorio no incluye pruebas automatizadas, autenticación ni configuración de despliegue.
