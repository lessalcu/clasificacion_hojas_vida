# Clasificación de Hojas de Vida con Machine Learning

Sistema backend para apoyar la evaluación y clasificación automática de hojas de vida mediante técnicas de Machine Learning.

El proyecto permite procesar perfiles de candidatos, consolidar información relevante, construir un dataset de entrenamiento y entrenar modelos de clasificación para comparar candidatos contra un perfil de puesto.

---

## Descripción general

El sistema tiene como objetivo apoyar el proceso de preselección de candidatos, reduciendo el tiempo de revisión manual y aportando trazabilidad al análisis inicial.

El flujo general contempla:

- Registro o carga de candidatos.
- Procesamiento de hojas de vida en PDF.
- Consolidación de perfiles de candidatos.
- Construcción de dataset de entrenamiento.
- Entrenamiento y evaluación de modelos de clasificación.
- Serialización y almacenamiento de modelos entrenados.

---

## Tecnologías utilizadas

### Backend

- Python
- Flask
- Flask-CORS
- python-dotenv

### Machine Learning

- scikit-learn
- TF-IDF
- KNN
- Árbol de Clasificación
- joblib

### Base de datos y almacenamiento

- Supabase
- PostgreSQL
- Supabase Storage

### Herramientas de desarrollo

- Visual Studio Code
- Postman
- Git
- GitHub

---

## Configuración del entorno

Crear un archivo `.env` en la raíz del proyecto.

Ejemplo:

```env
FLASK_ENV=development
HOST=127.0.0.1
PORT=5000
FRONTEND_URL=http://localhost:3000

SUPABASE_URL=TU_SUPABASE_URL
SUPABASE_SECRET_KEY=TU_SUPABASE_SECRET_KEY
SUPABASE_SERVICE_ROLE_KEY=TU_SUPABASE_SERVICE_ROLE_KEY
SUPABASE_PROJECT_REF=TU_SUPABASE_PROJECT_REF

MODEL_ARTIFACTS_DIR=artifacts
MODEL_ARTIFACT_BUCKET=model-artifacts
REPORTS_BUCKET=reports

TRAIN_TEST_SIZE=0.20
TRAINING_RANDOM_STATE=42
MIN_TRAINING_ROWS=10

TFIDF_MAX_FEATURES=5000
TFIDF_NGRAM_RANGE=1,2

KNN_DEFAULT_NEIGHBORS=5
TREE_MAX_DEPTH=20
TREE_MIN_SAMPLES_LEAF=2

AUTO_BUILD_DATASET_ON_TRAIN=True
DATASET_MIN_TEXT_LENGTH=80
DATASET_AUTO_LABEL_STRATEGY=relative
DATASET_POSITIVE_RATIO=0.35
DATASET_MATCH_THRESHOLD=0.55

CROSS_VALIDATION_ENABLED=True
CROSS_VALIDATION_FOLDS=5
```

> Importante: el archivo `.env` contiene claves sensibles y no debe subirse al repositorio.

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone URL_DEL_REPOSITORIO
cd clasificacion_hojas_vida
```

### 2. Crear entorno virtual

```powershell
python -m venv venv
```

### 3. Activar entorno virtual

```powershell
.\venv\Scripts\activate
```

Si PowerShell bloquea la activación:

```powershell
Set-ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Luego volver a activar:

```powershell
.\venv\Scripts\activate
```

### 4. Instalar dependencias

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5. Ejecutar el proyecto

```powershell
python run.py
```

La API se levantará en:

```text
http://127.0.0.1:5000
```

---

## Verificación básica

Probar en navegador o Postman:

```http
GET http://127.0.0.1:5000/
```

Respuesta esperada:

```json
{
  "success": true,
  "message": "Backend base initialized"
}
```

---

## Consideraciones de seguridad

- No subir el archivo `.env`.
- No subir claves de Supabase al repositorio.
- No subir artefactos generados en `artifacts/`.
- Usar `service_role` únicamente desde el backend.
- Mantener Row Level Security habilitado en Supabase.

---

## `.gitignore` recomendado

```gitignore
.env
venv/
__pycache__/
*.pyc
artifacts/
```
