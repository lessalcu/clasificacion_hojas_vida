# ENTRENAMIENTO KNN Y ARBOLES

## Qué incluye
- Integración del endpoint de entrenamiento en Flask.
- Servicio para construir el dataframe de entrenamiento a partir de:
  - `dataset_sample`
  - `candidate_profile`
  - `job_profile`
- Entrenamiento de:
  - KNN
  - Árbol de Clasificación
- Evaluación con:
  - Recall
  - F1-Score
  - Precisión
  - Accuracy
  - Matriz de confusión
- Persistencia en:
  - `model_version`
  - `training_run`
- Serialización local con `joblib`
- Subida opcional a Supabase Storage en `model-artifacts` REFORMULAR  OBLIGATORIA

## Endpoint
POST `/api/v1/model-training/job-profiles/<job_profile_id>/train`

Body opcional:
```json
{
  "created_by": null,
  "dataset_version": "v1",
  "persist_to_storage": true
}
```

## Script manual
```powershell
py scripts\train_models.py <job_profile_id>
```

## Variables nuevas sugeridas en `.env`
```env
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
```
