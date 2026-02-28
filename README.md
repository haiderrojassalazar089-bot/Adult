# End-to-End MLOps Pipeline – Drift Monitoring & Auto Retrain

Proyecto de Machine Learning implementando un pipeline completo bajo principios de MLOps, incluyendo entrenamiento, persistencia de artefactos, monitoreo de drift estadístico y retraining automático.

---

## Objetivo

Construir un sistema que:

- Entrene un modelo supervisado  
- Guarde artefactos reproducibles  
- Genere predicciones en producción  
- Registre datos de entrada  
- Detecte drift estadístico  
- Ejecute retraining automático cuando se supere un umbral definido  

---

## Ciclo de Vida MLOps Implementado

1. Ingesta de datos  
2. Validación y calidad de datos  
3. Feature Engineering  
4. Entrenamiento del modelo  
5. Registro de artefactos  
6. Esquemas de entrada para producción  
7. Monitoreo de drift  
8. Retraining automático  

---

## 1. Ingesta y Validación de Datos

### Resultados del análisis inicial

- Total filas: 48,842  
- Total columnas: 15  
- Duplicados detectados: 0  
- Clases objetivo: `<=50K` y `>50K`  

### Valores nulos detectados

- `workclass`: 963  
- `occupation`: 966  
- `native-country`: 274  

Archivo generado automáticamente: `artifacts/ingestion_report.json`

### Estadísticas de Variables Numéricas

- `age`: promedio 38,6 años (17-90), 216 outliers  
- `fnlwgt`: promedio 189.664, 1.453 outliers  
- `education-num`: promedio 10, 1.794 outliers  
- `capital-gain` y `capital-loss`: outliers esperables  
- `hours-per-week`: promedio 40,4 horas, 13.496 outliers  

### Variables Categóricas

- Cardinalidad razonable en todas las columnas  
- Algunas categorías raras (<1%) en `workclass`, `education`, `marital-status`, `occupation`, `race` y `native-country`  
- Categorías raras se podrán agrupar o codificar durante el preprocesamiento  

### Análisis del Target (`income`)

- `<=50K`: 76,1%  
- `>50K`: 23,9%  
- Desbalance moderado (3,18 veces más registros de la clase `<=50K`)  
- Entropía 0,55  

> El dataset es confiable, consistente y listo para procesamiento y entrenamiento.

---

## 2. Feature Engineering

### Transformaciones realizadas

- Identificación automática de variables numéricas y categóricas  
- Eliminación de columnas innecesarias  
- Imputación de valores faltantes:
  - Numéricas: mediana  
  - Categóricas: categoría más frecuente  
- Escalado de variables numéricas con StandardScaler  
- Codificación de variables categóricas con One-Hot Encoding, manejando categorías desconocidas  
- Registro de metadata y nombres de features para trazabilidad  

### Resultados

- Filas procesadas: 48,842  
- Features originales: 14 (6 numéricas, 8 categóricas)  
- Features finales: 111 (105 generadas por One-Hot Encoding)  

> Datos consistentes y listos para entrenamiento y producción, con manejo robusto de valores faltantes y categorías desconocidas.

---

## 3. Entrenamiento del Modelo

### Acciones realizadas

- Ajuste de modelo supervisado con datos procesados  
- Evaluación con conjunto de prueba  
- Guardado de artefactos generados  

### Métricas de Evaluación

| Métrica    | Valor |
|------------|-------|
| Accuracy   | 0.808 |
| Precision  | 0.568 |
| Recall     | 0.831 |
| F1-score   | 0.675 |

### Matriz de Confusión

|                 | Predicted <=50K | Predicted >50K |
|-----------------|----------------|----------------|
| Actual <=50K    | 5.951          | 1.480          |
| Actual >50K     | 395            | 1.943          |

### Artefactos Generados

- `model.pkl` (modelo entrenado)  
- `scaler.pkl` (normalización)  
- `train_reference.parquet` (baseline para drift)  

> El modelo es robusto y confiable para producción, con artefactos listos para predicción, monitoreo y retraining automático.

---

## 4. Registro de Artefactos

### Qué se hizo

- Versionado automático de artefactos en cada ejecución  
- Copiado seguro de:
  - Preprocesador (`preprocessor.joblib`)  
  - Métricas y metadata (`training_metrics.json`, `training_metadata.json`)  
  - Metadata de feature engineering (`feature_engineering_metadata.json`)  
  - Reportes de validación e ingestión (`validation_report.json`, `ingestion_report.json`)  
- Metadata de ejecución: `run_id`, timestamp UTC y lista de artefactos registrados  

> Garantiza reproducibilidad, trazabilidad y auditoría completa del pipeline.

---

## 5. Esquemas de Entrada (Despliegue)

- Validación de tipo y rango de todas las variables (`age` 17-90, `education_num` 1-16)  
- Inputs malformados o fuera de rango son rechazados antes de llegar al modelo  
- Contrato de datos asegura consistencia y confiabilidad de predicciones  

> Los esquemas de entrada mantienen integridad y reproducibilidad del pipeline en producción.

---

## 6. Monitoreo de Drift y Retraining Automático

- Comparación de distribuciones de producción vs baseline de entrenamiento  
- Detección de drift usando test Kolmogorov-Smirnov en variables numéricas  
- Umbral: p-value <0.05 indica drift  
- Retraining automático si se detecta drift, actualizando modelo, scaler y baseline  

### Ejemplo de funcionamiento

- Columna `feature1` | p-value: 0.23145 → No hay drift  
- Columna `feature2` | p-value: 0.01234 → Drift detectado  
- Se ejecuta retraining automático y se actualizan artefactos  

> Mantiene el modelo confiable frente a cambios en la distribución de los datos.

---

## Conclusión General

Este pipeline integra todas las etapas necesarias para un flujo de Machine Learning profesional:

- Datos confiables y reproducibles para entrenamiento y producción  
- Modelo robusto, auditado y con métricas equilibradas frente a desbalance de clases  
- Artefactos versionados para trazabilidad y reproducibilidad  
- Validación estricta de inputs que asegura integridad en producción  
- Monitoreo continuo de drift con retraining automático  

> Garantiza un ciclo de vida ML confiable, escalable y alineado con las mejores prácticas de MLOps, asegurando consistencia, precisión y adaptabilidad del modelo en producción.