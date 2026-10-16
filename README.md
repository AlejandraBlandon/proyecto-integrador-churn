# Predicción de pago a tiempo de créditos, monitoreo de data drift y despliegue del modelo

Proyecto integrador - Alejandra Blandon.

## 1. Caso de negocio

Rol: Científico de Datos Junior Advanced en el equipo de Datos y Analítica de una empresa financiera.

Objetivo: desarrollar un modelo predictivo de aprendizaje automático supervisado, a partir de información histórica de créditos, para anticipar el comportamiento de nuevos usuarios. La variable objetivo es `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).

Restricción de la empresa: cada proyecto debe respetar una estructura de carpetas definida, porque el paso a producción está automatizado con pipelines de validación en Jenkins. Por eso la estructura no se modificó.

## 2. Datos

- 10.763 registros y 23 variables (`Base_de_datos.xlsx`, convertido a `Base_de_datos.csv` en `Cargar_datos.ipynb`).
- Variable objetivo desbalanceada: 95,3 % paga a tiempo y 4,7 % no paga.
- Los préstamos van de noviembre de 2024 a abril de 2026.

## 3. Estructura del repositorio: qué es cada archivo

Ramas: `main`, `developer` y `certification`. El trabajo se desarrolló en `developer` y se integró a `main` mediante Pull Request.

| Archivo | Ubicación | Para qué sirve |
|---|---|---|
| `Cargar_datos.ipynb` | `mlops_pipeline/src/` | Carga `Base_de_datos.xlsx` y lo convierte a `Base_de_datos.csv`. |
| `comprension_eda.ipynb` | `mlops_pipeline/src/` | Análisis exploratorio (univariable, bivariable y multivariable), calidad de datos y conclusiones. |
| `ft_engineering.py` | `mlops_pipeline/src/` | Ingeniería de características: limpieza, `ColumnTransformer` y división train/test. |
| `model_training_evaluation.py` | `mlops_pipeline/src/` | Funciones `build_model` y `summarize_classification`, entrenamiento y comparación de modelos, y guardado del modelo final. |
| `model_deploy.py` | `mlops_pipeline/src/` | Carga el modelo guardado, define la lógica de predicción y expone la API con FastAPI. |
| `model_monitoring.py` | `mlops_pipeline/src/` | Monitoreo de data drift (KS, PSI, Jensen-Shannon, Chi-cuadrado) y aplicación Streamlit. |
| `modelo_rf.joblib` | `mlops_pipeline/src/` | Pipeline entrenado (preprocesamiento + Random Forest), generado por `model_training_evaluation.py`. |
| `config.json` | `mlops_pipeline/src/` | Contiene `project_code`, que usa `set_up.bat` para nombrar el entorno virtual. |
| `Base_de_datos.xlsx` | raíz | Dataset original provisto. |
| `Base_de_datos.csv` | raíz | Dataset en formato csv, usado por todos los scripts. |
| `requirements.txt` | raíz | Librerías del proyecto completo (análisis, modelado, monitoreo). |
| `requirements-api.txt` | raíz | Librerías mínimas de la API, con versiones fijas; lo usa el `Dockerfile`. |
| `set_up.bat` | raíz | Crea el entorno virtual `fg-venv`, instala `requirements.txt` y registra el kernel de Jupyter (Windows). |
| `Dockerfile` | raíz | Instrucciones para construir la imagen Docker de la API. |
| `.dockerignore` | raíz | Archivos que Docker no copia a la imagen (entorno virtual, notebooks, datos). |
| `.gitignore` | raíz | Archivos que Git no versiona. |
| `README.md` | raíz | Esta documentación. |

## 4. Proceso por avance

### Avance 1 - Versionamiento y análisis exploratorio
- Estructura de carpetas, entorno virtual (`set_up.bat`) y `requirements.txt`.
- `Cargar_datos.ipynb`: carga del dataset y conversión a `.csv`.
- `comprension_eda.ipynb`: exploración inicial, análisis univariable, bivariable y multivariable.

Hallazgos de calidad de datos, y tratamiento aplicado:
- `tendencia_ingresos`: 58 valores numéricos mezclados con las categorías Estable/Creciente/Decreciente. No se pudo recuperar el valor original, por lo que se trataron como nulos.
- `puntaje`: el valor `95.227787` se repite 9.407 veces (~87 % del dataset). Se trató como valor de relleno y se convirtió a nulo.
- `puntaje_datacredito`: los valores 0 y negativos se trataron como nulos.
- `salario_cliente` y `total_otros_prestamos`: valores mayores a 100.000.000 se trataron como errores de captura y se convirtieron a nulo.
- `fecha_prestamo`: convertida a fecha; se derivaron `mes_prestamo` y `anio_prestamo`.

### Avance 2 - Ingeniería de características y modelado
- `ft_engineering.py`: `ColumnTransformer` con pipelines por tipo de variable (numéricas: `SimpleImputer`; nominales: `SimpleImputer` + `OneHotEncoder`; ordinal: `SimpleImputer` + `OrdinalEncoder`) y división train/test estratificada (80 % / 20 %, `random_state=42`).
- `model_training_evaluation.py`: funciones `build_model` y `summarize_classification`; se entrenaron Regresión Logística, Árbol de Decisión y Random Forest, con tabla y gráfico comparativos.
- Con `puntaje`: Árbol de Decisión y Random Forest obtuvieron 1,0 en todas las métricas y la Regresión Logística ~0,98. Se eligió Random Forest.
- Sin `puntaje`: ROC-AUC de 0,63 (Regresión Logística), 0,53 (Árbol) y 0,67 (Random Forest).

### Avance 3 - Monitoreo y data drift
- `model_monitoring.py` calcula el drift entre una población de referencia y ventanas mensuales:
  - Variables numéricas: Kolmogorov-Smirnov, PSI y distancia de Jensen-Shannon.
  - Variables categóricas: Chi-cuadrado y V de Cramér.
- Aplicación en Streamlit: tabla de métricas con semáforo, barra de riesgo, histogramas referencia vs. actual, evolución temporal del PSI y alertas con recomendaciones automáticas.

### Avance 4 - Despliegue del modelo (API y Docker)
- `model_training_evaluation.py` guarda el Random Forest final como un pipeline completo en `modelo_rf.joblib`, entrenado con los mismos datos de entrenamiento del Avance 2. Al verificarlo dio ROC-AUC y accuracy de 1,0 en test, igual que en el Avance 2. El archivo también guarda las columnas que espera el modelo y la versión de scikit-learn (1.9.1).
- `model_deploy.py` carga ese archivo, aplica las mismas reglas de limpieza del EDA y expone una API con FastAPI:

| Endpoint | Qué hace |
|---|---|
| `GET /health` | Indica que la API está activa y con qué versión de scikit-learn se guardó el modelo. |
| `POST /predict` | Predicción por lotes. Recibe `{"registros": [ {...}, {...} ]}` en JSON y devuelve una predicción por registro. |
| `POST /predict_csv` | Predicción por lotes desde un archivo CSV (una fila = un crédito). |

- Cada resultado trae `prediccion` (1 paga, 0 no paga), `probabilidad_pago` y `advertencias`. Si el registro llega sin `puntaje`, se agrega una advertencia (ver limitaciones).
- Las columnas que no lleguen en un registro se completan como vacías y el pipeline las imputa.
- La documentación interactiva la genera FastAPI en `/docs`.
- Pruebas realizadas: `/predict_csv` con `Base_de_datos.csv` devolvió 10.763 predicciones; con 10 registros reales (5 que pagaron y 5 que no) las 10 predicciones coincidieron con lo ocurrido.
- Imagen Docker (`Dockerfile`): base `python:3.14-slim`, instala `requirements-api.txt` (scikit-learn 1.9.1, pandas 3.0.6 y numpy 2.5.3 fijados), copia `model_deploy.py` y `modelo_rf.joblib`, y arranca la API con Uvicorn en el puerto 8000. La imagen se construyó y se probó corriendo el contenedor en forma local: `/health`, `/predict` y `/predict_csv` respondieron correctamente.

## 5. Decisiones de diseño del monitoreo

**Es una simulación.** No hay datos de producción reales. Se usó `fecha_prestamo` para dividir el dataset:
- Referencia: préstamos hasta mayo de 2025 (7.713 filas).
- Actual: préstamos desde junio de 2025 (3.050 filas), en ventanas mensuales. Febrero, marzo y abril de 2026 se agruparon en una sola ventana porque tenían muy pocas filas.
- El Random Forest se entrenó solo con la referencia y se usó para predecir sobre ambas poblaciones.
- `mes_prestamo` y `anio_prestamo` se excluyeron del monitoreo, porque difieren entre poblaciones por construcción.

**Umbrales de alerta:**

| Métrica | Atención | Crítico | Origen |
|---|---|---|---|
| PSI | 0,10 | 0,25 | Convención de la industria de riesgo crediticio |
| KS (estadístico) | 0,10 | 0,20 | Criterio propio |
| V de Cramér | 0,10 | 0,30 | Criterio propio |

Otros criterios propios: una variable con menos de 30 filas válidas en la ventana no se evalúa ("Muestra insuficiente"); se alerta si los nulos de `puntaje` superan en 5 puntos o más a los de la referencia; se sugiere reentrenar si `prob_pago` está en estado crítico o si hay 3 o más variables críticas.

## 6. Hallazgos del monitoreo

- Drift creciente: `capital_prestado`, `cuota_pactada`, `total_otros_prestamos` y `huella_consulta` pasan de estables a críticas.
- Las predicciones del modelo (`prob_pago`) quedan en estado crítico desde diciembre de 2025. En la última ventana (feb-abr 2026) hay 6 variables críticas y la app recomienda evaluar un reentrenamiento.
- `plazo_meses` hace lo contrario: es crítica en jun-ago de 2025 y luego se estabiliza.
- `promedio_ingresos_datacredito` está en estado crítico en todas las ventanas, con entre 71 y 491 filas válidas por ventana. Sus nulos suben de 27,2 % a 44,1 % en la última ventana.
- Las variables categóricas se mantienen estables (V de Cramér menor a 0,11 en todas las ventanas).
- La tasa de pago real pasó de 94,68 % (referencia) a 96,69 % (actual).
- No se determinó el motivo de estos cambios.

## 7. Limitaciones

1. **El relleno de `puntaje` se asocia con la variable objetivo.** Las 9.407 filas con el valor de relleno corresponden en su totalidad a clientes que pagaron a tiempo (0 casos de no pago); los 511 clientes que no pagaron tienen un puntaje real. Esto explica el ROC-AUC de 1,0 y es una forma de data leakage. Por qué ocurre en el sistema de origen no se pudo determinar; debe validarse con el equipo de negocio antes de llevar el modelo a producción. El ROC-AUC de 0,67 sin `puntaje` es una estimación más conservadora. Se decidió dejarlo como limitación documentada y no reentrenar.
2. **La API refleja esa limitación.** Cuando un registro llega sin `puntaje`, la respuesta incluye una advertencia, porque en esos casos el modelo tiende a predecir "paga". En una prueba con 10 registros, 4 de los 5 que pagaron no tenían puntaje y los 5 que no pagaron sí lo tenían. Son 10 casos, un ejemplo y no una medición.
3. **Pocas filas con `puntaje` real.** Desde septiembre de 2025 hay menos de 20 filas con valor real por ventana, por lo que su PSI no es confiable. Se monitorea la tasa de nulos de `puntaje` (85,9 % en la referencia, entre 88 % y 98 % en las ventanas actuales).
4. **El `.joblib` solo se carga con la misma versión de scikit-learn con la que se guardó (1.9.1).** Por eso `requirements-api.txt` la fija.
5. El modelo usa `mes_prestamo` y `anio_prestamo` como variables, que no generalizan a fechas futuras.
6. Las ventanas más recientes tienen pocas filas (127), lo que agrega ruido al PSI.
7. Los umbrales distintos al PSI son criterios propios y no una norma.
8. La API y la imagen Docker se probaron en forma local; no se desplegaron en un servidor ni en la nube.
9. La API no valida los valores de cada campo (solo aplica las reglas de limpieza y deja que el pipeline impute lo que falte).

## 8. Cómo ejecutarlo

**Entorno y análisis**
1. Clonar el repositorio y pararse en la raíz.
2. En Windows, ejecutar `set_up.bat` (crea el entorno `fg-venv` a partir de `mlops_pipeline/src/config.json` e instala `requirements.txt`).
3. Activar el entorno: `source fg-venv/Scripts/activate` (Git Bash).

**Monitoreo**
4. Por consola: `python mlops_pipeline/src/model_monitoring.py`
5. Aplicación: `streamlit run mlops_pipeline/src/model_monitoring.py`

**API sin Docker**
6. `uvicorn model_deploy:app --app-dir mlops_pipeline/src --reload` y abrir `http://127.0.0.1:8000/docs`.

**API con Docker** (Docker Desktop abierto)
7. Construir la imagen: `docker build -t api-pago-tiempo .`
8. Correr el contenedor: `docker run -d --name api-pago -p 8000:8000 api-pago-tiempo`
9. Probar en `http://127.0.0.1:8000/health` y `http://127.0.0.1:8000/docs`.
10. Frenar y borrar el contenedor: `docker stop api-pago` y `docker rm api-pago`.

**Ejemplo de llamada a `/predict`:**

```json
{
  "registros": [
    {"fecha_prestamo": "2025-03-10 10:00:00", "tipo_credito": 4, "capital_prestado": 2000000,
     "plazo_meses": 10, "edad_cliente": 40, "tipo_laboral": "Empleado", "salario_cliente": 3000000}
  ]
}
```