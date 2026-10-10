


# Predicción de pago a tiempo de créditos y monitoreo de data drift

Proyecto integrador 

## 1. Caso de negocio

Rol: Científico de Datos Junior Advanced en el equipo de Datos y Analítica de una empresa financiera.

Objetivo: desarrollar un modelo predictivo de aprendizaje automático supervisado, a partir de información histórica de créditos, para anticipar el comportamiento de nuevos usuarios. La variable objetivo es `Pago_atiempo` (1 = pagó a tiempo, 0 = no pagó a tiempo).

Restricción de la empresa: cada proyecto debe respetar una estructura de carpetas definida, porque el paso a producción está automatizado con pipelines de validación en Jenkins. Por eso la estructura no se modificó.

## 2. Estructura del repositorio

```
mlops_pipeline/
    src/
        Cargar_datos.ipynb
        comprension_eda.ipynb
        ft_engineering.py
        model_training_evaluation.py
        model_deploy.py
        model_monitoring.py
        config.json
Base_de_datos.csv
requirements.txt
.gitignore
README.md
set_up.bat
```

Ramas: `main`, `developer` y `certification`. El trabajo se desarrolló en `developer` y se integró a `main` mediante Pull Request.

## 3. Datos

- 10.763 registros y 23 variables (`Base_de_datos.xlsx`, convertido a `Base_de_datos.csv` en `Cargar_datos.ipynb`).
- Variable objetivo desbalanceada: 95,3 % paga a tiempo y 4,7 % no paga.
- Los préstamos van de noviembre de 2024 a abril de 2026.

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
- `ft_engineering.py`: `ColumnTransformer` con pipelines por tipo de variable (numéricas: `SimpleImputer`; nominales: `SimpleImputer` + `OneHotEncoder`; ordinal: `SimpleImputer` + `OrdinalEncoder`) y división train/test estratificada.
- `model_training_evaluation.py`: funciones `build_model` y `summarize_classification`; se entrenaron Regresión Logística, Árbol de Decisión y Random Forest, con tabla y gráfico comparativos.
- Con `puntaje`: Árbol de Decisión y Random Forest obtuvieron 1,0 en todas las métricas y la Regresión Logística ~0,98. Se eligió Random Forest.
- Sin `puntaje`: ROC-AUC de 0,63 (Regresión Logística), 0,53 (Árbol) y 0,67 (Random Forest).

### Avance 3 - Monitoreo y data drift
- `model_monitoring.py` calcula el drift entre una población de referencia y ventanas mensuales:
  - Variables numéricas: Kolmogorov-Smirnov, PSI y distancia de Jensen-Shannon.
  - Variables categóricas: Chi-cuadrado y V de Cramér.
- Aplicación en Streamlit: tabla de métricas con semáforo, barra de riesgo, histogramas referencia vs. actual, evolución temporal del PSI y alertas con recomendaciones automáticas.

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

1. **El relleno de `puntaje` se asocia con la variable objetivo.** Las 9.407 filas con el valor de relleno corresponden en su totalidad a clientes que pagaron a tiempo (0 casos de no pago); los 511 clientes que no pagaron tienen un puntaje real. Esto explica el ROC-AUC de 1,0 y es una forma de data leakage. Por qué ocurre en el sistema de origen no se pudo determinar; debe validarse con el equipo de negocio antes de llevar el modelo a producción. El ROC-AUC de 0,67 sin `puntaje` es una estimación más conservadora.
2. **Pocas filas con `puntaje` real.** Desde septiembre de 2025 hay menos de 20 filas con valor real por ventana, por lo que su PSI no es confiable. Se monitorea la tasa de nulos de `puntaje` (85,9 % en la referencia, entre 88 % y 98 % en las ventanas actuales).
3. El modelo usa `mes_prestamo` y `anio_prestamo` como variables, que no generalizan a fechas futuras.
4. Las ventanas más recientes tienen pocas filas (127), lo que agrega ruido al PSI.
5. Los umbrales distintos al PSI son criterios propios y no una norma.
6. `model_deploy.py` queda pendiente para el Avance 4.

## 8. Cómo ejecutarlo

1. Clonar el repositorio y pararse en la raíz.
2. En Windows, ejecutar `set_up.bat` (crea el entorno `fg-venv` a partir de `src/config.json` e instala `requirements.txt`).
3. Activar el entorno: `source fg-venv/Scripts/activate` (Git Bash).
4. Monitoreo por consola: `python mlops_pipeline/src/model_monitoring.py`
5. Aplicación: `streamlit run mlops_pipeline/src/model_monitoring.py`
