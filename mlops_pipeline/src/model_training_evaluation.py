# %% Importar librerías
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
# %% Cargar datos y replicar las mismas limpiezas y transformaciones de ft_engineering.py

df = pd.read_csv("../../Base_de_datos.csv")

# Limpiezas (idénticas a ft_engineering.py)
df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

categorias_validas = ["Estable", "Creciente", "Decreciente"]
df.loc[~df["tendencia_ingresos"].isin(categorias_validas), "tendencia_ingresos"] = np.nan

df.loc[df["puntaje"] == 95.227787, "puntaje"] = np.nan
df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan
df.loc[df["salario_cliente"] > 100_000_000, "salario_cliente"] = np.nan
df.loc[df["total_otros_prestamos"] > 100_000_000, "total_otros_prestamos"] = np.nan

df["tipo_credito"] = df["tipo_credito"].astype("category")
df["tipo_laboral"] = df["tipo_laboral"].astype("category")
df["tendencia_ingresos"] = df["tendencia_ingresos"].astype("category")

df["mes_prestamo"] = df["fecha_prestamo"].dt.month
df["anio_prestamo"] = df["fecha_prestamo"].dt.year

df.shape

# %% Definir columnas, separar features/target, y armar el ColumnTransformer

columnas_numericas = [
    "capital_prestado", "plazo_meses", "edad_cliente", "salario_cliente",
    "total_otros_prestamos", "cuota_pactada", "puntaje", "puntaje_datacredito",
    "cant_creditosvigentes", "huella_consulta", "saldo_mora", "saldo_total",
    "saldo_principal", "saldo_mora_codeudor", "creditos_sectorFinanciero",
    "creditos_sectorCooperativo", "creditos_sectorReal", "promedio_ingresos_datacredito",
    "mes_prestamo", "anio_prestamo"
]
columnas_nominales = ["tipo_laboral", "tipo_credito"]
columnas_ordinales = ["tendencia_ingresos"]
target = "Pago_atiempo"

X = df[columnas_numericas + columnas_nominales + columnas_ordinales]
y = df[target]

# Split train/test (mismos parámetros que en ft_engineering.py, para reproducibilidad)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ColumnTransformer (idéntico al de ft_engineering.py)
pipeline_numerico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
pipeline_nominal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])
pipeline_ordinal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[["Decreciente", "Estable", "Creciente"]]))
])
preprocesador = ColumnTransformer(transformers=[
    ("num", pipeline_numerico, columnas_numericas),
    ("nom", pipeline_nominal, columnas_nominales),
    ("ord", pipeline_ordinal, columnas_ordinales)
])

# Ajustar con train, transformar train y test
X_train_transformado = preprocesador.fit_transform(X_train)
X_test_transformado = preprocesador.transform(X_test)

print("X_train_transformado:", X_train_transformado.shape)
print("X_test_transformado:", X_test_transformado.shape)

# %% Función build_model: entrena un modelo y devuelve las predicciones

def build_model(modelo, X_train, y_train, X_test):
    """
    Entrena un modelo con los datos de entrenamiento y genera predicciones sobre test.
    
    Parámetros:
    - modelo: una instancia de un clasificador de sklearn (ej. LogisticRegression())
    - X_train, y_train: datos de entrenamiento (ya transformados)
    - X_test: datos de evaluación (ya transformados)
    
    Devuelve:
    - modelo: el modelo ya entrenado (con fit aplicado)
    - y_pred: las predicciones de clase (0 o 1) sobre X_test
    - y_pred_proba: las probabilidades predichas de la clase 1 (necesarias para ROC-AUC)
    """
    modelo.fit(X_train, y_train)  # entrena el modelo con los datos de train
    y_pred = modelo.predict(X_test)  # predice la clase (0 o 1) para cada fila de test
    y_pred_proba = modelo.predict_proba(X_test)[:, 1]  # probabilidad de que sea clase 1
    
    return modelo, y_pred, y_pred_proba

# %% Función summarize_classification: calcula y muestra las métricas de un modelo

def summarize_classification(nombre_modelo, y_test, y_pred, y_pred_proba):
    """
    Calcula las métricas de clasificación pedidas por el PI (precisión, recall, F1, ROC-AUC)
    y arma la matriz de confusión, para un modelo ya entrenado.
    
    Devuelve un diccionario con los resultados, para poder juntarlos todos después
    en una tabla comparativa.
    """
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)
    
    print(f"--- {nombre_modelo} ---")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")
    print()
    
    # Matriz de confusión: muestra cuántos aciertos y errores tuvo el modelo, por clase
    matriz = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["No paga (0)", "Paga (1)"])
    disp.plot(cmap="Blues")
    plt.title(f"Matriz de confusión - {nombre_modelo}")
    plt.show()
    
    # Devolvemos todo en un diccionario, para poder armar la tabla comparativa después
    return {
        "modelo": nombre_modelo,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc
    }


# %% Entrenar y evaluar Modelo 1: Regresión Logística

# max_iter=1000: por defecto sklearn usa 100 iteraciones para "converger" (encontrar la mejor
#   solución); con muchas columnas a veces no alcanza, así que le doy más margen
modelo_logistica = LogisticRegression(max_iter=1000, random_state=42)

modelo_logistica, y_pred_logistica, y_proba_logistica = build_model(
    modelo_logistica, X_train_transformado, y_train, X_test_transformado
)

resultados_logistica = summarize_classification(
    "Regresión Logística", y_test, y_pred_logistica, y_proba_logistica
)
# %% Entrenar y evaluar Modelo 2: Árbol de Decisión

modelo_arbol = DecisionTreeClassifier(random_state=42)

modelo_arbol, y_pred_arbol, y_proba_arbol = build_model(
    modelo_arbol, X_train_transformado, y_train, X_test_transformado
)

resultados_arbol = summarize_classification(
    "Árbol de Decisión", y_test, y_pred_arbol, y_proba_arbol
)

# %% Entrenar y evaluar Modelo 3: Random Forest

modelo_rf = RandomForestClassifier(random_state=42)

modelo_rf, y_pred_rf, y_proba_rf = build_model(
    modelo_rf, X_train_transformado, y_train, X_test_transformado
)

resultados_rf = summarize_classification(
    "Random Forest", y_test, y_pred_rf, y_proba_rf
)

# %% Investigar qué variable está generando el resultado perfecto (posible data leakage)

# feature_importances_  dice qué tanto peso le dio el Random Forest a cada columna
# para tomar sus decisiones. Si una sola variable tiene una importancia dominante,
# es la sospechosa principal de estar filtrando la respuesta.

nombres_columnas = preprocesador.get_feature_names_out()

importancias = pd.Series(modelo_rf.feature_importances_, index=nombres_columnas)
importancias.sort_values(ascending=False).head(10)


# %% [markdown]
# ### Investigación: resultados perfectos en Árbol de Decisión y Random Forest
# 
# Ambos modelos basados en árboles obtuvieron 100% en todas las métricas, un resultado 
# atípico que amerita investigación, ya que un desempeño perfecto rara vez ocurre con datos 
# reales y suele indicar sobreajuste o data leakage.
# 
# Al revisar la importancia de variables del Random Forest, se encontró que `puntaje` 
# concentra el 86% de la importancia total, muy por encima del resto de las variables 
# (la siguiente más relevante, `puntaje_datacredito`, apenas alcanza el 1.9%). Esto es 
# consistente con la altísima correlación (0.88) detectada en el EDA entre `puntaje` y 
# `Pago_atiempo`.
# 
# **Hipótesis:** `puntaje` podría estar actuando como una variable que "filtra" la respuesta 
# (data leakage), ya que un puntaje de crédito suele calcularse incorporando el historial de 
# pago de la persona, por lo que predecir `Pago_atiempo` a partir de `puntaje` podría ser, 
# en parte, circular.

# %% Armar una segunda versión de los datos, sin la variable puntaje (para comparar)

# Repetimos la lista de columnas numéricas, pero sacando "puntaje"
columnas_numericas_sin_puntaje = [c for c in columnas_numericas if c != "puntaje"]

X_sin_puntaje = df[columnas_numericas_sin_puntaje + columnas_nominales + columnas_ordinales]

# Mismo split, mismos parámetros (random_state=42) para que sea una comparación justa
X_train_sp, X_test_sp, y_train_sp, y_test_sp = train_test_split(
    X_sin_puntaje, y, test_size=0.2, random_state=42, stratify=y
)

# Armamos un ColumnTransformer nuevo (idéntico al anterior, pero sin "puntaje" en numéricas)
pipeline_numerico_sp = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])
preprocesador_sin_puntaje = ColumnTransformer(transformers=[
    ("num", pipeline_numerico_sp, columnas_numericas_sin_puntaje),
    ("nom", pipeline_nominal, columnas_nominales),
    ("ord", pipeline_ordinal, columnas_ordinales)
])

X_train_sp_transformado = preprocesador_sin_puntaje.fit_transform(X_train_sp)
X_test_sp_transformado = preprocesador_sin_puntaje.transform(X_test_sp)

print("Sin puntaje - Train:", X_train_sp_transformado.shape)
print("Sin puntaje - Test:", X_test_sp_transformado.shape)


# %% Entrenar los 3 modelos sin puntaje, para comparar

modelo_logistica_sp = LogisticRegression(max_iter=1000, random_state=42)
modelo_logistica_sp, y_pred_logistica_sp, y_proba_logistica_sp = build_model(
    modelo_logistica_sp, X_train_sp_transformado, y_train_sp, X_test_sp_transformado
)
resultados_logistica_sp = summarize_classification(
    "Regresión Logística (sin puntaje)", y_test_sp, y_pred_logistica_sp, y_proba_logistica_sp
)

# %%
modelo_arbol_sp = DecisionTreeClassifier(random_state=42)
modelo_arbol_sp, y_pred_arbol_sp, y_proba_arbol_sp = build_model(
    modelo_arbol_sp, X_train_sp_transformado, y_train_sp, X_test_sp_transformado
)
resultados_arbol_sp = summarize_classification(
    "Árbol de Decisión (sin puntaje)", y_test_sp, y_pred_arbol_sp, y_proba_arbol_sp
)

# %%
modelo_rf_sp = RandomForestClassifier(random_state=42)
modelo_rf_sp, y_pred_rf_sp, y_proba_rf_sp = build_model(
    modelo_rf_sp, X_train_sp_transformado, y_train_sp, X_test_sp_transformado
)
resultados_rf_sp = summarize_classification(
    "Random Forest (sin puntaje)", y_test_sp, y_pred_rf_sp, y_proba_rf_sp
)

# %% [markdown]
# ### Comparación: modelos con vs. sin `puntaje`
# 
# Al remover `puntaje`, ningún modelo alcanza resultados perfectos, esto confirma que su 
# inclusión generaba data leakage (o al menos una dependencia excesiva de una sola variable).
# 
# Sin embargo, el desempeño sin `puntaje` es considerablemente más débil:
# - **Regresión Logística** y **Random Forest** logran Accuracy 95.3% y Recall 100%, pero 
#   la matriz de confusión revela que esto es engañoso: casi siempre predicen "Paga (1)", 
#   acertando solo 1 de 102 casos de "No paga (0)". El ROC-AUC (0.63 y 0.67) confirma un 
#   poder de discriminación débil, apenas por encima del azar (0.50).
# - **Árbol de Decisión** es el único que detecta parte de la clase minoritaria (12 de 102), 
#   a costa de un accuracy general más bajo (90.4%) y un ROC-AUC aún más débil (0.53).
# 
# **Conclusión:** sin `puntaje`, el dataset no contiene suficiente señal para predecir 
# `Pago_atiempo` de forma confiable, ningún modelo logra un buen balance entre detectar la 
# clase minoritaria y mantener buena performance general. Esto reafirma que `puntaje` es, 
# lejos, la variable más informativa disponible, aunque su origen (posible componente de 
# historial de pago) amerite ser señalado como una limitación a validar con el equipo de 
# negocio antes de llevar el modelo a producción.


# %% Tabla resumen comparativa de los 3 modelos (con puntaje)

# Juntamos los diccionarios que devolvió summarize_classification para cada modelo
tabla_resumen = pd.DataFrame([
    resultados_logistica,
    resultados_arbol,
    resultados_rf
])

# Ordenamos por ROC-AUC descendente (una métrica robusta, buena para ordenar cuando
# hay desbalance de clases, ya que evalúa qué tan bien el modelo distingue entre clases
# en todos los umbrales posibles, no solo en el umbral por defecto de 0.5)
tabla_resumen = tabla_resumen.sort_values("roc_auc", ascending=False).reset_index(drop=True)

tabla_resumen

# %% Gráfico comparativo de métricas entre los 3 modelos

# Pasamos la tabla de formato "ancho" (una columna por métrica) a formato "largo"
# (una fila por cada combinación modelo-métrica), que es lo que necesita seaborn para
# poder agrupar barras por modelo y por métrica al mismo tiempo
tabla_larga = tabla_resumen.melt(
    id_vars="modelo",
    value_vars=["accuracy", "precision", "recall", "f1", "roc_auc"],
    var_name="metrica",
    value_name="valor"
)

plt.figure(figsize=(10, 6))
sns.barplot(data=tabla_larga, x="metrica", y="valor", hue="modelo")
plt.title("Comparación de métricas entre modelos (con puntaje)")
plt.ylim(0, 1.05)  # las métricas van de 0 a 1, fijamos el eje para que se vea parejo
plt.legend(title="Modelo", bbox_to_anchor=(1.05, 1), loc="upper left")  # leyenda afuera del gráfico
plt.tight_layout()
plt.show()

# %% [markdown]
# ### Interpretación: comparación de modelos (con puntaje)
# 
# Los 3 modelos muestran un desempeño muy alto, con Árbol de Decisión y Random Forest 
# alcanzando prácticamente el máximo en todas las métricas (1.0), y Regresión Logística 
# levemente por debajo (~0.98).
# 
# Como se documentó anteriormente, este desempeño casi perfecto está fuertemente influenciado 
# por el peso dominante de la variable `puntaje` (86% de importancia en Random Forest), y no 
# necesariamente refleja la capacidad del modelo de generalizar con información más diversa.

# %% [markdown]
# ## Selección del modelo final
# 
# **Modelo elegido: Random Forest**
# 
# Justificación:
# - Entre los modelos con mejor performance (Árbol de Decisión y Random Forest, ambos con 
#   métricas cercanas al máximo), se elige **Random Forest** porque, al combinar múltiples 
#   árboles de decisión entrenados con distintos subconjuntos de datos, es más robusto al 
#   sobreajuste que un único Árbol de Decisión — generaliza mejor ante datos nuevos que el 
#   modelo no vio durante el entrenamiento.
# - Random Forest superó levemente a Regresión Logística en todas las métricas evaluadas.
# 
# **Limitación importante a señalar:** el desempeño casi perfecto de este modelo está 
# fuertemente influenciado por la variable `puntaje`, que concentra el 86% de la importancia 
# en las decisiones del modelo. Al remover esta variable, el desempeño cae considerablemente 
# (ROC-AUC de 1.0 a 0.67), lo que sugiere que el resto de las variables del dataset no 
# contienen suficiente señal predictiva por sí solas.
# 
# **Recomendación para producción:** antes de desplegar este modelo, se debería validar con 
# el equipo de negocio el origen exacto de `puntaje` — específicamente, si su cálculo ya 
# incorpora información sobre el historial de pago del cliente (lo cual generaría un 
# problema de circularidad/data leakage), o si es un dato genuinamente independiente y 
# disponible al momento de evaluar un crédito nuevo.

# %%
