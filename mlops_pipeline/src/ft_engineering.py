# %% Importar librerías
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# %% Cargar el dataset crudo (el mismo csv generado en Cargar_datos.ipynb)
df = pd.read_csv("../../Base_de_datos.csv")
df.shape

# %% Replicar las limpiezas de datos definidas en comprension_eda.ipynb

# 1. Corregir tipo de dato de la fecha
df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

# 2. tendencia_ingresos: solo son válidas estas 3 categorías; el resto (valores numéricos
#    sueltos, error de calidad de datos) se convierte a nulo
categorias_validas = ["Estable", "Creciente", "Decreciente"]
df.loc[~df["tendencia_ingresos"].isin(categorias_validas), "tendencia_ingresos"] = np.nan

# 3. puntaje: 95.227787 era un valor de relleno (se repetía en el 87% de los casos), se trata como nulo
df.loc[df["puntaje"] == 95.227787, "puntaje"] = np.nan

# 4. puntaje_datacredito: 0 y negativos no son válidos para un puntaje, se tratan como nulo
df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan

# 5. salario_cliente y total_otros_prestamos: valores mayores a 100 millones son outliers
#    extremos (error de captura), se tratan como nulo
df.loc[df["salario_cliente"] > 100_000_000, "salario_cliente"] = np.nan
df.loc[df["total_otros_prestamos"] > 100_000_000, "total_otros_prestamos"] = np.nan

# 6. Convertir columnas categóricas a tipo category
df["tipo_credito"] = df["tipo_credito"].astype("category")
df["tipo_laboral"] = df["tipo_laboral"].astype("category")
df["tendencia_ingresos"] = df["tendencia_ingresos"].astype("category")

df.isnull().sum()

# %% Derivar atributos de fecha_prestamo (pendiente del EDA)
# La fecha completa tiene demasiada cardinalidad para ser útil tal cual (casi 1 valor por fila),
# así que extraemos el mes y el año, que sí pueden capturar patrones de estacionalidad

df["mes_prestamo"] = df["fecha_prestamo"].dt.month
df["anio_prestamo"] = df["fecha_prestamo"].dt.year

df[["fecha_prestamo", "mes_prestamo", "anio_prestamo"]].head()

# %% Definir grupos de columnas para el ColumnTransformer

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

# La variable objetivo (lo que queremos predecir)
target = "Pago_atiempo"

# X = todas las columnas que va a usar el modelo para predecir (features)
# y = la columna que el modelo tiene que aprender a predecir
X = df[columnas_numericas + columnas_nominales + columnas_ordinales]
y = df[target]

print(X.shape, y.shape)


# %% Armar el ColumnTransformer (según el diagrama: SimpleImputer + encoders por tipo de columna)

# Pipeline para numéricas: solo imputación de nulos
# strategy="median" usa la mediana para rellenar, más robusta que el promedio cuando hay outliers
pipeline_numerico = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median"))
])

# Pipeline para categóricas nominales: imputación + OneHotEncoder
# OneHotEncoder convierte cada categoría en una columna binaria (0/1), sin asumir ningún orden
# handle_unknown="ignore" evita que el pipeline falle si en test aparece una categoría no vista en train
pipeline_nominal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Pipeline para categóricas ordinales: imputación + OrdinalEncoder
# OrdinalEncoder convierte cada categoría en un número, respetando el orden que le indiquemos
# categories=[[...]] define explícitamente el orden: Decreciente(0) < Estable(1) < Creciente(2)
pipeline_ordinal = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(categories=[["Decreciente", "Estable", "Creciente"]]))
])

# ColumnTransformer combina los 3 pipelines, aplicando cada uno solo a las columnas que le corresponden
preprocesador = ColumnTransformer(transformers=[
    ("num", pipeline_numerico, columnas_numericas),
    ("nom", pipeline_nominal, columnas_nominales),
    ("ord", pipeline_ordinal, columnas_ordinales)
])

preprocesador

# %% Dividir en conjuntos de entrenamiento y evaluación

# test_size=0.2 significa que el 20% de los datos se reserva para evaluar, 80% para entrenar
# random_state=42 fija una "semilla" para que la división sea siempre la misma si vuelvo a correr
#   el código (reproducibilidad: un concepto clave que pide el PI en los objetivos de aprendizaje)
# stratify=y asegura que la proporción de Pago_atiempo (0/1) se mantenga igual en train y test
#   esto es muy importante en nuestro caso porque el dataset está desbalanceado (95%/5%):
#   sin stratify, podríamos terminar con un conjunto de test con muy pocos o ningún caso de "0"

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train:", X_train.shape, y_train.shape)
print("Test:", X_test.shape, y_test.shape)
print()
print("Proporción de Pago_atiempo en train:")
print(y_train.value_counts(normalize=True))
print()
print("Proporción de Pago_atiempo en test:")
print(y_test.value_counts(normalize=True))


# %% Aplicar el ColumnTransformer: ajustar con train, transformar train y test

# fit_transform() en X_train: el preprocesador "aprende" (fit) los parámetros de las
#   transformaciones solo con los datos de entrenamiento (ej: qué mediana usar para imputar,
#   qué categorías existen para el OneHotEncoder), y luego los aplica (transform)
# Esto es clave para evitar "data leakage": si dejo que el preprocesador vea el test
#   al momento de aprender esos parámetros, estaria filtrando información que en un
#   escenario real todavía no existe (el test simula datos "futuros" que el modelo no vio)

X_train_transformado = preprocesador.fit_transform(X_train)

# transform() (sin fit) en X_test: aplico las mismas transformaciones ya aprendidas en train,
#   sin volver a aprender nada nuevo de test
X_test_transformado = preprocesador.transform(X_test)

print("Forma de X_train transformado:", X_train_transformado.shape)
print("Forma de X_test transformado:", X_test_transformado.shape)
# %%
