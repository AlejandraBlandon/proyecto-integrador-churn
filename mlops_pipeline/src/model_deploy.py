# %% Importar librerías
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

# %% Cargar el modelo guardado
# Ruta robusta: funciona por celdas o como script, desde cualquier carpeta
try:
    RAIZ = Path(__file__).resolve().parents[2]
except NameError:
    RAIZ = Path.cwd().resolve().parents[1]

RUTA_MODELO = RAIZ / "mlops_pipeline" / "src" / "modelo_rf.joblib"

paquete = joblib.load(RUTA_MODELO)
modelo = paquete["modelo"]                      # pipeline completo (preprocesador + Random Forest)
COLUMNAS_ENTRADA = paquete["columnas_features"] # columnas que el modelo espera
print("Modelo cargado. scikit-learn con el que se guardó:", paquete["sklearn_version"])

# %% Limpieza de los datos de entrada (las mismas reglas del EDA y de ft_engineering.py)
CATEGORIAS_TENDENCIA = ["Estable", "Creciente", "Decreciente"]

def limpiar_datos(df_crudo):
    """Recibe registros crudos (DataFrame) y los deja listos para el modelo."""
    df = df_crudo.copy()

    # Datos mínimos necesarios para derivar mes y año
    if "fecha_prestamo" not in df.columns:
        raise ValueError("Falta la columna 'fecha_prestamo'.")
    df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])
    df["mes_prestamo"] = df["fecha_prestamo"].dt.month
    df["anio_prestamo"] = df["fecha_prestamo"].dt.year

    # Si falta alguna columna del modelo, se agrega vacía (el pipeline imputa los nulos)
    for col in COLUMNAS_ENTRADA:
        if col not in df.columns:
            df[col] = np.nan

    # Reglas de validación definidas en el EDA
    df.loc[~df["tendencia_ingresos"].isin(CATEGORIAS_TENDENCIA), "tendencia_ingresos"] = np.nan
    df.loc[df["puntaje"] == 95.227787, "puntaje"] = np.nan
    df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan
    df.loc[df["salario_cliente"] > 100_000_000, "salario_cliente"] = np.nan
    df.loc[df["total_otros_prestamos"] > 100_000_000, "total_otros_prestamos"] = np.nan

    # Mismos tipos que en el entrenamiento
    for col in ["tipo_credito", "tipo_laboral", "tendencia_ingresos"]:
        df[col] = df[col].astype("category")

    return df[COLUMNAS_ENTRADA]

# %% Lógica de predicción
def predecir(df_crudo):
    """Devuelve, por cada registro: la predicción (1 paga, 0 no paga), la probabilidad y advertencias."""
    datos = limpiar_datos(df_crudo)
    pred = modelo.predict(datos)
    proba = modelo.predict_proba(datos)[:, 1]

    resultados = []
    for i in range(len(datos)):
        advertencias = []
        if pd.isna(datos.iloc[i]["puntaje"]):
            advertencias.append("Registro sin puntaje: el modelo tiende a predecir 'paga' en estos casos "
                                "(limitación documentada en el README).")
        resultados.append({
            "prediccion": int(pred[i]),
            "probabilidad_pago": round(float(proba[i]), 4),
            "advertencias": advertencias,
        })
    return resultados

# %% API con FastAPI
from typing import Any
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(
    title="API de predicción de pago a tiempo",
    description="Recibe uno o varios créditos y devuelve si se predice que el cliente pagará a tiempo.",
    version="1.0.0",
)

class LoteRegistros(BaseModel):
    """Un lote de registros. Cada registro es un diccionario con las columnas del dataset."""
    registros: list[dict[str, Any]]

@app.get("/health")
def estado():
    """Indica que la API está activa y con qué versión de scikit-learn se guardó el modelo."""
    return {"estado": "ok", "sklearn_version_modelo": paquete["sklearn_version"]}

@app.post("/predict")
def predict(lote: LoteRegistros):
    """Predicción por lotes: devuelve una predicción por cada registro, en el mismo orden."""
    if len(lote.registros) == 0:
        raise HTTPException(status_code=422, detail="El lote no tiene registros.")
    try:
        resultados = predecir(pd.DataFrame(lote.registros))
    except (ValueError, KeyError, TypeError) as error:
        raise HTTPException(status_code=422, detail=f"No se pudo procesar el lote: {error}")
    return {"cantidad": len(resultados), "resultados": resultados}


