# %% Importar librerías
import pandas as pd
import numpy as np
from scipy.stats import ks_2samp, chi2_contingency
from scipy.spatial.distance import jensenshannon

# %% Cargar el dataset y aplicar las mismas limpiezas que en ft_engineering.py
from pathlib import Path

# Ruta robusta: funciona ejecutando por celdas o como script desde cualquier carpeta
try:
    RAIZ = Path(__file__).resolve().parents[2]      # modo script: raíz del repo
except NameError:
    RAIZ = Path.cwd().resolve().parents[1]          # modo celdas: cwd es mlops_pipeline/src

df = pd.read_csv(RAIZ / "Base_de_datos.csv")

# Limpiezas idénticas a las de ft_engineering.py (por reproducibilidad)
df["fecha_prestamo"] = pd.to_datetime(df["fecha_prestamo"])

categorias_validas = ["Estable", "Creciente", "Decreciente"]
df.loc[~df["tendencia_ingresos"].isin(categorias_validas), "tendencia_ingresos"] = np.nan

df.loc[df["puntaje"] == 95.227787, "puntaje"] = np.nan
df.loc[df["puntaje_datacredito"] <= 0, "puntaje_datacredito"] = np.nan
df.loc[df["salario_cliente"] > 100_000_000, "salario_cliente"] = np.nan
df.loc[df["total_otros_prestamos"] > 100_000_000, "total_otros_prestamos"] = np.nan

df["mes_prestamo"] = df["fecha_prestamo"].dt.month
df["anio_prestamo"] = df["fecha_prestamo"].dt.year

# %% Ver cómo se reparten los préstamos por mes
df["periodo"] = df["fecha_prestamo"].dt.to_period("M")
df["periodo"].value_counts().sort_index()
# %%



# %% Definir población de referencia (histórica) y población actual

# Fecha de corte: todo lo anterior a junio 2025 es "referencia" (lo que el modelo conoció)
# y lo posterior es "actual" (simula los datos que llegarían a producción)
fecha_corte = pd.Timestamp("2025-06-01")

df_referencia = df[df["fecha_prestamo"] < fecha_corte].copy()
df_actual = df[df["fecha_prestamo"] >= fecha_corte].copy()

#Ventanas de monitoreo: un mes cada una...
df_actual["ventana"] = df_actual["periodo"].astype(str)

#salvo los últimos meses, que tienen muy pocos casos y se agrupan en una sola ventana
df_actual.loc[df_actual["fecha_prestamo"] >= pd.Timestamp("2026-02-01"), "ventana"] = "2026-02 a 2026-04"

print("Referencia:", df_referencia.shape)
print("Actual:", df_actual.shape)
print()
print("Filas por ventana:")
print(df_actual["ventana"].value_counts().sort_index())
print()
print("% que paga a tiempo - referencia:", round(df_referencia["Pago_atiempo"].mean() * 100, 2))
print("% que paga a tiempo - actual:", round(df_actual["Pago_atiempo"].mean() * 100, 2))
# %%



# %% Imports adicionales para el modelo
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

# %% Definir columnas (idénticas a ft_engineering.py)
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
columnas_features = columnas_numericas + columnas_nominales + columnas_ordinales
target = "Pago_atiempo"

# %% Armar el pipeline completo: preprocesamiento + Random Forest
preprocesador = ColumnTransformer(transformers=[
    ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), columnas_numericas),
    ("nom", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OneHotEncoder(handle_unknown="ignore"))]), columnas_nominales),
    ("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                      ("encoder", OrdinalEncoder(categories=[["Decreciente", "Estable", "Creciente"]]))]),
     columnas_ordinales)
])

modelo_monitoreo = Pipeline(steps=[
    ("preprocesador", preprocesador),
    ("modelo", RandomForestClassifier(random_state=42))
])

# %% Entrenar solo con la población de referencia y predecir sobre ambas poblaciones
modelo_monitoreo.fit(df_referencia[columnas_features], df_referencia[target])

for d in (df_referencia, df_actual):
    d["prediccion"] = modelo_monitoreo.predict(d[columnas_features])
    # probabilidad de que el cliente pague a tiempo (clase 1)
    d["prob_pago"] = modelo_monitoreo.predict_proba(d[columnas_features])[:, 1]

# %% Revisar las predicciones
print("Prob. media de pago - referencia:", round(df_referencia["prob_pago"].mean(), 4))
print("Prob. media de pago - actual:", round(df_actual["prob_pago"].mean(), 4))
print("ROC-AUC sobre la población actual:",
      round(roc_auc_score(df_actual[target], df_actual["prob_pago"]), 4))
# %%


# %% Funciones de drift para variables numéricas

# %% Función PSI corregida

def calcular_psi(referencia, actual, bins=10):
    """
    Population Stability Index (PSI).
    - Si la variable tiene pocos valores distintos (<= bins), cada valor es un tramo.
    - Si no, los tramos se arman con percentiles de la referencia.
    - Si los cortes se colapsan (ej: variable casi toda en 0), devuelve NaN
      en vez de un 0 engañoso.
    """
    referencia = referencia.dropna()
    actual = actual.dropna()

    if referencia.nunique() <= bins:
        categorias = sorted(set(referencia.unique()) | set(actual.unique()))
        prop_ref = referencia.value_counts(normalize=True).reindex(categorias, fill_value=0).values
        prop_act = actual.value_counts(normalize=True).reindex(categorias, fill_value=0).values
    else:
        cortes = np.unique(np.percentile(referencia, np.linspace(0, 100, bins + 1)))
        if len(cortes) < 3:
            return np.nan
        cortes[0], cortes[-1] = -np.inf, np.inf
        prop_ref = np.histogram(referencia, bins=cortes)[0] / len(referencia)
        prop_act = np.histogram(actual, bins=cortes)[0] / len(actual)

    prop_ref = np.clip(prop_ref, 1e-6, None)
    prop_act = np.clip(prop_act, 1e-6, None)
    return float(np.sum((prop_act - prop_ref) * np.log(prop_act / prop_ref)))

def calcular_js(referencia, actual, bins=10):
    """
    Distancia de Jensen-Shannon entre las dos distribuciones (va de 0 a 1).
    scipy devuelve la raíz de la divergencia, por eso se llama "distancia".
    """
    referencia = referencia.dropna()
    actual = actual.dropna()
    cortes = np.histogram_bin_edges(pd.concat([referencia, actual]), bins=bins)
    p = np.histogram(referencia, bins=cortes)[0].astype(float)
    q = np.histogram(actual, bins=cortes)[0].astype(float)
    return float(jensenshannon(p / p.sum(), q / q.sum(), base=2))


def drift_numerico(referencia, actual):
    """Calcula KS, PSI y Jensen-Shannon para una variable numérica."""
    ks_stat, ks_pvalue = ks_2samp(referencia.dropna(), actual.dropna())
    return {
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "psi": calcular_psi(referencia, actual),
        "js": calcular_js(referencia, actual),
    }

# %% Probar las funciones sobre todas las variables numéricas
variables_a_revisar = columnas_numericas + ["prob_pago"]

resultados_numericos = []
for col in variables_a_revisar:
    r = drift_numerico(df_referencia[col], df_actual[col])
    r["variable"] = col
    resultados_numericos.append(r)

tabla_drift_num = (pd.DataFrame(resultados_numericos)
                   .set_index("variable")
                   .sort_values("psi", ascending=False)
                   .round(4))
tabla_drift_num
# %%


# %% Variables a monitorear (sin mes/año, que difieren por construcción)
columnas_monitoreo = [c for c in columnas_numericas if c not in ("mes_prestamo", "anio_prestamo")]
variables_a_revisar = columnas_monitoreo + ["prob_pago"]

resultados_numericos = []
for col in variables_a_revisar:
    r = drift_numerico(df_referencia[col], df_actual[col])
    r["variable"] = col
    resultados_numericos.append(r)

tabla_drift_num = (pd.DataFrame(resultados_numericos)
                   .set_index("variable")
                   .sort_values("psi", ascending=False)
                   .round(4))
print(tabla_drift_num.to_string())

# %% Revisar los nulos de puntaje en cada población
print("% nulos en puntaje - referencia:", round(df_referencia["puntaje"].isnull().mean() * 100, 1))
print("% nulos en puntaje - actual:", round(df_actual["puntaje"].isnull().mean() * 100, 1))
print("Filas con puntaje real - referencia:", df_referencia["puntaje"].notna().sum())
print("Filas con puntaje real - actual:", df_actual["puntaje"].notna().sum())
# %%


# %% Verificación: ¿el valor de relleno de puntaje se asocia con pagar o no?
# True = puntaje nulo (era el relleno 95.227787), False = puntaje real
df_completo = pd.concat([df_referencia, df_actual])
print(pd.crosstab(df_completo["puntaje"].isnull(), df_completo["Pago_atiempo"],
                  normalize="index").round(3) * 100)
print()
print(pd.crosstab(df_completo["puntaje"].isnull(), df_completo["Pago_atiempo"]))
# %%

# %% [markdown]
# ### Hallazgo: el relleno de `puntaje` se asocia con el target
#
# Las 9.407 filas donde `puntaje` tenía el valor de relleno (95.227787, tratado como nulo)
# corresponden en su totalidad a clientes que pagaron a tiempo (0 casos de no pago).
# Todos los clientes que no pagaron (511) tienen un puntaje real.
# Esto implica que la ausencia de puntaje funciona como señal directa del target,
# lo que explica el ROC-AUC de 1.0 y es una forma de data leakage.
# La tasa de nulos de `puntaje` subió de 85,9% (referencia) a 91,2% (actual) y se
# monitorea como alerta prioritaria.
# %%


# %% Función de drift para variables categóricas

def drift_categorico(referencia, actual, umbral_raras=0.01):
    """
    Chi-cuadrado de homogeneidad entre referencia y actual para una variable categórica.
    - Los nulos pasan a ser la categoría "Sin dato".
    - Las categorías con menos de umbral_raras (1%) en la referencia se agrupan en "Otros".
    - Cramér's V indica el tamaño del cambio (0 = sin cambio, 1 = cambio total).
    """
    ref = referencia.astype(object).where(referencia.notna(), "Sin dato").astype(str)
    act = actual.astype(object).where(actual.notna(), "Sin dato").astype(str)

    frecuentes = ref.value_counts(normalize=True)
    frecuentes = frecuentes[frecuentes >= umbral_raras].index
    ref = ref.where(ref.isin(frecuentes), "Otros")
    act = act.where(act.isin(frecuentes), "Otros")

    valores = pd.concat([ref, act], ignore_index=True)
    grupo = pd.Series(["referencia"] * len(ref) + ["actual"] * len(act))
    tabla = pd.crosstab(valores, grupo)

    chi2, p_valor, _, _ = chi2_contingency(tabla)
    n = tabla.values.sum()
    v_cramer = np.sqrt(chi2 / (n * (min(tabla.shape) - 1)))
    return {"chi2": chi2, "chi2_pvalue": p_valor, "cramers_v": v_cramer}

# %% Agregar la variable "puntaje_nulo" (con o sin puntaje real)
for d in (df_referencia, df_actual):
    d["puntaje_nulo"] = np.where(d["puntaje"].isnull(), "Sin puntaje", "Con puntaje")

# %% Calcular el drift de las variables categóricas
columnas_categoricas_monitoreo = ["tipo_laboral", "tipo_credito", "tendencia_ingresos", "puntaje_nulo"]

resultados_categoricos = []
for col in columnas_categoricas_monitoreo:
    r = drift_categorico(df_referencia[col], df_actual[col])
    r["variable"] = col
    resultados_categoricos.append(r)

tabla_drift_cat = (pd.DataFrame(resultados_categoricos)
                   .set_index("variable")
                   .sort_values("cramers_v", ascending=False)
                   .round(4))
print(tabla_drift_cat.to_string())
# %%


# %% Drift por ventana: cada ventana actual contra la referencia
variables_num = [c for c in columnas_numericas if c not in ("mes_prestamo", "anio_prestamo")] + ["prob_pago"]
ventanas = sorted(df_actual["ventana"].unique())

filas = []
for v in ventanas:
    sub = df_actual[df_actual["ventana"] == v]

    # variables numéricas: KS, PSI y Jensen-Shannon
    for col in variables_num:
        r = drift_numerico(df_referencia[col], sub[col])
        filas.append({"ventana": v, "variable": col, "tipo": "numerica", "n": len(sub),
                      "ks_stat": r["ks_stat"], "psi": r["psi"], "js": r["js"], "cramers_v": np.nan})

    # variables categóricas: Chi-cuadrado y V de Cramér
    for col in columnas_categoricas_monitoreo:
        r = drift_categorico(df_referencia[col], sub[col])
        filas.append({"ventana": v, "variable": col, "tipo": "categorica", "n": len(sub),
                      "ks_stat": np.nan, "psi": np.nan, "js": np.nan, "cramers_v": r["cramers_v"]})

tabla_drift_ventanas = pd.DataFrame(filas)

# %% PSI por variable a lo largo del tiempo
pivot_psi = (tabla_drift_ventanas[tabla_drift_ventanas["tipo"] == "numerica"]
             .pivot(index="variable", columns="ventana", values="psi").round(3))
print(pivot_psi.to_string())

# %% V de Cramér por variable categórica a lo largo del tiempo
pivot_cramer = (tabla_drift_ventanas[tabla_drift_ventanas["tipo"] == "categorica"]
                .pivot(index="variable", columns="ventana", values="cramers_v").round(3))
print(pivot_cramer.to_string())

# %% Resumen por ventana: tamaño, nulos de puntaje, predicciones y tasa de pago real
resumen_ventanas = (df_actual.groupby("ventana")
                    .agg(filas=("Pago_atiempo", "size"),
                         pct_nulos_puntaje=("puntaje", lambda s: s.isnull().mean() * 100),
                         prob_pago_media=("prob_pago", "mean"),
                         pct_pago_real=("Pago_atiempo", lambda s: s.mean() * 100))
                    .round(2))
print(resumen_ventanas.to_string())
# %%


# %% Umbrales de alerta (ver README: PSI es convención de la industria, el resto son criterios propios)
UMBRALES = {
    "psi": (0.10, 0.25),
    "ks_stat": (0.10, 0.20),
    "cramers_v": (0.10, 0.30),
}
MIN_FILAS_VALIDAS = 30          # menos filas válidas que esto: no se evalúa
DIF_NULOS_PUNTAJE = 5           # puntos porcentuales de aumento de nulos que disparan alerta
MIN_VARIABLES_CRITICAS = 3      # cantidad de variables críticas que sugiere reentrenar

def nivel_alerta(valor, metrica):
    """Devuelve 0 (estable), 1 (atención) o 2 (crítico); None si el valor es NaN."""
    if pd.isna(valor):
        return None
    bajo, alto = UMBRALES[metrica]
    if valor >= alto:
        return 2
    if valor >= bajo:
        return 1
    return 0

# %% Agregar la cantidad de filas válidas por variable y ventana
def contar_validos(fila):
    sub = df_actual[df_actual["ventana"] == fila["ventana"]]
    if fila["tipo"] == "numerica":
        return int(sub[fila["variable"]].notna().sum())
    return len(sub)   # en categóricas los nulos cuentan como categoría

tabla_drift_ventanas["n_validos"] = tabla_drift_ventanas.apply(contar_validos, axis=1)

# %% Estado (semáforo) de cada variable en cada ventana
def estado_fila(fila):
    if fila["n_validos"] < MIN_FILAS_VALIDAS:
        return "Muestra insuficiente"
    if fila["tipo"] == "categorica":
        nivel = nivel_alerta(fila["cramers_v"], "cramers_v")
    else:
        nivel = nivel_alerta(fila["psi"], "psi")
        if nivel is None:                       # PSI no calculable (ej: saldo_mora)
            nivel = nivel_alerta(fila["ks_stat"], "ks_stat")
    return ["Estable", "Atención", "Crítico"][nivel]

tabla_drift_ventanas["estado"] = tabla_drift_ventanas.apply(estado_fila, axis=1)

estado_pivot = tabla_drift_ventanas.pivot(index="variable", columns="ventana", values="estado")
print(estado_pivot.to_string())

# %% Alertas y recomendaciones automáticas por ventana
def generar_alertas(ventana):
    sub = tabla_drift_ventanas[tabla_drift_ventanas["ventana"] == ventana]
    mensajes = []

    criticas = sub[sub["estado"] == "Crítico"]["variable"].tolist()
    insuficientes = sub[sub["estado"] == "Muestra insuficiente"]["variable"].tolist()

    if criticas:
        mensajes.append(f"CRÍTICO - drift alto en: {', '.join(criticas)}. Revisar estas variables.")
    if "prob_pago" in criticas or len(criticas) >= MIN_VARIABLES_CRITICAS:
        mensajes.append("RECOMENDACIÓN - se sugiere evaluar un reentrenamiento del modelo.")

    # Alerta específica: aumento de nulos en puntaje (variable con riesgo de data leakage)
    pct_ref = df_referencia["puntaje"].isnull().mean() * 100
    pct_ventana = df_actual.loc[df_actual["ventana"] == ventana, "puntaje"].isnull().mean() * 100
    if pct_ventana - pct_ref >= DIF_NULOS_PUNTAJE:
        mensajes.append(f"ALERTA - nulos de puntaje: {pct_ventana:.1f}% vs {pct_ref:.1f}% en referencia.")

    if insuficientes:
        mensajes.append(f"AVISO - muestra insuficiente (menos de {MIN_FILAS_VALIDAS} filas válidas) en: "
                        f"{', '.join(insuficientes)}. No se evalúan.")
    if not mensajes:
        mensajes.append("Sin alertas.")
    return mensajes

for v in ventanas:
    print(f"\n=== Ventana {v} ===")
    for m in generar_alertas(v):
        print(" -", m)
# %%



# %%
