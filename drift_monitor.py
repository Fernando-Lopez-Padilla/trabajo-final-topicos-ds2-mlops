# ============================================================
# Módulo de monitoreo de Data Drift
# Trabajo Final - Tópicos en Data Science II
# ============================================================

import pandas as pd
import numpy as np
from huggingface_hub import hf_hub_download


# ------------------------------------------------------------
# 1. Configuración
# ------------------------------------------------------------

REPO_ID = "Alucard08/deis-atenciones-urgencia-processed-2018-2023"

ARCHIVO_DATOS = "urgencias_semanales_2018_2023.csv"

VARIABLES_DRIFT = [
    "lag_1",
    "lag_2",
    "lag_4",
    "media_4_sem"
]


# ------------------------------------------------------------
# 2. Carga de la serie semanal desde Hugging Face
# ------------------------------------------------------------

ruta_datos = hf_hub_download(
    repo_id=REPO_ID,
    filename=ARCHIVO_DATOS,
    repo_type="dataset"
)

serie_semanal = pd.read_csv(ruta_datos)

serie_semanal["fecha"] = pd.to_datetime(
    serie_semanal["fecha"]
)

serie_semanal = (
    serie_semanal
    .sort_values("fecha")
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 3. Reconstrucción de variables utilizadas para monitoreo
# ------------------------------------------------------------

df_modelo = serie_semanal.copy()

df_modelo["semana_anio"] = (
    df_modelo["fecha"]
    .dt.isocalendar()
    .week
    .astype(int)
)

df_modelo["mes"] = df_modelo["fecha"].dt.month
df_modelo["anio"] = df_modelo["fecha"].dt.year

# Valores históricos de demanda
df_modelo["lag_1"] = df_modelo["atenciones"].shift(1)
df_modelo["lag_2"] = df_modelo["atenciones"].shift(2)
df_modelo["lag_4"] = df_modelo["atenciones"].shift(4)

# Promedio de las cuatro semanas anteriores
df_modelo["media_4_sem"] = (
    df_modelo["atenciones"]
    .shift(1)
    .rolling(window=4)
    .mean()
)

# Eliminar semanas iniciales sin historial suficiente
df_modelo = (
    df_modelo
    .dropna()
    .reset_index(drop=True)
)


# ------------------------------------------------------------
# 4. Período de referencia y ventanas de producción
# ------------------------------------------------------------

referencia = (
    df_modelo[
        df_modelo["anio"] <= 2019
    ]
    .copy()
)

ventanas = {
    anio: df_modelo[
        df_modelo["anio"] == anio
    ].copy()

    for anio in range(2020, 2024)
}


# ------------------------------------------------------------
# 5. Función Population Stability Index (PSI)
# ------------------------------------------------------------

def calcular_psi(referencia, produccion, bins=10):

    referencia = pd.Series(referencia).dropna()
    produccion = pd.Series(produccion).dropna()

    # Crear límites usando cuantiles del período de referencia
    limites = np.unique(
        np.quantile(
            referencia,
            np.linspace(0, 1, bins + 1)
        )
    )

    limites[0] = -np.inf
    limites[-1] = np.inf

    ref_dist = pd.cut(
        referencia,
        bins=limites,
        include_lowest=True
    ).value_counts(
        normalize=True,
        sort=False
    )

    prod_dist = pd.cut(
        produccion,
        bins=limites,
        include_lowest=True
    ).value_counts(
        normalize=True,
        sort=False
    )

    # Evitar divisiones por cero
    epsilon = 1e-6

    ref_dist = np.clip(
        ref_dist.values,
        epsilon,
        None
    )

    prod_dist = np.clip(
        prod_dist.values,
        epsilon,
        None
    )

    psi = np.sum(
        (prod_dist - ref_dist)
        * np.log(prod_dist / ref_dist)
    )

    return psi


# ------------------------------------------------------------
# 6. Cálculo del Data Drift por variable y ventana
# ------------------------------------------------------------

resultados = []

for anio, datos in ventanas.items():

    for variable in VARIABLES_DRIFT:

        psi = calcular_psi(
            referencia[variable],
            datos[variable]
        )

        if psi < 0.10:
            estado = "Estable"

        elif psi <= 0.20:
            estado = "Cambio moderado"

        else:
            estado = "Drift significativo"

        resultados.append(
            {
                "anio": anio,
                "variable": variable,
                "PSI": psi,
                "estado": estado
            }
        )


resultados_drift = pd.DataFrame(resultados)


# ------------------------------------------------------------
# 7. Resumen por ventana
# ------------------------------------------------------------

resumen = (
    resultados_drift
    .groupby("anio")
    .agg(
        PSI_promedio=("PSI", "mean"),
        PSI_maximo=("PSI", "max"),
        variables_con_drift=(
            "estado",
            lambda x: (
                x == "Drift significativo"
            ).sum()
        )
    )
    .reset_index()
)


# ------------------------------------------------------------
# 8. Guardar resultados
# ------------------------------------------------------------

resultados_drift.to_csv(
    "drift_monitor_resultados.csv",
    index=False
)


# ------------------------------------------------------------
# 9. Mostrar resultados
# ------------------------------------------------------------

print("\n==============================================")
print(" MONITOREO DE DATA DRIFT - PSI")
print("==============================================\n")

print("Período de referencia: 2018-2019")
print("Ventanas de producción: 2020-2023\n")

print("PSI por variable:\n")

print(
    resultados_drift
    .round({"PSI": 3})
    .to_string(index=False)
)

print("\nResumen por ventana:\n")

print(
    resumen
    .round(
        {
            "PSI_promedio": 2,
            "PSI_maximo": 2
        }
    )
    .to_string(index=False)
)

print(
    "\n✅ Resultados guardados en "
    "'drift_monitor_resultados.csv'"
)