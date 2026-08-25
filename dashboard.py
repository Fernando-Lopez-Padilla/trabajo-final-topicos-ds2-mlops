# ============================================================
# Dashboard MLOps - Atenciones de Urgencia en Chile
# Trabajo Final - Tópicos en Data Science II
# ============================================================

import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download


# ------------------------------------------------------------
# 1. Configuración general
# ------------------------------------------------------------

st.set_page_config(
    page_title="Monitoreo MLOps - Urgencias Chile",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 Monitoreo MLOps de Atenciones de Urgencia")
st.caption(
    "Predicción y monitoreo de Data Drift — Chile, 2018–2023"
)


# ------------------------------------------------------------
# 2. Carga de datos desde Hugging Face
# ------------------------------------------------------------

REPO_ID = "Alucard08/deis-atenciones-urgencia-processed-2018-2023"


@st.cache_data
def cargar_datos(nombre_archivo):

    ruta = hf_hub_download(
        repo_id=REPO_ID,
        filename=nombre_archivo,
        repo_type="dataset"
    )

    return pd.read_csv(ruta)


monitoreo = cargar_datos("monitoreo.csv")
drift = cargar_datos("drift_por_variable.csv")
predicciones = cargar_datos("predicciones.csv")
serie = cargar_datos("urgencias_semanales_2018_2023.csv")


# Conversión de fechas
serie["fecha"] = pd.to_datetime(serie["fecha"])
predicciones["fecha"] = pd.to_datetime(predicciones["fecha"])


# ------------------------------------------------------------
# 3. Selección de ventana de producción
# ------------------------------------------------------------

st.sidebar.header("Configuración")

anio = st.sidebar.selectbox(
    "Ventana de producción",
    sorted(monitoreo["anio"].unique())
)

fila = monitoreo[
    monitoreo["anio"] == anio
].iloc[0]

drift_anio = drift[
    drift["anio"] == anio
].copy()

pred_anio = predicciones[
    predicciones["anio"] == anio
].copy()


# ------------------------------------------------------------
# 4. Indicadores principales
# ------------------------------------------------------------

st.subheader(f"Estado del sistema — {anio}")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Estado",
    fila["estado_general"]
)

col2.metric(
    "PSI máximo",
    f'{fila["PSI_maximo"]:.2f}'
)

col3.metric(
    "Variación MAE",
    f'{fila["variacion_MAE_pct"]:.1f}%'
)

col4.metric(
    "Variables con drift",
    int(fila["variables_con_drift"])
)


# ------------------------------------------------------------
# 5. Alerta y acción recomendada
# ------------------------------------------------------------

if fila["estado_general"] == "🚨 Crítico":

    st.error(
        "🚨 Se detectó Data Drift junto con un deterioro "
        "significativo del desempeño del modelo."
    )

    st.warning(
        "Acción recomendada: revisar los datos y evaluar "
        "el reentrenamiento del modelo."
    )

elif fila["estado_general"] == "⚠️ Drift":

    st.warning(
        "⚠️ Se detectó Data Drift, pero el desempeño "
        "del modelo se mantiene dentro del rango definido."
    )

    st.info(
        "Acción recomendada: mantener el modelo "
        "y aumentar el monitoreo."
    )

else:

    st.success(
        "✅ El sistema se encuentra estable."
    )


# ------------------------------------------------------------
# 6. Evolución histórica
# ------------------------------------------------------------

st.divider()

st.subheader("Evolución histórica de las atenciones de urgencia")

serie_grafico = (
    serie[
        ["fecha", "atenciones"]
    ]
    .set_index("fecha")
)

st.line_chart(
    serie_grafico,
    y="atenciones",
    height=350
)

st.caption(
    "La serie permite observar el cambio abrupto registrado "
    "durante 2020 y la recuperación posterior."
)


# ------------------------------------------------------------
# 7. Predicción vs demanda real
# ------------------------------------------------------------

st.divider()

st.subheader(f"Predicción vs demanda real — {anio}")

comparacion = (
    pred_anio[
        ["fecha", "atenciones", "prediccion"]
    ]
    .rename(
        columns={
            "atenciones": "Real",
            "prediccion": "Predicción"
        }
    )
    .set_index("fecha")
)

st.line_chart(
    comparacion,
    height=350
)


# ------------------------------------------------------------
# 8. Data Drift por variable
# ------------------------------------------------------------

st.divider()

st.subheader(f"Data Drift por variable — {anio}")

tabla_drift = drift_anio[
    ["variable", "PSI", "estado"]
].copy()

tabla_drift["PSI"] = tabla_drift["PSI"].round(3)

st.dataframe(
    tabla_drift,
    use_container_width=True,
    hide_index=True
)

grafico_psi = (
    drift_anio[
        ["variable", "PSI"]
    ]
    .set_index("variable")
)

st.bar_chart(
    grafico_psi,
    y="PSI",
    height=300
)

st.caption(
    "PSI < 0.10: estable | "
    "0.10–0.20: cambio moderado | "
    "PSI > 0.20: drift significativo"
)


# ------------------------------------------------------------
# 9. Performance del modelo por año
# ------------------------------------------------------------

st.divider()

st.subheader("Desempeño del modelo por ventana de producción")

tabla_performance = monitoreo[
    [
        "anio",
        "MAE_referencia",
        "MAE_modelo",
        "variacion_MAE_pct",
        "estado_general"
    ]
].copy()

tabla_performance["MAE_referencia"] = (
    tabla_performance["MAE_referencia"].round(0)
)

tabla_performance["MAE_modelo"] = (
    tabla_performance["MAE_modelo"].round(0)
)

tabla_performance["variacion_MAE_pct"] = (
    tabla_performance["variacion_MAE_pct"].round(1)
)

st.dataframe(
    tabla_performance,
    use_container_width=True,
    hide_index=True
)


# ------------------------------------------------------------
# 10. Ayuda para interpretación
# ------------------------------------------------------------

with st.expander("¿Cómo interpretar el monitoreo?"):

    st.markdown(
        """
        **Data Drift:** indica que las variables que recibe el modelo
        presentan una distribución diferente de aquella utilizada
        durante el entrenamiento.

        **Deterioro de performance:** indica que el error del modelo
        aumentó respecto del período de referencia 2018–2019.

        El sistema considera un estado **crítico** cuando se presentan
        simultáneamente:

        - PSI superior a 0.20.
        - Incremento del MAE superior al 20 %.

        Si existe drift pero el desempeño continúa estable, se genera
        una advertencia de monitoreo sin activar automáticamente el
        reentrenamiento.
        """
    )