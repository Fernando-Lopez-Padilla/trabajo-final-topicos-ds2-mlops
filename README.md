# Trabajo Final — Tópicos en Data Science II

## Sistema MLOps para predicción y monitoreo de Data Drift en atenciones de urgencia en Chile

Proyecto desarrollado para la asignatura **Tópicos en Data Science II** del Magíster en Ciencia de Datos de la Universidad de Las Américas.

## Objetivo

Desarrollar un pipeline local de Machine Learning que permita predecir la demanda semanal de atenciones de urgencia en Chile y monitorear cambios en la distribución de los datos (*Data Drift*) y en el desempeño del modelo a través del tiempo.

## Fuente de datos

Los datos utilizados corresponden a registros públicos de atenciones de urgencia publicados por el Departamento de Estadísticas e Información de Salud (DEIS), Ministerio de Salud de Chile.

Período analizado:

- 2018
- 2019
- 2020
- 2021
- 2022
- 2023

### Dataset RAW

Hugging Face:

`Alucard08/deis-atenciones-urgencia-raw-2018-2023`

### Dataset procesado

Hugging Face:

`Alucard08/deis-atenciones-urgencia-processed-2018-2023`

## Modelo

El objetivo predictivo corresponde al número de atenciones de urgencia de la semana siguiente.

Variables utilizadas:

- semana del año
- mes
- demanda de la semana anterior (`lag_1`)
- demanda de dos semanas anteriores (`lag_2`)
- demanda de cuatro semanas anteriores (`lag_4`)
- promedio de las cuatro semanas anteriores (`media_4_sem`)

El modelo seleccionado fue una **Regresión Lineal**, entrenada con información correspondiente al período 2018–2019.

## Monitoreo

El período 2018–2019 se utiliza como referencia.

Las ventanas de producción simuladas corresponden a:

- 2020
- 2021
- 2022
- 2023

El Data Drift se evalúa mediante **Population Stability Index (PSI)**.

Criterios utilizados:

- PSI < 0.10: estable
- PSI entre 0.10 y 0.20: cambio moderado
- PSI > 0.20: drift significativo

El deterioro del modelo se evalúa mediante el cambio porcentual del MAE respecto del período de referencia.

## Regla de alerta

Se considera un estado crítico cuando se cumplen simultáneamente:

- PSI > 0.20
- incremento del MAE > 20 %

En este caso se activa una revisión para posible reentrenamiento.

El reentrenamiento es semi-automático y requiere validación previa de los datos.

## Componentes del proyecto

- `dashboard.py`: dashboard de monitoreo desarrollado con Streamlit.
- `app.py`: API de inferencia desarrollada con FastAPI.
- `retrain_check.py`: evaluación del gatillo de reentrenamiento.
- `modelo_urgencias.joblib`: modelo entrenado.
- `features.json`: variables utilizadas por el modelo.
- `requirements.txt`: dependencias del proyecto.

## Instalación

Clonar el repositorio:

```bash
git clone https://github.com/Fernando-Lopez-Padilla/trabajo-final-topicos-ds2-mlops.git