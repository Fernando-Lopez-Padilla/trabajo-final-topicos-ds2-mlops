# ============================================================
# Gatillo de reentrenamiento
# Trabajo Final - Tópicos en Data Science II
# ============================================================

import pandas as pd
from huggingface_hub import hf_hub_download


REPO_ID = "Alucard08/deis-atenciones-urgencia-processed-2018-2023"

# Descargar resultados de monitoreo
ruta = hf_hub_download(
    repo_id=REPO_ID,
    filename="monitoreo.csv",
    repo_type="dataset"
)

monitoreo = pd.read_csv(ruta)

print("\nEvaluación del gatillo de reentrenamiento:\n")

for _, fila in monitoreo.iterrows():

    anio = int(fila["anio"])
    psi = fila["PSI_maximo"]
    variacion_mae = fila["variacion_MAE_pct"]

    # Regla definida para el proyecto
    if psi > 0.20 and variacion_mae > 20:

        estado = "🚨 ACTIVAR REVISIÓN PARA REENTRENAMIENTO"

    elif psi > 0.20:

        estado = "⚠️ MANTENER MODELO Y MONITOREAR"

    else:

        estado = "✅ OPERACIÓN NORMAL"

    print(
        f"{anio} | PSI: {psi:.2f} | "
        f"Variación MAE: {variacion_mae:.1f}% | {estado}"
    )