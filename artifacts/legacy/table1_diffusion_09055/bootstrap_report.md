# Intervalo bootstrap del 95 % para la balanced accuracy 0.9055 (experimento DEO)

Generado: 2026-09-22 13:30:25

Protocolo: `SnsEnc/docs/protocolo_bootstrap_deo_09055.md`  
Ejecucion historica: `eatdrinkanother_94u_1f_autoencoder_s_20251028-174115`  
Checkpoint: `best_model_eatdrinkanother_94u_20251028-174115.keras`

## 1. Reproduccion del resultado historico

La particion de prueba se reconstruyo a partir de los sujetos **guardados** en
`eatdrinkanother_94u_1.txt`; no se genero una particion aleatoria nueva.
Se respetaron los argumentos efectivos del flujo archivado (`overlap=False`, `seg5=True`,
ventanas de 500 x 9, sin la rama de normalizacion que solo aplica a datasets `de*`).
La inferencia se ejecuto una vez y las salidas se convirtieron a clase con `argmax`.

| Propiedad | Historico | Reproducido |
|---|---:|---:|
| Sujetos de prueba | 29 | 29 |
| Ventanas de prueba | 71.591 | 71.591 |
| Clase 0: DRINK | 4.933 | 4.933 |
| Clase 1: EAT | 5.243 | 5.243 |
| Clase 2: ANOTHER | 61.415 | 61.415 |
| Balanced accuracy | 0.9055 | 0.9055 |

Balanced accuracy con precision completa: **0.905541577683**.

Recalls por clase reproducidos: DRINK = 0.8849, EAT = 0.9014, ANOTHER = 0.9304.
Valores historicos del archivo de metricas: DRINK = 0.8849, EAT = 0.9014, ANOTHER = 0.9304.

Matriz de confusion reproducida (filas = clase real, columnas = clase predicha):

| real \\ pred | DRINK | EAT | ANOTHER | total |
|---|---:|---:|---:|---:|
| **DRINK** | 4365 | 310 | 258 | 4933 |
| **EAT** | 304 | 4726 | 213 | 5243 |
| **ANOTHER** | 3121 | 1155 | 57139 | 61415 |

Verificaciones automaticas del Paso 1: 20 de 20 superadas.

## 2. Intervalo bootstrap por sujetos

- Unidad de remuestreo: sujeto de prueba completo (cluster bootstrap).
- Replicas: 10.000.
- Semilla: 20251028 (`numpy.random.default_rng (PCG64), integers(0, 29) con reemplazo`).
- Intervalo principal: percentil bilateral del 95 %.
- Modelo fijo en todas las replicas; clases fijas `[0, 1, 2]`.
- No se remuestrearon ventanas individuales ni ejemplos sinteticos; ver ClusterBootstrap:
  <https://pmc.ncbi.nlm.nih.gov/articles/PMC7148287/>.

**Balanced accuracy observada: 0.9055.**

**IC95 percentil: [0.8883, 0.9223].**

Amplitud del intervalo: 0.0340. Distancia al limite inferior: 0.0173; al superior: 0.0168
(el intervalo puede ser asimetrico respecto de la estimacion puntual).

Intervalos percentil del 95 % de los recalls por clase:

| Clase | Recall observado | IC95 |
|---|---:|---:|
| DRINK | 0.8849 | [0.8487, 0.9160] |
| EAT | 0.9014 | [0.8729, 0.9279] |
| ANOTHER | 0.9304 | [0.9149, 0.9442] |

Diagnosticos de las replicas (no sustituyen a la estimacion puntual):

- Media bootstrap: 0.9055 (sesgo aparente -0.00006 respecto de la BA observada).
- Desviacion estandar bootstrap: 0.0087. Mediana: 0.9056. Rango: [0.8701, 0.9368].
- Replicas invalidas (alguna clase sin soporte): 0.
- Sujetos de prueba sin alguna clase real: ninguno.

La estimacion puntual publicada es la BA observada, 0.9055. Tener 10.000 replicas reduce el error
numerico del bootstrap; la informacion independiente sigue proviniendo de 29 sujetos.

## 3. Configuraciones comparadas

- Ejecuciones en `recurrent_models_20251026-024834`: **96**.
- Ejecuciones con evaluacion de prueba: **96**.
- Particion de prueba identica en todas las ejecuciones: **si** (md5 f6a8685edc340018d792f59bb337697d).
- Rejilla: 3 modelos preentrenados x 2 tasas de aprendizaje x 4 activaciones x 4 modos de cabeza.
  - Activaciones: gelu, relu, silu, tanh.
  - Modos de cabeza: balanced, classic, deep, light.
  - Tasas de aprendizaje: 0.0001, 1e-05.

La ejecucion seleccionada (`eatdrinkanother_94u_1f_autoencoder_s_20251028-174115`, configuracion 19) ocupa el **puesto 1 de 96** por balanced
accuracy de PRUEBA (0.9055) y el puesto 6 por balanced accuracy de validacion (0.9075).
Es decir, la configuracion se eligio maximizando la metrica del propio conjunto de prueba.

Diez configuraciones con mayor balanced accuracy de prueba:

| # | Ejecucion | Preentrenado | lr | activacion | cabeza | BA val | BA test |
|---:|---|---|---|---|---|---:|---:|
| 1 | `20251028-174115` **(seleccionada)** | recurrent_models_20251012-100533 | 0.0001 | gelu | deep | 0.9075 | 0.9055 |
| 2 | `20251029-095836` | recurrent_models_20251012-100533 | 0.0001 | tanh | classic | 0.9081 | 0.8986 |
| 3 | `20251029-000735` | recurrent_models_20251012-100533 | 0.0001 | tanh | balanced | 0.9025 | 0.8964 |
| 4 | `20251029-204319` | recurrent_models_20251012-100533 | 0.0001 | relu | deep | 0.9004 | 0.8952 |
| 5 | `20251030-064931` | recurrent_models_20251012-100533 | 0.0001 | silu | light | 0.9142 | 0.8945 |
| 6 | `20251030-032319` | recurrent_models_20251012-100533 | 0.0001 | silu | balanced | 0.9092 | 0.8943 |
| 7 | `20251029-063520` | recurrent_models_20251012-100533 | 0.0001 | tanh | deep | 0.8970 | 0.8935 |
| 8 | `20251029-133102` | recurrent_models_20251012-100533 | 0.0001 | relu | balanced | 0.9054 | 0.8925 |
| 9 | `20251028-111518` | recurrent_models_20251012-100533 | 0.0001 | gelu | balanced | 0.9012 | 0.8924 |
| 10 | `20251028-203129` | recurrent_models_20251012-100533 | 0.0001 | gelu | classic | 0.9076 | 0.8907 |

El bootstrap principal mantiene fija la ejecucion historica. Reseleccionar el maximo entre
configuraciones en cada replica estimaria otra cantidad y no corregiria por si mismo el sesgo
de seleccion, y no se usa la configuracion con el intervalo mas favorable para redefinir el
resultado principal. Si hubo comparaciones adicionales fuera de esta carpeta, deben
incorporarse a la descripcion del proceso de seleccion.

## 4. Interpretacion y limitaciones

Los sujetos de prueba quedaron excluidos de todas las etapas de entrenamiento, incluido el
generador sintetico, segun la confirmacion del investigador. Sin embargo, la configuracion de
fine-tuning se eligio comparando la balanced accuracy de prueba entre las 96 configuraciones
evaluadas. El test no intervino en el ajuste de pesos, pero si en la seleccion.

Por tanto el intervalo es **exploratorio**, para el modelo seleccionado en ese mismo test. No
debe interpretarse como un IC confirmatorio con cobertura del 95 % garantizada despues de la
seleccion; el bootstrap no elimina ese optimismo. El analisis cuantifica solo la variabilidad
asociada al muestreo de sujetos de prueba: no incorpora la variabilidad de la generacion
sintetica, del preentrenamiento ni del fine-tuning, y no corrige el sesgo de seleccion.

Para una evaluacion confirmatoria haria falta un conjunto independiente de sujetos que no haya
intervenido en esa seleccion, con la configuracion ahora fija. Como alternativa, una validacion
cruzada anidada por sujetos evaluaria el procedimiento completo, pero constituye otro
experimento y no produce un intervalo para el 0.9055 historico.

## 5. Texto propuesto para comunicar el resultado

> El modelo CaBiGRU preentrenado con datos sinteticos de beber y comer y ajustado con los
> datos originales de entrenamiento obtuvo una balanced accuracy de 0.9055 en 71.591 ventanas
> de 29 sujetos de prueba. Se estimo un intervalo bootstrap percentil del 95 % de
> [0.8883, 0.9223] mediante 10.000 remuestreos de sujetos completos, manteniendo fijo el modelo.
> Los sujetos de prueba quedaron excluidos de todo entrenamiento. Dado que la configuracion
> se selecciono por su desempeno en este mismo test entre las 96 configuraciones evaluadas,
> el analisis es exploratorio y el intervalo no corrige el sesgo de seleccion ni incorpora
> la variabilidad del entrenamiento.

## 6. Reproducibilidad

| Elemento | Valor |
|---|---|
| Python | 3.10.18 |
| TensorFlow | 2.20.0 |
| Keras | 3.12.1 |
| NumPy | 2.2.6 |
| pandas | 2.2.3 |
| scikit-learn | 1.6.1 |
| Nodo | dl-08 |
| GPU | /physical_device:GPU:0 |
| sha256 checkpoint | `7fd7b815bf6e09154981ac50207b88459ecac69ae00e5430ab92d4e2138e7fa5` |
| sha256 csv fuente | `2fb9f4d045ad841b684730bdaeafb7354748275c9da39bc048ecfb23ef5aa6e6` |
| sha256 archivo de particion | `c06e6c4d2da9bf65ca218fb368eeb5488d6d910d2cebf5d4ac330688ac4ae379` |

Comando del Paso 1: `/home/elian.riveros/miniconda3/envs/a2-project/bin/python /home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/bootstrap_deo_09055/scripts/01_export_test_predictions.py --out-dir /home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/bootstrap_deo_09055/results/bootstrap_20260922-132658_95781`

Comando del Paso 2: `/home/elian.riveros/miniconda3/envs/a2-project/bin/python /home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/bootstrap_deo_09055/scripts/02_run_bootstrap.py --out-dir /home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main/bootstrap_deo_09055/results/bootstrap_20260922-132658_95781`

### Entregables

- `test_predictions.parquet` (2.2 MB)
- `subject_confusion_matrices.csv` (0.0 MB)
- `bootstrap_replicates.csv` (0.7 MB)
- `bootstrap_summary.json` (0.0 MB)
- `configuration_inventory.csv` (0.1 MB)
- `configuration_inventory_summary.json` (0.0 MB)
- `reproduction_manifest.json` (0.0 MB)

