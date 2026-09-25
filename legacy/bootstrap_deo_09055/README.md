# Bootstrap por sujetos para la balanced accuracy 0.9055 (experimento DEO)

Implementacion del protocolo `SnsEnc/docs/protocolo_bootstrap_deo_09055.md`.

Estima un intervalo bootstrap percentil del 95 % para la balanced accuracy de prueba del
modelo CaBiGRU preentrenado con datos sinteticos de beber y comer y ajustado con los datos
originales de entrenamiento. El modelo permanece fijo; solo se remuestrean sujetos de prueba.

## Como se lanza

```bash
cd /home/elian.riveros/dl-13-elian/notebooks/workspaces/lstm-har/MultiTask-LSTM-HAR-main
sbatch submit_bootstrap_deo_09055.sh
```

Variables opcionales:

- `STAGES="2 4"` limita las etapas que se ejecutan (por defecto `1 2 3 4`).
- `RUN_TAG=mi_etiqueta` fija el nombre de la carpeta de salida.

Las salidas van a `bootstrap_deo_09055/results/bootstrap_<fecha>_<jobid>/` y los logs de
slurm a `bootstrap_deo_09055/logs/`.

## Etapas

| Etapa | Script | Que hace |
|---|---|---|
| 1 | `scripts/01_export_test_predictions.py` | Reconstruye la particion de prueba original, ejecuta inferencia una vez y exporta predicciones por ventana con su sujeto |
| 2 | `scripts/02_run_bootstrap.py` | Bootstrap por sujetos: 10.000 replicas, semilla 20251028, percentil bilateral del 95 % |
| 3 | `scripts/03_configuration_inventory.py` | Inventario de las configuraciones del barrido y criterio de seleccion |
| 4 | `scripts/04_make_report.py` | `report.md` con reproduccion, intervalo, interpretacion y limitaciones |

`scripts/common.py` concentra rutas, constantes del protocolo y los valores historicos.

## Artefactos historicos (solo lectura)

```
saved_models_prev/202508-202512/recurrent_models_20251026-024834/
  eatdrinkanother_94u_1f_autoencoder_s_20251028-174115/
    best_model_eatdrinkanother_94u_20251028-174115.keras   <- checkpoint evaluado
    eatdrinkanother_94u_1.txt                              <- sujetos guardados
    hyperparams_19.txt                                     <- configuracion + preentrenado
    metrics_test_eatdrinkanother_94u_20251028-174115.txt   <- metricas historicas
```

Dataset fuente (10,3 GB, 246.329 ventanas de 94 sujetos):

```
/home/elian.riveros/dl-13-elian/notebooks/workspaces/files/
  fullraws3_vivabem012_drink0eat1another2_94u_nowspr_noaer_acc_gyr_mag__coord__5sec_100hz.csv
```

Ninguno de estos archivos se modifica.

## Decisiones de reproduccion

- **La particion no se vuelve a sortear.** El flujo archivado (`utils.split_data_val`) genera la
  particion con `train_test_split` sin `random_state`; el protocolo prohibe recrearla aunque se
  use la misma semilla. Se leen los sujetos guardados en `eatdrinkanother_94u_1.txt` y se
  filtran las filas del csv por sujeto, conservando el orden original del archivo.
- **Sujeto por ventana.** `utils.get_processed_fold` calcula `users_test` pero deja comentada la
  asignacion a `dict_arrays`. Aqui el sujeto se toma de la columna 0 de la misma fila del csv,
  que con `seg5=True` corresponde 1:1 con la ventana de 500x9. `uuid_test` (29 sujetos unicos)
  no sustituye a ese vector de longitud 71.591.
- **`overlap=False`.** Es el argumento efectivo del llamador archivado, aunque
  `hyperparams_19.txt` registre `overlap_shift=0.5`.
- **Sin normalizacion adicional.** La rama de z-score de `get_processed_fold` solo se aplica a
  datasets cuyo nombre empieza por `de`; `eatdrinkanother_94u` no la activa.
- **Etiquetas.** `dict2` se construye sobre `np.unique(df[2])` de todo el dataframe, igual que el
  original. En este dataset los valores globales son `[0, 1, 2]`, asi que el mapeo es la identidad.
- **Entorno `a2-project`** (TF 2.20 / Keras 3.12). Los entornos `tensor`/`tensor-clean` llevan
  Keras 2.15 y no pueden cargar este checkpoint, guardado con el formato de Keras 3.
- **`pyarrow` local.** Instalado en `pylibs/` con `pip install --target`; no se modifico el
  entorno conda. El script de slurm lo agrega a `PYTHONPATH`. Si no estuviera disponible, la
  etapa 1 escribe `test_predictions.csv.gz` en su lugar.
- **Sin `TF_FORCE_RUN_EAGERLY`.** A diferencia de `submit_job_rec_main.sh`, que lo activa para
  entrenar, aqui se deja desactivado: el modo eager desactiva los kernels fusionados de cuDNN y
  volveria impracticable la inferencia sobre 71.591 ventanas. La etapa 1 verifica que la
  balanced accuracy reproducida sea `0.9055` a cuatro decimales.

## Verificaciones de aceptacion

La etapa 1 aborta con codigo 3 si falla cualquiera de estas comprobaciones, y no ajusta
predicciones, etiquetas ni preprocesamiento para forzar la coincidencia:

- 55 / 10 / 29 sujetos de train / val / test.
- 71.591 ventanas de prueba, con 4.933 DRINK, 5.243 EAT y 61.415 ANOTHER.
- Los 29 sujetos guardados presentes, `window_id` unico y `x_test` finito.
- Balanced accuracy formateada a cuatro decimales igual a `0.9055`, recalls por clase iguales a
  los del archivo historico y BA igual a la media de los tres recalls.

La etapa 2 se detiene si el total de ventanas o la BA observada no coinciden con lo historico, y
retiene el intervalo principal si aparece alguna replica sin las tres clases reales.

## Entregables

`test_predictions.parquet`, `subject_confusion_matrices.csv`, `bootstrap_replicates.csv`,
`bootstrap_summary.json`, `configuration_inventory.csv`,
`configuration_inventory_summary.json`, `reproduction_manifest.json` y `report.md`.

## Limitacion principal

El intervalo es **exploratorio**. Los sujetos de prueba quedaron fuera de todo entrenamiento,
pero la configuracion de fine-tuning se eligio comparando la balanced accuracy de prueba entre
las 96 configuraciones del barrido, todas evaluadas sobre la misma particion. El bootstrap no
corrige ese optimismo ni incorpora la variabilidad de la generacion sintetica, el
preentrenamiento o el fine-tuning.
