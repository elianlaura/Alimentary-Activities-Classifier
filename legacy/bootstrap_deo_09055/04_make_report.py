"""Genera report.md: reproduccion, intervalo, interpretacion y limitacion por seleccion."""
import argparse
import json
import os
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


def load(path):
    with open(path) as fh:
        return json.load(fh)


def fmt(x, nd=4):
    return "n/d" if x is None else ("%%.%df" % nd) % x


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    args = p.parse_args()
    d = args.out_dir

    man = load(os.path.join(d, "reproduction_manifest.json"))
    boot = load(os.path.join(d, "bootstrap_summary.json"))
    inv = load(os.path.join(d, "configuration_inventory_summary.json"))
    inv_df = pd.read_csv(os.path.join(d, "configuration_inventory.csv"))

    res, obs, ival, diag = man["results"], boot["observed"], boot["interval"], boot["diagnostics"]
    ci = ival["ci95_balanced_accuracy"]
    env = man["environment"]
    ba = obs["balanced_accuracy"]
    cm = obs["confusion_matrix"]

    L = []
    w = L.append
    w("# Intervalo bootstrap del 95 % para la balanced accuracy 0.9055 (experimento DEO)")
    w("")
    w("Generado: %s" % time.strftime("%Y-%m-%d %H:%M:%S"))
    w("")
    w("Protocolo: `SnsEnc/docs/protocolo_bootstrap_deo_09055.md`  ")
    w("Ejecucion historica: `%s`  " % os.path.basename(C.RUN_DIR))
    w("Checkpoint: `%s`" % os.path.basename(C.CHECKPOINT))
    w("")

    w("## 1. Reproduccion del resultado historico")
    w("")
    w("La particion de prueba se reconstruyo a partir de los sujetos **guardados** en")
    w("`%s`; no se genero una particion aleatoria nueva." % os.path.basename(C.SPLIT_FILE))
    w("Se respetaron los argumentos efectivos del flujo archivado (`overlap=False`, `seg5=True`,")
    w("ventanas de %d x %d, sin la rama de normalizacion que solo aplica a datasets `de*`)."
      % (C.N_TIMESTEPS, C.N_FEATURES))
    w("La inferencia se ejecuto una vez y las salidas se convirtieron a clase con `argmax`.")
    w("")
    w("| Propiedad | Historico | Reproducido |")
    w("|---|---:|---:|")
    w("| Sujetos de prueba | %d | %d |" % (C.HISTORIC["n_test_subjects"], res["n_test_subjects"]))
    w("| Ventanas de prueba | %s | %s |" % (format(C.HISTORIC["n_test_windows"], ",d").replace(",", "."),
                                            format(res["n_test_windows"], ",d").replace(",", ".")))
    for i, name in enumerate(C.CLASS_NAMES):
        w("| Clase %d: %s | %s | %s |" % (
            i, name,
            format(C.HISTORIC["class_counts"][i], ",d").replace(",", "."),
            format(res["class_counts"][str(i)] if str(i) in res["class_counts"] else res["class_counts"][i],
                   ",d").replace(",", ".")))
    w("| Balanced accuracy | %s | %s |" % (C.HISTORIC["balanced_accuracy_4dp"], res["balanced_accuracy_4dp"]))
    w("")
    w("Balanced accuracy con precision completa: **%.12f**." % ba)
    w("")
    w("Recalls por clase reproducidos: " + ", ".join(
        "%s = %.4f" % (n, r) for n, r in zip(C.CLASS_NAMES, res["recalls"])) + ".")
    w("Valores historicos del archivo de metricas: " + ", ".join(
        "%s = %.4f" % (n, r) for n, r in zip(C.CLASS_NAMES, C.HISTORIC["sensitivity_per_class_4dp"])) + ".")
    w("")
    w("Matriz de confusion reproducida (filas = clase real, columnas = clase predicha):")
    w("")
    w("| real \\\\ pred | " + " | ".join(C.CLASS_NAMES) + " | total |")
    w("|---|" + "---:|" * (len(C.CLASS_NAMES) + 1))
    for i, name in enumerate(C.CLASS_NAMES):
        w("| **%s** | %s | %d |" % (name, " | ".join(str(v) for v in cm[i]), sum(cm[i])))
    w("")
    w("Verificaciones automaticas del Paso 1: %d de %d superadas."
      % (sum(1 for c in man["checks"] if c["passed"]), len(man["checks"])))
    failed = [c for c in man["checks"] if not c["passed"]]
    if failed:
        w("")
        w("**Verificaciones fallidas:** " + ", ".join("`%s` (%s)" % (c["check"], c["detail"]) for c in failed))
    w("")

    w("## 2. Intervalo bootstrap por sujetos")
    w("")
    w("- Unidad de remuestreo: sujeto de prueba completo (cluster bootstrap).")
    w("- Replicas: %s." % format(boot["method"]["replicates"], ",d").replace(",", "."))
    w("- Semilla: %d (`%s`)." % (boot["method"]["seed"], boot["method"]["rng"]))
    w("- Intervalo principal: percentil bilateral del 95 %.")
    w("- Modelo fijo en todas las replicas; clases fijas `%s`." % boot["method"]["classes"])
    w("- No se remuestrearon ventanas individuales ni ejemplos sinteticos; ver ClusterBootstrap:")
    w("  <%s>." % boot["method"]["reference"])
    w("")
    if ci is not None:
        w("**Balanced accuracy observada: %.4f.**" % ba)
        w("")
        w("**IC95 percentil: [%.4f, %.4f].**" % (ci[0], ci[1]))
        w("")
        w("Amplitud del intervalo: %.4f. Distancia al limite inferior: %.4f; al superior: %.4f"
          % (ci[1] - ci[0], ba - ci[0], ci[1] - ba))
        w("(el intervalo puede ser asimetrico respecto de la estimacion puntual).")
    else:
        w("**Intervalo retenido.** %s" % ival["status"])
    w("")
    if ival["ci95_recalls"]:
        w("Intervalos percentil del 95 % de los recalls por clase:")
        w("")
        w("| Clase | Recall observado | IC95 |")
        w("|---|---:|---:|")
        for name in C.CLASS_NAMES:
            k = name.lower()
            lo, hi = ival["ci95_recalls"][k]
            w("| %s | %.4f | [%.4f, %.4f] |" % (name, obs["recalls"][k], lo, hi))
        w("")
    w("Diagnosticos de las replicas (no sustituyen a la estimacion puntual):")
    w("")
    w("- Media bootstrap: %.4f (sesgo aparente %+.5f respecto de la BA observada)."
      % (diag["bootstrap_mean"], diag["bias_mean_minus_observed"]))
    w("- Desviacion estandar bootstrap: %.4f. Mediana: %.4f. Rango: [%.4f, %.4f]."
      % (diag["bootstrap_sd"], diag["bootstrap_median"], diag["bootstrap_min"], diag["bootstrap_max"]))
    w("- Replicas invalidas (alguna clase sin soporte): %d." % diag["invalid_replicates"])
    w("- Sujetos de prueba sin alguna clase real: %s."
      % (", ".join(diag["subjects_missing_a_true_class"]) or "ninguno"))
    w("")
    w("La estimacion puntual publicada es la BA observada, %.4f. Tener %s replicas reduce el error"
      % (ba, format(boot["method"]["replicates"], ",d").replace(",", ".")))
    w("numerico del bootstrap; la informacion independiente sigue proviniendo de %d sujetos."
      % obs["n_subjects"])
    w("")

    w("## 3. Configuraciones comparadas")
    w("")
    w("- Ejecuciones en `%s`: **%d**." % (os.path.basename(inv["sweep_dir"]), inv["n_runs"]))
    w("- Ejecuciones con evaluacion de prueba: **%d**." % inv["n_runs_with_test_eval"])
    w("- Particion de prueba identica en todas las ejecuciones: **%s** (md5 %s)."
      % ("si" if inv["identical_test_split_across_runs"] else "no",
         ", ".join(inv["split_file_md5"])))
    grid = inv["sweep_grid"]
    w("- Rejilla: %d modelos preentrenados x %d tasas de aprendizaje x %d activaciones x %d modos de cabeza."
      % (len(grid["pretrained_models"]), len(grid["learning_rate"]),
         len(grid["activation"]), len(grid["head_mode"])))
    w("  - Activaciones: %s." % ", ".join(grid["activation"]))
    w("  - Modos de cabeza: %s." % ", ".join(grid["head_mode"]))
    w("  - Tasas de aprendizaje: %s." % ", ".join(grid["learning_rate"]))
    w("")
    sel = inv["selected_run"]
    w("La ejecucion seleccionada (`%s`, configuracion %s) ocupa el **puesto %s de %d** por balanced"
      % (sel["run_folder"], sel["config_index"], int(sel["rank_by_test_balacc"]), inv["n_runs"]))
    w("accuracy de PRUEBA (%.4f) y el puesto %d por balanced accuracy de validacion (%.4f)."
      % (sel["balanced_accuracy_test"],
         int(inv_df["balanced_accuracy_val"].rank(ascending=False, method="min")
             [inv_df["selected_run"]].iloc[0]),
         sel["balanced_accuracy_val"]))
    w("Es decir, la configuracion se eligio maximizando la metrica del propio conjunto de prueba.")
    w("")
    w("Diez configuraciones con mayor balanced accuracy de prueba:")
    w("")
    top = inv_df.sort_values("balanced_accuracy_test", ascending=False).head(10)
    w("| # | Ejecucion | Preentrenado | lr | activacion | cabeza | BA val | BA test |")
    w("|---:|---|---|---|---|---|---:|---:|")
    for _, r in top.iterrows():
        pre = str(r.get("pretrained_model", "")).split("/")
        pre = pre[1] if len(pre) > 1 else "n/d"
        w("| %d | `%s`%s | %s | %s | %s | %s | %.4f | %.4f |" % (
            int(r["rank_by_test_balacc"]), r["run_id"],
            " **(seleccionada)**" if r["selected_run"] else "",
            pre, r.get("learning_rate"), r.get("activation"), r.get("head_mode"),
            r.get("balanced_accuracy_val"), r["balanced_accuracy_test"]))
    w("")
    w("El bootstrap principal mantiene fija la ejecucion historica. Reseleccionar el maximo entre")
    w("configuraciones en cada replica estimaria otra cantidad y no corregiria por si mismo el sesgo")
    w("de seleccion, y no se usa la configuracion con el intervalo mas favorable para redefinir el")
    w("resultado principal. Si hubo comparaciones adicionales fuera de esta carpeta, deben")
    w("incorporarse a la descripcion del proceso de seleccion.")
    w("")

    w("## 4. Interpretacion y limitaciones")
    w("")
    w("Los sujetos de prueba quedaron excluidos de todas las etapas de entrenamiento, incluido el")
    w("generador sintetico, segun la confirmacion del investigador. Sin embargo, la configuracion de")
    w("fine-tuning se eligio comparando la balanced accuracy de prueba entre las %d configuraciones"
      % inv["n_runs_with_test_eval"])
    w("evaluadas. El test no intervino en el ajuste de pesos, pero si en la seleccion.")
    w("")
    w("Por tanto el intervalo es **exploratorio**, para el modelo seleccionado en ese mismo test. No")
    w("debe interpretarse como un IC confirmatorio con cobertura del 95 % garantizada despues de la")
    w("seleccion; el bootstrap no elimina ese optimismo. El analisis cuantifica solo la variabilidad")
    w("asociada al muestreo de sujetos de prueba: no incorpora la variabilidad de la generacion")
    w("sintetica, del preentrenamiento ni del fine-tuning, y no corrige el sesgo de seleccion.")
    w("")
    w("Para una evaluacion confirmatoria haria falta un conjunto independiente de sujetos que no haya")
    w("intervenido en esa seleccion, con la configuracion ahora fija. Como alternativa, una validacion")
    w("cruzada anidada por sujetos evaluaria el procedimiento completo, pero constituye otro")
    w("experimento y no produce un intervalo para el 0.9055 historico.")
    w("")
    if ci is not None:
        w("## 5. Texto propuesto para comunicar el resultado")
        w("")
        w("> El modelo CaBiGRU preentrenado con datos sinteticos de beber y comer y ajustado con los")
        w("> datos originales de entrenamiento obtuvo una balanced accuracy de %.4f en %s ventanas"
          % (ba, format(obs["n_windows"], ",d").replace(",", ".")))
        w("> de %d sujetos de prueba. Se estimo un intervalo bootstrap percentil del 95 %% de"
          % obs["n_subjects"])
        w("> [%.4f, %.4f] mediante %s remuestreos de sujetos completos, manteniendo fijo el modelo."
          % (ci[0], ci[1], format(boot["method"]["replicates"], ",d").replace(",", ".")))
        w("> Los sujetos de prueba quedaron excluidos de todo entrenamiento. Dado que la configuracion")
        w("> se selecciono por su desempeno en este mismo test entre las %d configuraciones evaluadas,"
          % inv["n_runs_with_test_eval"])
        w("> el analisis es exploratorio y el intervalo no corrige el sesgo de seleccion ni incorpora")
        w("> la variabilidad del entrenamiento.")
        w("")

    w("## 6. Reproducibilidad")
    w("")
    w("| Elemento | Valor |")
    w("|---|---|")
    w("| Python | %s |" % env["python"])
    w("| TensorFlow | %s |" % env["tensorflow"])
    w("| Keras | %s |" % env["keras"])
    w("| NumPy | %s |" % env["numpy"])
    w("| pandas | %s |" % env["pandas"])
    w("| scikit-learn | %s |" % env["scikit_learn"])
    w("| Nodo | %s |" % env["node"])
    w("| GPU | %s |" % (", ".join(env["gpus"]) or "ninguna"))
    w("| sha256 checkpoint | `%s` |" % man["inputs"]["checkpoint"]["sha256"])
    w("| sha256 csv fuente | `%s` |" % man["inputs"]["source_csv"].get("sha256", "no calculado"))
    w("| sha256 archivo de particion | `%s` |" % man["inputs"]["split_file"]["sha256"])
    w("")
    w("Comando del Paso 1: `%s`" % man["command"])
    w("")
    w("Comando del Paso 2: `%s`" % boot["command"])
    w("")
    w("### Entregables")
    w("")
    for f in ["test_predictions.parquet", "test_predictions.csv.gz", "subject_confusion_matrices.csv",
              "bootstrap_replicates.csv", "bootstrap_summary.json", "configuration_inventory.csv",
              "configuration_inventory_summary.json", "reproduction_manifest.json", "report.md"]:
        path = os.path.join(d, f)
        if os.path.exists(path):
            w("- `%s` (%.1f MB)" % (f, os.path.getsize(path) / 1e6))
    w("")

    out = os.path.join(d, "report.md")
    with open(out, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("[out] %s" % out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
