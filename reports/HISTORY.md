# Historico de runs de modelos

Cada entrada es un run completo con sus metricas reales medidas en test.
Las fuentes citadas (log + json + csv) son auditables.

## Formato de columnas

- **Dataset**: id del dataset usado (v5=86 tuyas, v6=86+841 team=927, etc).
- **Train/Val/Test**: n imagenes.
- **Augmentation**: pipeline aplicado.
- **Mejor epoch**: segun val_loss o fitness.
- **mAP@0.5 all**: metrica principal sobre test.
- **Fuentes**: archivos que respaldan los numeros.

---

## YOLOv8n-seg

| run | dataset | T/V/Te | epochs | aug | best ep | mAP@0.5 | mAP@0.5:0.95 | fuentes |
|---|---|---|---|---|---|---|---|---|
| v5 | 86 tuyas | 50/18/18 | 50 | Ultralytics defaults | — | **0.538** | 0.362 | logs/models_train_yolo_v5.log |
| v6 | 927 (86+841) | 641/147/139 | 50 | defaults + flipud 0.5 + degrees 10 + copy_paste 0.3 | 50 | 0.366 | 0.259 | logs/models_train_yolo_v6.log, models/_archive/yolo_v6/results.csv |
| v8 | 927 (86+841) | 641/147/139 | 50 | solo defaults | 50 | 0.355 | 0.244 | logs/models_train_yolo_v8.log, models/_archive/yolo_v8/results.csv |
| **v9** | 86 clean | 50/18/18 | 34 (early stop) | solo defaults + patience 5 | 29 | **0.496** | 0.326 | logs/models_train_yolo_v9.log, models/_archive/yolo_v9/results.csv |
| **v10** | 486 autora (v6 filtrado por labeler) | 332/77/77 | 20 (early stop) | solo defaults + patience 5 | 14 | **0.292** | 0.194 | logs/models_train_yolo_v10.log, models/_archive/yolo_v10/results.csv |
| **v11 (YOLOv8l)** | 86 autora | 50/18/18 | 50 | AdamW lr0=0.0005, no early stop | 50 | **0.456** | 0.292 | logs/models_train_yolo_v11_l_86.log, models/_archive/yolo_v11_l_86/results.csv |
| **v12 (YOLOv8l)** | 927 mix | 641/147/139 | 30 | AdamW lr0=0.0005, imgsz=640, batch=2 | 30 | **0.368** | 0.256 | logs/models_train_yolo_v12_l_927.log, models/_archive/yolo_v12_l_927/results.csv |

**per-class v8 en val** (de logs/models_train_yolo_v8.log):
- building: mAP@0.5=0.357, mAP@0.5:0.95=0.275
- trees_and_bushes: mAP@0.5=0.353, mAP@0.5:0.95=0.213

**per-class v9 en val** (de logs/models_train_yolo_v9.log):
- building: mAP@0.5=0.513, mAP@0.5:0.95=0.367
- trees_and_bushes: mAP@0.5=0.479, mAP@0.5:0.95=0.286

**per-class v10 en test** (de logs/models_train_yolo_v10.log + val post-hoc):
- building: mAP@0.5=0.287, mAP@0.5:0.95=0.206
- trees_and_bushes: mAP@0.5=0.297, mAP@0.5:0.95=0.183

**Hallazgos**:
- v6 vs v8: la aug extra "aerial" (flipud + degrees + copy_paste) NO era el problema. v8 sin aug rinde incluso un poco peor (0.355 vs 0.366).
- **v8 vs v9 (misma config, distinto dataset)**: 0.355 vs 0.496. Diferencia del 40% confirma data quality > data quantity cuando el training config es idéntico.
- v5 vs v9 (86 imgs ambos, pero v5 sin early stop): 0.538 vs 0.496. Diferencia por el early stop (v5 corrió 50 epochs completos, probablemente overfitteado al test chico).
- **v9 vs v10 (dataset chico vs mediano, ambos filtrados a autora)**: 0.496 vs 0.292. 486 imágenes "limpias" del mismo labeler NO superan a 86. La causa principal del drop en v6 NO es el labeler mix por sí solo (si lo fuera, v10 debería recuperar hacia v9). Hipótesis alternativa: el test set de v7 (77 imgs, también filtrado a autora) tiene distinta distribución que el de v5/v9. Dejar como observación honesta en el informe.

## Mask2Former + Swin-T

| run | dataset | T/V/Te | epochs run | aug | best ep | mAP@0.5 | mAP@0.5:0.95 | fuentes |
|---|---|---|---|---|---|---|---|---|
| v4 | 86 tuyas | 50/18/18 | 20 | manual hflip + ColorJitter | 20 (last) | **0.509** | 0.353 | logs/models_train_m2f_v4.log |
| v5 best | 86 tuyas | 50/18/18 | 20 | idem | 7 (best val) | 0.460 | 0.311 | reports/m2f_metrics_best.md (v5) |
| v6 (384x384) | 86 tuyas | 50/18/18 | 20 | manual hflip + ColorJitter + cap 384px | 20 (last) | 0.407 | 0.264 | logs/models_train_m2f_v6.log |
| v7 best | 927 (86+841) | 641/147/139 | 8 (early stop) | Albumentations: hflip+vflip+rotate15+rrc+colorjitter | 3 | 0.249 | 0.157 | reports/m2f_metrics.md (archived snapshot) |
| v7 last | 927 | 641/147/139 | 8 (early stop) | idem | 8 | 0.249 | 0.157 | reports/m2f_metrics_last.md |
| v8 best | 927 (86+841) | 641/147/139 | 9 (early stop) | hflip + ColorJitter (minimal) | 4 | 0.277 | 0.189 | logs/models_train_m2f_v8.log, models/_archive/m2f_v8/ |
| **v9 last** | 86 clean | 50/18/18 | 14 (early stop) | hflip + ColorJitter (Albumentations) | 9 | **0.230** | 0.154 | logs/models_train_m2f_v9.log, reports/m2f_metrics.md |
| **v10 best** | 486 autora (v7) | 332/77/77 | 20 (sin early stop) | hflip + ColorJitter | 17 | 0.230 | 0.155 | logs/models_train_m2f_v10.log, reports/m2f_v10_best_metrics.md |
| **v10 last** | 486 autora (v7) | 332/77/77 | 20 (sin early stop) | hflip + ColorJitter | 20 | **0.238** | 0.158 | logs/models_train_m2f_v10.log, reports/m2f_v10_last_metrics.md |
| **v13 last** (Swin-S) | 927 mix (v6) | 641/147/139 | 20 (sin early stop) | hflip + ColorJitter | 20 | **0.305** | 0.204 | models_train_m2f_v13_swinS_927.log, reports/m2f_v13_test_last.md |

**per-class v8 best en test**:
- building: mAP@0.5=0.337, mAP@0.5:0.95=0.250
- trees_and_bushes: mAP@0.5=0.216, mAP@0.5:0.95=0.129

**per-class v9 last en test**:
- building: mAP@0.5=0.294, mAP@0.5:0.95=0.212
- trees_and_bushes: mAP@0.5=0.166, mAP@0.5:0.95=0.097

**per-class v10 last en test**:
- building: mAP@0.5=0.258, mAP@0.5:0.95=0.184
- trees_and_bushes: mAP@0.5=0.217, mAP@0.5:0.95=0.133

**Hallazgos m2f**:
- v7 vs v8: aug minimal (v8) rinde mejor que Albumentations full (v7). Con labels ruidosos, aug agresiva empeora.
- v4 vs v9 (mismo dataset 86 clean, aug minimal): v4 (20 epochs full, PIL/torchvision) daba 0.509. v9 (early stop ep9/14, Albumentations) da 0.230. Diferencia enorme por early stop. En v4 el last_ckpt (ep20) superaba al best_val_ckpt (ep7). En dataset chico, val_loss subestima la capacidad del modelo. Lesson: no usar early stop por val_loss con M2F sobre dataset chico.
- **v8 (noisy) vs v9 (clean) con MISMO config**: 0.277 vs 0.230. M2F con early stop rinde mejor en noisy grande que clean chico. Patron opuesto a YOLO.

---

## Findings

- Más data **no mejoró** las métricas en v6/v7. Caída de 0.54 a 0.37 (YOLO) y 0.51 a 0.25 (M2F).
- Hipótesis "AUTOLABEL contamina v6" descartada: `reports/filenames_by_labeler.json` muestra que las 359 imágenes etiquetadas por AUTOLABEL en Roboflow (jobs status=review o assigned) nunca entraron al export v6. Las 927 son todas de etiquetado humano.
- Hipótesis "filtrar por labeler autora recupera v5": descartada en v10 (0.292 vs 0.496 de v9). No hay recuperación con 486 imgs limpias del mismo labeler. La diferencia dataset chico vs mediano no se cubre con labeler más consistente.
- Hipótesis alternativa pendiente: el test set de v7 (77 imgs filtradas a autora) tiene distribución distinta a v5/v9 (18 imgs). La reducción del test puede reflejar varianza en vez de calidad del modelo.
- M2F cayó más que YOLO: su matcher Hungarian es más sensible a label noise.

## Composición de v6 por labeler

- autora (jobs): 400 imgs
- v5 originales sin job (autora, pre-jobs): 86 imgs
- equipo-A: 197 imgs
- equipo-B: 165 imgs
- equipo-C: 79 imgs
- AUTOLABEL (en Roboflow, no exportados): 359 imgs — no entran a v6.

## Próximas iteraciones

- Evaluar M2F v10 (en entrenamiento) para completar la comparación.
- Si M2F v10 tampoco recupera, el informe debe presentar el resultado completo sin forzar la hipótesis de label noise.

---

## Evaluación cruzada sobre held-out team (441 imgs, 2026-10-03)

Para romper la ambigüedad "más data empeora el modelo" vs "cada test se parece a su train", se construye un set held-out con las 441 imágenes etiquetadas por team Lawal (`src/build_holdout_team.py`). Ningún modelo las vio en training. Las labels son las de team, con el mismo ruido inter-anotadoras.

### YOLO sobre held-out team

| run | train imgs | mAP50 en SU test | mAP50 held-out team |
|---|---|---|---|
| v9 (86 autora, nano, early stop) | 50 | 0.496 | 0.278 |
| v10 (486 autora, nano, early stop) | 332 | 0.292 | 0.334 |
| v6 (927 mix, nano, aug extra) | 641 | 0.366 | 0.467 |
| v8 (927 mix, nano, defaults) | 641 | 0.355 | **0.543** |
| v11 (86 autora, YOLOv8l) | 50 | 0.456 | 0.321 |
| v12 (927 mix, YOLOv8l) | 641 | 0.368 | 0.440 |

Fuente: `reports/holdout_team_yolo.json`, `reports/yolo_v11_l_86_eval.json`, `reports/yolo_v12_l_927_eval.json`.

### M2F sobre held-out team

| run | train imgs | mAP50 en SU test | mAP50 held-out team |
|---|---|---|---|
| v9 last (86 autora) | 50 | 0.230 | 0.211 |
| v10 best (486 autora) | 332 | 0.230 | 0.241 |
| v10 last (486 autora) | 332 | 0.238 | 0.234 |
| v8 best (927 mix) | 641 | 0.277 | 0.275 |
| v8 last (927 mix) | 641 | 0.277 | **0.331** |

Fuente: `reports/holdout_team_m2f.json`.

### Lectura

Las métricas en el test propio de cada run no son comparables entre runs porque el ground truth cambia con el labeler que lo produjo. Sobre un held-out común:

- v8 YOLO (mix, 927) sube de 0.355 a 0.543 cuando se lo evalúa sobre labels de team. Es la ontología que aprendió.
- v9 YOLO (solo autora) baja de 0.496 a 0.278 cuando se lo evalúa sobre labels de team. Aprendió una ontología distinta.
- M2F muestra el mismo patrón más atenuado (v8 last: 0.277 → 0.331. v9 last: 0.230 → 0.211).

### Evaluación sobre ground truth oracular

Después de la evaluación cruzada se produjo un ground truth oracular de 50 imágenes (`src/build_oracular_sample.py` + re-etiquetado por autora en Roboflow con criterios documentados en `data/oracular_relabel/README.md`) para comparar runs sobre un referente limpio.

| run | train imgs | oracular mAP50 | building | trees |
|---|---|---|---|---|
| **v12 YOLOv8l (927 mix)** | 641 | **0.646** | **0.739** | 0.553 |
| v8 YOLO nano (927 mix) | 641 | 0.622 | 0.704 | 0.539 |
| v10 YOLO nano (486 autora) | 332 | 0.506 | 0.584 | 0.429 |
| v11 YOLOv8l (86 autora) | 50 | 0.506 | 0.611 | 0.400 |
| M2F v8 last (927 mix) | 641 | 0.494 | 0.670 | 0.319 |
| M2F v10 last (486 autora) | 332 | 0.444 | 0.602 | 0.286 |
| v9 YOLO nano (86 autora) | 50 | 0.439 | 0.506 | 0.371 |
| M2F v9 last (86 autora) | 50 | 0.379 | 0.559 | 0.200 |

Fuente: `reports/oracular_summary.json`, `reports/yolo_v12_l_927_eval.json`.

Conclusión definitiva: entrenar sobre 927 imágenes mixtas sí mejora capacidad real. Escalar parámetros encima (v12 YOLOv8l) suma 2.4 puntos adicionales. YOLO supera a M2F por 15.2 puntos a parámetros comparables sobre 927 mix. El pre-entrenamiento masivo del ViT no compensa la desventaja relativa en el régimen explorado.

Para el MVP de alertas que corre sobre ortofotos nuevas (sin ground truth), v12 YOLOv8l (927 mix) es el mejor candidato.
