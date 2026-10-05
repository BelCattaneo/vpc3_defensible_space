# TP Visión por Computadora III · Defensible Space

Trabajo Práctico Final de Visión por Computadora III (CEIA · FIUBA).

Detección automática de incumplimiento del protocolo municipal de *defensible space* (Zona 1, 2 m) en Villa La Angostura (Neuquén), sobre ortofotos de dron.

Marco: proyecto Lawal aprobado por HOT Open Call 2026 ("Map your city with fAIr").

Entrega: 5 de octubre de 2026 (Clase 7).

## Resumen

Pregunta: ¿el sesgo inductivo débil de los Vision Transformers constituye una desventaja frente a un *baseline* CNN sobre dataset de dominio específico con pocas muestras, o el pre-entrenamiento masivo compensa?

Respuesta sobre *ground truth* oracular (50 imágenes re-etiquetadas por una sola persona con criterios documentados):

| Run | Arquitectura | Params | Dataset | mAP@0.5 |
|---|---|---:|---|---:|
| v12 | `YOLOv8l` | 46 M | 927 mix | **0.646** |
| v8 | `YOLOv8n` | 3,3 M | 927 mix | 0.622 |
| v13 | `Mask2Former + Swin-S` | 69 M | 927 mix | 0.508 |
| v8 | `Mask2Former + Swin-T` | 47 M | 927 mix | 0.494 |

A parámetros equiparables (46 M vs 47 M) la CNN supera a la ViT por 15,2 puntos. Escalar el ViT a Swin-S aporta 1,4 puntos y no cierra la brecha. El informe completo discute el recorrido experimental y la ambigüedad entre efecto de arquitectura, de capacidad y de evaluador.

## Setup

Requiere [uv](https://github.com/astral-sh/uv) como manejador de paquetes.

```bash
make setup
```

Instala el ambiente en `.venv/` con las dependencias declaradas en `pyproject.toml`.

Todos los pasos del pipeline están automatizados en el `Makefile`. `make help` lista los targets disponibles con una descripción corta de cada uno.

## Estructura

```
data/
  raw/              ortofotos originales por sector (demo incluida)
  interim/tiles/    tiles 1024x1024 georreferenciadas por sector
  processed/        datasets COCO y YOLO versionados
docs/               informe HTML, walkthrough y documentos auxiliares
models/             pesos entrenados (demo_nano y demo_champion versionados)
reports/            alertas, mapas HTML, métricas, historial de experimentos
references/         bibliografía
src/                código de todas las etapas
logs/               stdout de training e inferencia (gitignored)
```

## Demo end-to-end

El repo incluye un *demo* autocontenido para que los evaluadores corran el *pipeline* completo sobre datos reales sin descargar nada externo. Pensado para dos escenarios distintos.

Lo que viene en el repo:

- `data/raw/demo/orthophoto_*.tif` (89 MB) — una *task* real de Villa La Angostura.
- `data/processed/dataset_coco_v5/` y `dataset_yolo_v5/` (17 MB cada uno) — 86 imágenes etiquetadas por la autora.
- `models/demo_nano/weights/best.pt` (6,5 MB) — pesos `YOLOv8n-seg` pre-entrenados sobre v5 (equivale al *run* v5).
- `models/demo_champion/weights/best.pt` (88 MB) — pesos del campeón `YOLOv8l-seg` v12 (46 M parámetros, entrenado sobre las 927 imágenes mixtas).

### Modo A, solo inferencia con el campeón (~5 min)

Para ver el *pipeline* operativo con la mejor calidad de detección sin entrenar nada:

```bash
make demo
```

Equivale a `make demo-tiles` + `make demo-infer-champion`. Produce `reports/alerts_map_demo.html` (14 MB). Abrir en el navegador y mover los *sliders* de confianza para filtrar edificios y vegetación en vivo.

### Modo B, training desde cero + inferencia (~20 min)

Para ver el *pipeline* completo con *training* real:

```bash
make demo-full
```

Equivale a `make demo-train` (entrena `YOLOv8n-seg` sobre las 86 imágenes v5) + `make demo-tiles` + `make demo-infer-nano` (infiere con los pesos recién entrenados). Sobrescribe `models/demo_nano/weights/best.pt`.

### Resultados finales sobre los sectores completos

La ortofoto del demo cubre una sola *task* del vuelo. Los dos mapas HTML sobre los sectores completos vienen pre-generados en el repo para abrir directamente en el navegador:

- `reports/alerts_map_barrio-norte_final.html` — 714 *tiles*, 1.100 edificios detectados
- `reports/alerts_map_correntoso-arauco_final.html` — 2.056 *tiles*, 797 edificios detectados

Ambos son autocontenidos: ortomosaico de fondo + detecciones en *overlay* + *widget* de confianza en vivo. No requieren servidor, se abren con `open reports/alerts_map_<sector>_final.html` en macOS o doble click.

## Pipeline general

Los pasos listados abajo describen el *pipeline* completo tal como se corrió a lo largo del proyecto. Para solo validar el flujo end-to-end, el **demo** de la sección anterior alcanza. Los targets de preparación de datos y de *training* sobre los *splits* v6/v7 asumen el dataset completo (no versionado por tamaño); el *demo* trae los artefactos mínimos para que todo lo demás funcione.

`make help` lista todos los *targets* con descripción corta.

### 1. Preparación de datos

```bash
make tiles                 # orthophoto -> tiles 1024x1024
make labelers              # mapea anotadora -> imágenes (Roboflow API)
make dataset-v7            # 486 imágenes de la autora
make dataset-holdout       # 441 imágenes solo del equipo (held-out)
make dataset-oracular      # muestrea 50 imágenes para re-etiquetar
make coco-to-yolo          # convierte export COCO a formato YOLO
```

### 2. Training

```bash
make train-yolo-nano       # YOLOv8n baseline (86 autora)
make train-yolo-large      # YOLOv8l 927 mix, imgsz 640 (campeón v12)
make train-m2f-t           # Mask2Former + Swin-T (default)
make train-m2f-s           # Mask2Former + Swin-S 927 mix (v13)
```

### 3. Evaluación

```bash
make eval-m2f              # mAP vía pycocotools
make compare               # paneles side-by-side GT | YOLO | M2F
```

### 4. Alertas y mapa

Parametrizable por sector (`SECTOR=barrio-norte` por defecto):

```bash
make alerts-all                              # pipeline end-to-end sobre barrio-norte
make alerts-all SECTOR=correntoso-arauco     # mismo sobre otro sector
```

o paso a paso:

```bash
make alerts     SECTOR=correntoso-arauco     # compute_alerts por tile
make aggregate  SECTOR=correntoso-arauco     # agregación georreferenciada
make mosaic     SECTOR=correntoso-arauco     # mosaico Web Mercator
make map        SECTOR=correntoso-arauco     # mapa HTML con slider
```

El HTML resultante trae un *widget* abajo a la derecha con dos *checkboxes* (edificios, vegetación) y dos *sliders* de confianza independientes para explorar *recall* y *precision* en vivo. Overrides adicionales: `WEIGHTS=models/...` cambia el *checkpoint*, `CONF=0.3` el umbral de confianza de inferencia.

## Documentación

`docs/` contiene los entregables y documentos de referencia del proyecto:

- `INFORME.html` — informe final con pregunta de investigación, recorrido experimental, matriz *heatmap* de resultados sobre los tres regímenes de evaluación, análisis y conclusiones.
- `CODE_WALKTHROUGH.html` — recorrido por cada módulo de `src/` con justificación de decisiones y preguntas de defensa.
- `PROPUESTA_TP.html` — planteo original del TP.
- `DATA_PIPELINE.html` — pipeline operativo de datos (captura, procesamiento con ODM, anotación en Roboflow).
- `LABELING_GUIDELINES.html` — guías de anotación usadas por el equipo.

`reports/HISTORY.md` y `reports/EXPERIMENTS.md` guardan el historial detallado de *runs* con comandos reproducibles.

## Referencias

`references/` — bibliografía descargada (ver `references/README.md`).
