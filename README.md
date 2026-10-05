# TP Visión por Computadora III · Defensible Space

Trabajo Práctico Final de Visión por Computadora III (MIA · FIUBA).

Detección automática de incumplimiento del protocolo municipal de *defensible space* (Zona 1, 2 m) en Villa La Angostura (Neuquén), sobre ortofotos de dron.

Marco: proyecto Lawal aprobado por HOT Open Call 2026 ("Map your city with fAIr").

Entrega: 5 de octubre de 2026 (Clase 7).

## Resumen

Pregunta: ¿el sesgo inductivo débil de los Vision Transformers constituye una desventaja frente a un *baseline* CNN sobre dataset de dominio específico con pocas muestras, o el pre-entrenamiento masivo compensa?

Respuesta sobre *ground truth* oracular (50 imágenes re-etiquetadas por una sola persona con criterios documentados), ambos modelos evaluados con `pycocotools` sobre el mismo GT sanitizado y mismo *threshold* de confianza:

| Run | Arquitectura | Params | Dataset | mAP@0.5 |
|---|---|---:|---|---:|
| v8  | `Mask2Former + Swin-T` | 47 M  | 927 mix | **0.643** |
| v13 | `Mask2Former + Swin-S` | 69 M  | 927 mix | 0.634 |
| v10 | `Mask2Former + Swin-T` | 47 M  | 486 autora | 0.599 |
| v15 | `Mask2Former + Swin-S` | 69 M  | 486 autora | 0.590 |
| v14 | `Mask2Former + Swin-S` | 69 M  | 86 autora | 0.574 |
| v9  | `Mask2Former + Swin-T` | 47 M  | 86 autora | 0.532 |
| v12 | `YOLOv8l`              | 46 M  | 927 mix | 0.532 |
| v8  | `YOLOv8n`              | 3,3 M | 927 mix | 0.516 |

A parámetros equiparables (47 M vs 46 M), `Mask2Former + Swin-T` supera a `YOLOv8l` por 11,1 puntos sobre 927 mix. Los seis *runs* de M2F lideran los seis primeros puestos del ranking. Escalar la ViT de Swin-T a Swin-S no agrega ganancia marginal. El informe completo discute el recorrido experimental y la nota metodológica sobre `return_binary_maps=True` que es necesaria para medir mAP de instance segmentation correctamente con la implementación de `transformers` de HuggingFace.

## Setup

Requiere [uv](https://github.com/astral-sh/uv) como manejador de paquetes. Para instalarlo:

```bash
# macOS y Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Después del clone:

```bash
make setup
```

Instala el ambiente en `.venv/` con las dependencias declaradas en `pyproject.toml`.

### Device para training e inferencia

Por defecto el código autodetecta el *device* disponible en orden `cuda` (NVIDIA), `mps` (Apple Silicon), `cpu`. No hace falta configurar nada. Para forzar un *device* específico se puede exportar `DEVICE=cpu` (o `cuda`, `mps`) antes del comando de `make`.

Todos los pasos del pipeline están automatizados en el `Makefile`. `make help` lista los targets disponibles con una descripción corta de cada uno.

## Estructura

```
data/
  raw/              ortofotos originales por sector (demo incluida)
  interim/tiles/    tiles 1024x1024 georreferenciadas por sector
  processed/        datasets COCO y YOLO versionados
docs/               informe HTML, walkthrough y documentos auxiliares
models/             pesos entrenados (demo_nano y demo_champion versionados)
reports/
  maps/             mapas HTML interactivos finales por sector
  ...               métricas por run, holdouts, historial de experimentos
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

Equivale a `make demo-tiles` + `make demo-infer-champion`. Produce `reports/maps/alerts_map_demo.html` (14 MB). Abrir en el navegador y mover los *sliders* de confianza para filtrar edificios y vegetación en vivo.

### Modo B, training desde cero + inferencia (~20 min)

Para ver el *pipeline* completo con *training* real:

```bash
make demo-full
```

Equivale a `make demo-train` (entrena `YOLOv8n-seg` sobre las 86 imágenes v5) + `make demo-tiles` + `make demo-infer-nano` (infiere con los pesos recién entrenados). Sobrescribe `models/demo_nano/weights/best.pt`.

### Resultados finales sobre los sectores completos

La ortofoto del demo cubre una sola *task* del vuelo. Los mapas sobre los sectores completos corridos con el campeón (`Mask2Former + Swin-T` v8) vienen pre-generados en el repo para abrir directamente en el navegador:

- `reports/maps/alerts_map_barrio-norte_m2f.html`
- `reports/maps/alerts_map_correntoso-arauco_m2f.html`

Ambos autocontenidos: ortomosaico de fondo + detecciones en *overlay* + *widget* de confianza en vivo. No requieren servidor, se abren con doble click o `open reports/maps/alerts_map_<sector>_m2f.html` en macOS (`xdg-open` en Linux, `start` en Windows).

## Pipeline general

Los pasos listados abajo describen el *pipeline* completo tal como se corrió a lo largo del proyecto. Para solo validar el flujo end-to-end alcanza con el **demo** de la sección anterior. Los targets de preparación de datos y de *training* sobre los *splits* v6/v7 asumen el dataset completo (no versionado por tamaño); el *demo* trae los artefactos mínimos para que todo lo demás funcione.

`make help` imprime la lista completa de *targets* con una línea de descripción cada uno. Resumen por fase:

### 1. Preparación de datos

```bash
make tiles                 # ortofoto -> tiles 1024x1024 georreferenciados
make labelers              # mapea anotadora -> imágenes vía Roboflow API
make dataset-v7            # dataset filtrado por autora (486 imágenes)
make dataset-holdout       # dataset held-out team (441 imágenes)
make dataset-oracular      # muestrea 50 imágenes para re-etiquetar
make coco-to-yolo          # convierte export COCO a formato YOLO
```

Variables requeridas: `AUTHOR_EMAIL` para `dataset-v7`, `TEAM_EMAILS` para `dataset-holdout`. Ambos leen la clave `ROBOFLOW_API_KEY` desde `.env`.

### 2. Training

```bash
make train-yolo-nano       # YOLOv8n-seg baseline (86 autora)
make train-yolo-large      # YOLOv8l-seg sobre 927 mix (campeón v12)
make train-m2f-t           # Mask2Former + Swin-T (default)
make train-m2f-s           # Mask2Former + Swin-S sobre 927 mix (v13)
```

### 3. Evaluación

```bash
make eval-yolo             # mAP de YOLO sobre test propio vía pycocotools
make eval-m2f              # mAP de M2F sobre test propio vía pycocotools
make compare               # paneles side-by-side GT | YOLO | M2F
```

### 4. Alertas y mapa

Parametrizable por sector. `SECTOR=barrio-norte` por defecto:

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

Variables opcionales: `WEIGHTS=models/...` cambia el *checkpoint*, `CONF=0.3` el umbral de confianza de inferencia.

El HTML resultante trae un *widget* abajo a la derecha con dos *checkboxes* (edificios, vegetación) y dos *sliders* de confianza independientes para explorar *recall* y *precision* en vivo.

### Demo

Targets listados en la sección [Demo end-to-end](#demo-end-to-end). Son atajos sobre los targets de arriba fijando `SECTOR=demo` y pesos específicos:

```bash
make demo                  # inferencia con pesos champion (~5 min)
make demo-full             # training desde cero + inferencia (~20 min)
make demo-tiles            # solo tilea la ortofoto demo
make demo-train            # solo entrena YOLOv8n sobre v5
make demo-infer-champion   # alerts + mapa con pesos champion
make demo-infer-nano       # alerts + mapa con pesos nano recién entrenados
```

### Mantenimiento

```bash
make clean-pycache         # elimina directorios __pycache__
```

## Documentación

`docs/` contiene los entregables y documentos de referencia del proyecto:

- `INFORME.html` — informe final con pregunta de investigación, recorrido experimental, matriz *heatmap* de resultados sobre los tres regímenes de evaluación, análisis y conclusiones.
- `PROPUESTA_TP.html` — planteo original del TP.
- `DATA_PIPELINE.html` — pipeline operativo de datos (captura, procesamiento con ODM, anotación en Roboflow).
- `LABELING_GUIDELINES.html` — guías de anotación usadas por el equipo.

`reports/HISTORY.md` y `reports/EXPERIMENTS.md` guardan el historial detallado de *runs* con comandos reproducibles.

## Referencias

`references/` — bibliografía descargada (ver `references/README.md`).
