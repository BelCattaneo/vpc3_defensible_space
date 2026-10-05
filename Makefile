# Pipeline del TP Defensible Space.
# Convencion: `make help` lista todos los targets con su descripcion.
# Parametros configurables al final del archivo.

.PHONY: help setup \
        tiles labelers dataset-v7 dataset-holdout dataset-oracular coco-to-yolo \
        train-yolo-nano train-yolo-large train-m2f-t train-m2f-s \
        eval-m2f compare \
        alerts aggregate mosaic map alerts-all \
        clean-pycache

help: ## Lista de targets disponibles
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z0-9_-]+:.*?## / {printf "  %-20s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

setup: ## Instala dependencias con uv
	uv sync

LOGS = logs

$(LOGS):
	@mkdir -p $(LOGS)

# ------------------------------------------------------------
# Preparacion de datos
# ------------------------------------------------------------
tiles: ## Genera tiles 1024x1024 desde data/raw/
	.venv/bin/python src/tile.py

labelers: ## Fetch mapping anotadora -> imagenes a reports/filenames_by_labeler.json
	.venv/bin/python src/filter_autolabel.py

dataset-v7: ## Dataset filtrado por autora (preserve-splits, 486 imagenes). Requiere AUTHOR_EMAIL
	.venv/bin/python src/build_dataset.py \
		--labelers $(AUTHOR_EMAIL) \
		--output data/processed/dataset_coco_v7 \
		--include-unmapped

dataset-holdout: ## Dataset held-out team (merge-to-test, 441 imagenes). Requiere TEAM_EMAILS
	.venv/bin/python src/build_dataset.py \
		--labelers $(TEAM_EMAILS) \
		--output data/processed/dataset_coco_holdout_team \
		--mode merge-to-test

dataset-oracular: ## Muestrea 50 imagenes para re-etiquetar en Roboflow
	.venv/bin/python src/build_oracular_sample.py

coco-to-yolo: ## Convierte export COCO a formato YOLO
	.venv/bin/python src/coco_to_yolo.py

# ------------------------------------------------------------
# Training
# ------------------------------------------------------------
train-yolo-nano: | $(LOGS) ## Entrena YOLOv8n-seg (baseline, 86 autora)
	.venv/bin/python -u src/train_yolo_seg.py 2>&1 | tee $(LOGS)/train-yolo-nano.log

train-yolo-large: | $(LOGS) ## Entrena YOLOv8l-seg sobre 927 mix (campeon v12)
	DATASET_YOLO=data/processed/dataset_yolo_v6 \
	MODEL_OUT=models/yolo_v12_l_927 \
	YOLO_MODEL=yolov8l-seg.pt \
	YOLO_IMG_SIZE=640 YOLO_BATCH=2 YOLO_EPOCHS=30 \
	YOLO_LR0=0.0005 YOLO_OPTIMIZER=AdamW \
	.venv/bin/python -u src/train_yolo_seg.py 2>&1 | tee $(LOGS)/train-yolo-large.log

train-m2f-t: | $(LOGS) ## Entrena Mask2Former + Swin-T (default)
	.venv/bin/python -u src/train_mask2former.py 2>&1 | tee $(LOGS)/train-m2f-t.log

train-m2f-s: | $(LOGS) ## Entrena Mask2Former + Swin-S sobre 927 mix (v13)
	DATASET_COCO=data/processed/dataset_coco_v6 \
	MODEL_OUT=models/m2f_v13_swinS_927 \
	M2F_MODEL_ID=facebook/mask2former-swin-small-coco-instance \
	.venv/bin/python -u src/train_mask2former.py 2>&1 | tee $(LOGS)/train-m2f-s.log

# ------------------------------------------------------------
# Evaluacion
# ------------------------------------------------------------
eval-m2f: ## Evalua M2F sobre test propio con pycocotools
	.venv/bin/python src/evaluate_m2f.py

eval-yolo: ## Evalua YOLO sobre test propio con pycocotools (mismo pipeline que M2F)
	.venv/bin/python src/evaluate_yolo.py

compare: ## Genera paneles side-by-side GT | YOLO | M2F
	.venv/bin/python src/compare_models.py

# ------------------------------------------------------------
# Alertas y mapa por sector
# Parametros: SECTOR, WEIGHTS, CONF (ver seccion de abajo).
# ------------------------------------------------------------
alerts: | $(LOGS) ## Corre compute_alerts para SECTOR (default barrio-norte)
	YOLO_WEIGHTS=$(WEIGHTS) \
	.venv/bin/python -u src/compute_alerts.py \
		--model yolo \
		--input data/interim/tiles/$(SECTOR) \
		--output $(ALERTS_DIR) \
		--pattern "*.tif" --conf-threshold $(CONF) 2>&1 | tee $(LOGS)/alerts-$(SECTOR_SLUG).log

aggregate: ## Agrega alertas por tile a un GeoJSON georreferenciado
	.venv/bin/python src/aggregate_alerts.py \
		--alerts-dir $(ALERTS_DIR) \
		--tiles-root data/interim/tiles \
		--output $(ALERTS_DIR).geojson

mosaic: ## Genera mosaico Web Mercator para fondo del mapa
	.venv/bin/python src/build_mosaic.py \
		--sector-dir data/interim/tiles/$(SECTOR) \
		--output reports/orthomosaic_$(SECTOR_SLUG).png

map: ## Renderiza mapa HTML interactivo con slider de confianza
	.venv/bin/python src/generate_map.py \
		--geojson $(ALERTS_DIR).geojson \
		--output reports/maps/alerts_map_$(SECTOR_SLUG)_final.html \
		--title "Alertas $(SECTOR)" \
		--orthomosaic reports/orthomosaic_$(SECTOR_SLUG).png

alerts-all: alerts aggregate map ## alerts + aggregate + map (end-to-end)

# ------------------------------------------------------------
# Demo end-to-end (autocontenido en el repo)
# Dos modos independientes, el usuario elige cual correr:
#   make demo           inferencia con los pesos del champion (YOLOv8l v12,
#                       entrenado sobre 927 mix). Produce el mapa final en
#                       ~5 min sin entrenar nada.
#   make demo-train     entrena YOLOv8n desde cero sobre las 86 imagenes
#                       etiquetadas (~15 min), luego infiere con esos pesos.
# ------------------------------------------------------------
DEMO_SECTOR            = demo
DEMO_CHAMPION_WEIGHTS  = models/demo_champion/weights/best.pt
DEMO_NANO_WEIGHTS      = models/demo_nano/weights/best.pt
DEMO_ALERTS            = reports/alerts_demo

demo-tiles: ## Genera tiles 1024x1024 de la ortofoto demo
	.venv/bin/python src/tile.py --sector $(DEMO_SECTOR)

demo-train: | $(LOGS) ## Entrena YOLOv8n desde cero sobre dataset v5 (86 autora)
	DATASET_YOLO=data/processed/dataset_yolo_v5 \
	MODEL_OUT=models/demo_nano \
	.venv/bin/python -u src/train_yolo_seg.py 2>&1 | tee $(LOGS)/demo-train.log

# Internal: inferencia + aggregate + mosaic + map. WEIGHTS lo pasa el caller.
define run_demo_infer
	YOLO_WEIGHTS=$(1) \
	.venv/bin/python -u src/compute_alerts.py \
		--model yolo \
		--input data/interim/tiles/$(DEMO_SECTOR) \
		--output $(DEMO_ALERTS) \
		--pattern "*.tif" --conf-threshold 0.15 2>&1 | tee $(LOGS)/demo-infer.log
	.venv/bin/python src/aggregate_alerts.py \
		--alerts-dir $(DEMO_ALERTS) \
		--tiles-root data/interim/tiles \
		--output $(DEMO_ALERTS).geojson
	.venv/bin/python src/build_mosaic.py \
		--sector-dir data/raw/$(DEMO_SECTOR) \
		--output reports/orthomosaic_$(DEMO_SECTOR).png
	.venv/bin/python src/generate_map.py \
		--geojson $(DEMO_ALERTS).geojson \
		--output reports/maps/alerts_map_$(DEMO_SECTOR).html \
		--title "Demo defensible space" \
		--orthomosaic reports/orthomosaic_$(DEMO_SECTOR).png
endef

demo-infer-champion: demo-tiles | $(LOGS) ## alerts + mapa con pesos champion v12 (YOLOv8l, 46 M)
	$(call run_demo_infer,$(DEMO_CHAMPION_WEIGHTS))

demo-infer-nano: demo-tiles | $(LOGS) ## alerts + mapa con pesos YOLOv8n recien entrenados
	$(call run_demo_infer,$(DEMO_NANO_WEIGHTS))

demo: demo-infer-champion ## Demo rapido: ortofoto + pesos champion -> mapa (~5 min)

demo-full: demo-train demo-infer-nano ## Demo con training desde cero + inferencia con esos pesos (~20 min)

# ------------------------------------------------------------
# Limpieza
# ------------------------------------------------------------
clean-pycache: ## Elimina directorios __pycache__
	find . -type d -name __pycache__ -prune -exec rm -rf {} +

# ------------------------------------------------------------
# Parametros configurables (override en linea: make map SECTOR=correntoso-arauco)
# ------------------------------------------------------------
SECTOR     ?= barrio-norte
# Nombres de salida coinciden con los entregables del repo (reports/alerts_<sector>_final.*).
# El sufijo _final marca el pipeline post-fixes (dedup IoU-only, danger_area
# por unary_union, extraccion M2F con return_binary_maps=True).
SECTOR_SLUG = $(SECTOR)
WEIGHTS    ?= models/demo_champion/weights/best.pt
CONF       ?= 0.15
ALERTS_DIR  = reports/alerts_$(SECTOR_SLUG)_final
