# Oracular re-labeling: 50 tiles del held-out team

Esta carpeta contiene 50 tiles del *held-out* team (25 de barrio-norte + 25 de
correntoso-arauco), seleccionadas al azar en dos pasadas:

- **Prioridad 1 (20 tiles)**: las primeras 10 + 10 picks con semilla 42 (listadas
  en `priority_first_20.txt`). Son las que ya habías empezado a etiquetar. Si
  solo llegás a hacer 20, hacé estas.
- **Extensión (30 tiles)**: 15 + 15 adicionales con semilla 43. Agregan poder
  estadístico cuando se pueda terminar las primeras 20.

Ningún modelo las vio durante *training*.

El objetivo es producir un *ground truth* con máximo cuidado por una sola
persona, aplicando los criterios explícitos de más abajo, para que sirva como
referencia de evaluación "limpia" al comparar YOLO, Mask2Former, YOLOv8l y
DeepForest sobre el mismo *test set*.

## Cómo subir a Roboflow

1. Crear un proyecto nuevo en Roboflow: `defensible_space_oracular`.
2. Tipo: **Instance Segmentation**.
3. Clases: `building`, `trees_and_bushes` (exactamente estos nombres, en ese
   orden, para evitar renombres al exportar).
4. Subir las 20 imágenes de esta carpeta (archivos `.jpg`).
5. Etiquetar siguiendo los criterios de abajo. Si tiene sentido usar *Smart
   Polygon* (SAM) para iniciar, revisarlo manualmente antes de confirmar.
6. Generar un *version* nuevo con split `100% test` (no hace falta train/val).
7. Exportar como **COCO Segmentation**.
8. Descomprimir el export en `data/processed/dataset_coco_oracular/test/`
   (debe quedar la carpeta con las imágenes y el `_annotations.coco.json` ahí
   adentro).

## Criterios de etiquetado

El objetivo es *máxima consistencia*: un oráculo decide una vez, documenta su
decisión y la aplica igual a todas las imágenes.

### `building`

- **Qué contar**: cualquier estructura con techo claramente identificable como
  habitable: casa, cabaña, hostería, galpón grande con posible uso habitacional.
- **Qué NO contar**: pérgolas sin techo, cercos, veredas, parquización sin
  construcción, autos, piletas, muelles, cobertizos chicos sueltos (menos de
  ~4 m²).
- **Fragmentación**: una vivienda principal con galería techada contigua se
  etiqueta como *un solo polígono* que incluye las dos partes del techo. No se
  divide en dos.
- **Edificios parcialmente visibles**: si el techo está cortado por el borde
  del tile, igual se etiqueta la parte visible. (El pipeline de alertas marca
  `at_edge=True` y lo excluye del cálculo de distancia).
- **Sombras**: no incluir la sombra proyectada por el edificio.

### `trees_and_bushes`

- **Qué contar**: cualquier vegetación leñosa (árbol o arbusto) con copa visible
  desde arriba. Los árboles cercanos entre sí pueden agruparse en un único
  polígono si la copa aparece como *una masa continua* desde arriba (no hay
  separación clara entre copas individuales).
- **Qué NO contar**: pasto cortado, césped de jardín, pastizal bajo, cultivos
  rasantes.
- **Grupos vs individuos**: si se ve una copa individual con sombra propia bien
  separada de sus vecinos, va como polígono propio. Si varias copas se tocan
  y comparten follaje visible, van juntas en un solo polígono (hasta que haya
  separación clara).
- **Vegetación sobre edificio**: si una copa cuelga parcialmente sobre un techo,
  se etiqueta la copa completa como `trees_and_bushes` (polígonos pueden solapar
  con `building` en ese caso).

## Después del etiquetado

Pasame la carpeta exportada a `data/processed/dataset_coco_oracular/test/` y
corro la evaluación de los modelos sobre ese *ground truth*. Comparo contra las
métricas del *held-out* team actual para separar "capacidad del modelo" de
"sesgo del evaluador".
