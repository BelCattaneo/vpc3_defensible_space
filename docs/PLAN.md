# Plan del Trabajo Práctico Final — Vision Transformers (VpC III)

**Materia:** Vision por Computadora III · CEIA · FIUBA
**Docentes:** Esp. Abraham Rodriguez · Mg. Oksana Bokhonok
**Fecha de entrega:** 5 de octubre de 2026 (Clase 7)
**Presentación:** Clase 8 (posterior a la entrega)
**Modalidad:** Individual (a confirmar con docentes)

---

## 1. Contexto y objetivo

### 1.1. Marco del proyecto

El presente trabajo práctico se enmarca en una propuesta más amplia presentada por la cooperativa Lawal a la convocatoria HOT Open Call 2026 ("Map your city with fAIr"), titulada *Measuring the Defensible Space*. La propuesta fue aceptada por Humanitarian OpenStreetMap Team y financia el relevamiento con drones, los mapatones comunitarios y la publicación de datos abiertos sobre el riesgo de incendios de interfase urbano-forestal en Villa La Angostura, Neuquén.

El trabajo práctico **no reemplaza** al proyecto HOT, sino que aporta una capa técnica complementaria: mientras la propuesta HOT define un pipeline operacional basado en fAIr (que por defecto emplea arquitecturas CNN como YOLOv8), el presente trabajo explora **arquitecturas Vision Transformer** sobre el mismo problema y dataset, produciendo un análisis comparativo alineado con el programa de la materia.

### 1.2. Objetivo académico

Comparar empíricamente arquitecturas de la familia Vision Transformer (CLIP, SAM, DETR/ViT) contra un baseline CNN (YOLOv8) sobre la tarea de detección de riesgo de incendio de interfase urbano-forestal, utilizando imágenes de dron a alta resolución (6 cm/px) del sector El Mallín, Villa La Angostura.

### 1.3. Pregunta central de investigación

*¿El sesgo inductivo débil de los Vision Transformers, señalado explícitamente en la Clase 1 (slide 17), constituye una desventaja significativa en imágenes aéreas de dominio específico con datasets pequeños (~1000 samples), o el pre-entrenamiento masivo en modelos foundation compensa esa desventaja?*

---

## 2. Encuadre académico

La rúbrica del TP, tal como se define en la Clase 1 (slides 4-5) y en el `README.md` del repositorio CEIA-ViT, exige:

| Requerimiento | Cómo se cumple en este plan |
|---|---|
| Proyecto estructurado en git | Repositorio propio con estructura tipo Cookiecutter Data Science |
| Código funcional y modular (nivel preproducción) | `src/` modular, configs YAML externas, logging con niveles, tests con pytest, MLflow |
| Informe técnico PDF | Documento final con objetivo, arquitectura, implementación, evaluación, resultados, conclusiones y planificación |
| README orientativo | Redactado desde Fase 1 y actualizado en cada fase |
| Presentación de 15 min en Clase 8 | Preparada en Fase 7, foco en métricas + visualizaciones + aplicación real |
| Aplicación en contexto real | La aplicación al problema de Villa La Angostura está garantizada por el marco HOT |

---

## 3. Sobre el dataset

### 3.1. Recursos ya disponibles

**Imagen y anotaciones específicas de Villa La Angostura:**

- **Imagen de dron de El Mallín**: publicada en OpenAerialMap durante la semana MA.PA de mayo 2026. Resolución nativa: 6 cm/px. Superficie: aproximadamente 1.08 km². Licencia CC-BY 4.0.
- **Buildings en OpenStreetMap**: 1.561 edificios registrados en Villa La Angostura (referencia: página 10 de la propuesta HOT). Sirven como bootstrap para la clase "estructuras", aunque requieren verificación contra la imagen de dron por posible desactualización.
- **Datasets fAIr con etiquetas de edificios**: tres datasets ya publicados sobre Villa La Angostura, con modelo YOLOv8 entrenado y accuracy oficial:
  - Dataset 382 · *Villa La Angustura - Barrio* — YOLO_V8_V1 · accuracy 22.5%
  - Dataset 383 · *Villa La Angostura • Barrio Calafate* — YOLO_V8_V2 · accuracy 63.6%
  - Dataset 393 · *A neighborhood in Villa La Angostura* — YOLO_V8_V2 · accuracy 57.7%

**Datasets externos de benchmark (descargados en `data/external/`):**

- **UAVid** (mirror `arakesh/uavid-15-hq-mixedres` de HuggingFace): 4 GB · 200 imágenes de entrenamiento + 70 de test a 4K px · 8 clases con `building` y `tree` como semánticas explícitas · licencia research non-commercial. Benchmark estándar más citado en literatura de segmentación aérea en imágenes de dron.
- **Aeroscapes** (mirror `dronefreak/Aeroscapes` de HuggingFace): ~400 MB · 1000 train + 648 val · 12 clases incluyendo `construction`, `vegetation`, `road` · licencia CC-BY-SA 4.0. Complementario a UAVid.

### 3.2. Recursos ausentes que deben generarse

- **Anotaciones de árboles individuales**: solamente 33 árboles registrados en OSM para toda la localidad. Insuficiente.
- **Anotaciones de tanques de gas LPG**: no existen en OSM.
- **Anotaciones de pilas de leña**: no existen en OSM.
- **Anotaciones de reservorios de agua**: 0 registrados en OSM.
- **Ground truth de canopy continuo**: no existe.

### 3.3. Estrategia de anotación

Se propone anotar manualmente un conjunto reducido pero suficiente (200–300 samples por clase relevante) sobre un subconjunto de tiles de El Mallín. La estrategia asume dos escenarios posibles:

- **Escenario A (deseable)**: se verifica que los mapatones HOT ya generaron anotaciones sobre El Mallín (durante la semana MA.PA de mayo 2026 o inmediatamente después). En ese caso, se reutilizan y se ajustan.
- **Escenario B (probable)**: no existen anotaciones previas específicas del sector. En ese caso, se realiza anotación manual con herramientas como Label Studio o Roboflow, priorizando las clases con mayor impacto en el compliance index (estructuras y árboles individuales).

### 3.4. Datasets externos

**Pesos pre-entrenados** para mitigar la limitación de tamaño del dataset propio:

- **COCO** (Common Objects in Context) para pesos base de DETR y YOLOv8.
- **SA-1B** para SAM (pre-entrenamiento nativo).
- **LAION-2B** para CLIP (pre-entrenamiento nativo).

**Datasets externos ya descargados para benchmark comparativo** (ver sección 3.1 para detalles):

- **UAVid** — para reporte de mIoU comparable con literatura de segmentación aérea reciente.
- **Aeroscapes** — para segundo punto de comparación con esquema de clases distinto.

Ninguno de estos datasets externos cubre biomas patagónicos ni clases específicas como tanques de gas o pilas de leña; el fine-tuning sobre samples propios de El Mallín sigue siendo necesario. El uso de los datasets externos se orienta a: (a) desarrollar el pipeline mientras se espera imagen adicional de Villa La Angostura, (b) reportar comparabilidad con el estado del arte, (c) medir el gap de dominio entre benchmarks estándar y el escenario patagónico real.

---

## 4. Riesgos identificados y mitigaciones

| Riesgo | Impacto | Mitigación |
|---|---|---|
| No existen anotaciones previas de El Mallín | Alto — retrasa Fase 2 dos semanas | Priorizar anotación de solo 2 clases (estructuras + árboles) si el tiempo aprieta |
| DETR fine-tuning no cabe en MacBook M4 | Medio — obliga a usar Colab | Diseñar el entrenamiento para caber en sesiones de 12 h de Colab gratuito. Plan B: comparar DETR pre-entrenado (sin fine-tune) vs YOLO pre-entrenado |
| Bugs de PyTorch backend MPS en modelos foundation | Medio — sorpresas de runtime | Test end-to-end con 10 samples antes de escalar cada modelo |
| SAM ViT-H inference lenta en M4 (~2-4 seg/tile) | Bajo — no bloquea, solo estira Fase 3 | Correr SAM ViT-H solo sobre subset de test, no sobre todo el dataset. MobileSAM sobre el dataset completo. |
| SAM ViT-H fine-tuning no cabe en M4 con 24 GB | Medio — se acepta como caveat | Solo MobileSAM se fine-tunea; SAM ViT-H se reporta zero-shot únicamente. Se documenta explícitamente en el informe. |
| Los mapatones HOT no llegan a tiempo | Bajo — no bloquea el TP | El TP no depende del cronograma HOT, se resuelve con anotación propia |

---

## 5. Cronograma

Duración total: 6 semanas calendario (25 de agosto → 5 de octubre 2026), más 1 semana adicional para la presentación de la Clase 8.

| Semana | Rango | Fase(s) principal(es) |
|---|---|---|
| 1 | 25 – 31 ago | Fase 1 (setup) |
| 2 | 1 – 7 sep | Fase 2 (data pipeline) |
| 3 | 8 – 14 sep | Fase 3 (zero-shot CLIP + SAM) |
| 4 | 15 – 21 sep | Fase 4 (baseline CNN + DETR) |
| 5 | 22 – 28 sep | Fase 5 (compliance index) + Fase 6 (inicio informe) |
| 6 | 29 sep – 5 oct | Fase 6 (informe final) y submission |
| 7 | 6 – 12 oct | Fase 7 (presentación Clase 8) |

---

## 6. Fases detalladas

### Fase 1 — Setup del repositorio y del ambiente

**Objetivo.** Crear el esqueleto del proyecto con estructura de nivel preproducción, listo para recibir código a partir de la Fase 2.

**Por qué es necesaria.** La rúbrica de la Clase 1 (slides 6-11) es explícita sobre la diferencia entre "código EDA" (notebook desordenado) y "código preproducción" (modular, configurable, logueado, testeado). No es opcional. Además, arrancar con la estructura correcta desde el día 1 evita refactors costosos hacia el final.

**Ideas principales.**
- Estructura tipo Cookiecutter Data Science (que la Clase 1 muestra explícitamente): separación clara entre `data/`, `src/`, `models/`, `notebooks/`, `reports/`.
- Configuraciones externas en YAML, nunca hardcodeadas.
- Logging con niveles (INFO, WARNING, ERROR, DEBUG) desde el arranque.
- MLflow para trackear experimentos: hiperparámetros, métricas, artefactos.
- Control de dependencias con `requirements.txt` pineado o `poetry`/`uv`.
- Pre-commit hooks básicos: black, ruff, isort.

**Actividades.**
1. Crear la estructura de carpetas.
2. Inicializar git y hacer el primer commit.
3. Configurar `pyproject.toml` o `requirements.txt` con PyTorch (con soporte MPS), transformers, ultralytics, segment-anything, open-clip-torch, mlflow, hydra o pydantic, pytest.
4. Escribir un `src/config.py` que cargue configuraciones desde `configs/*.yaml`.
5. Escribir un `src/utils/logging.py` con setup del logger.
6. Escribir un smoke test que verifique que PyTorch reconoce MPS.
7. Redactar el README inicial (objetivo, estructura, cómo correr).

**Deliverables.**
- Repositorio git funcional con estructura preproducción.
- README inicial.
- Smoke test que corra en verde.

**Criterio de done.** `pytest` corre sin errores, el logger imprime a stdout y a archivo, y `python -c "import torch; print(torch.backends.mps.is_available())"` devuelve `True`.

**Riesgos y mitigación.**
- Compatibilidad de PyTorch con MPS puede requerir versión específica; se pinea la versión en `requirements.txt` desde el arranque.

---

### Fase 2 — Data pipeline

**Objetivo.** Preparar los datos y las anotaciones necesarias para todos los experimentos posteriores. Al final de esta fase, todos los modelos deben poder consumir los mismos splits sin más preparación.

**Por qué es necesaria.** Sin datos, no hay experimentos. Además, si cada modelo (CLIP, SAM, DETR, YOLO) consume datos preparados de forma distinta, la comparación no es justa. Esta fase estandariza el input.

**Ideas principales.**
- Tileado consistente: se propone 512×512 píxeles con overlap del 20% para evitar cortar objetos en bordes.
- A 6 cm/px, un tile de 512 px corresponde a aproximadamente 30 metros de terreno — apropiado para las distancias del protocolo municipal (Zonas 1, 2 y 3).
- Splits estratificados por régimen (urban matrix vs forest fringe) para que las métricas por régimen sean comparables.
- Formato de anotaciones: COCO JSON (compatible con DETR, YOLOv8 con conversión, MMDetection, Detectron2).
- Ground truth de canopy: máscara raster binaria (para evaluar IoU de SAM).

**Actividades.**
1. Descargar los tiles de El Mallín desde OpenAerialMap (identificar el ID del pilot de mayo 2026).
2. Descargar labels de los datasets fAIr 382 / 383 / 393 vía API (ya sondeada, endpoints funcionales).
3. Consolidar UAVid y Aeroscapes (ya descargados en `data/external/`) en formato consumible por los mismos dataloaders del proyecto.
4. Verificar existencia de anotaciones previas del mapatón MA.PA (mail a HOT / Lawal si es necesario).
5. Reproyectar / retilear si es necesario.
6. Definir splits train / val / test estratificados (por ejemplo 70 / 15 / 15) para El Mallín; para UAVid y Aeroscapes usar los splits estándar publicados.
7. Anotar manualmente el ground truth para las clases priorizadas que no cubren los datasets fAIr: árboles individuales, tanques de gas, leña, reservorios de agua. Herramienta sugerida: Label Studio.
8. Anotar máscaras binarias de canopy continuo en un subset representativo (para evaluar SAM).
9. Escribir `src/data/dataset.py` con un `DefensibleSpaceDataset` que exponga los splits en formato PyTorch, y un `BenchmarkDataset` para UAVid/Aeroscapes.

**Deliverables.**
- `data/raw/` con tiles descargados.
- `data/interim/` con tiles re-tilados y splits.
- `data/processed/annotations_coco.json` con anotaciones.
- `data/processed/canopy_masks/` con máscaras de canopy.
- Módulo `src/data/dataset.py` con dataloaders.

**Criterio de done.** El dataloader devuelve batches con imagen + anotaciones sin errores; se puede visualizar un sample con anotaciones superpuestas.

**Riesgos y mitigación.**
- Anotar 200-300 samples por clase es la tarea más subestimada. Se prioriza anotar completamente 2 clases (estructuras + árboles) antes de intentar las 5.
- Si el mapatón MA.PA generó anotaciones reutilizables, la Fase 2 se acorta drásticamente. Confirmar temprano.

---

### Fase 3 — Experimentos zero-shot: CLIP + MobileSAM

**Objetivo.** Obtener resultados iniciales sobre El Mallín sin necesidad de entrenar nada, cubriendo la clasificación de tiles (CLIP) y la segmentación de canopy (SAM).

**Por qué es necesaria.** Esta fase es el *safety net* del cronograma: aunque las fases 4 y 5 fallen por completo, el TP tiene resultados sólidos sobre dos arquitecturas ViT-family aplicadas al problema. Además, corresponde directamente a los contenidos de las Clases 5 (CLIP) y 6 (SAM).

**Ideas principales.**

*CLIP para clasificación de tiles:*
- Se emplea OpenCLIP ViT-B/32 (versión open-source con pesos LAION-2B).
- La tarea binaria: "¿este tile contiene canopy continuo?" — corresponde a la regla de Zona 3 del protocolo municipal.
- Se prueban prompts en inglés (`"an aerial view of dense continuous forest canopy"` vs `"an aerial view of houses and streets"`) y en español.
- Métrica principal: accuracy y F1 contra ground truth binario por tile.
- Comparación: zero-shot vs una versión con head lineal fine-tuneada sobre embeddings congelados (si el tiempo lo permite).

*SAM para segmentación de canopy — comparación MobileSAM vs SAM ViT-H:*
- Se corren **ambas variantes** sobre El Mallín en modo zero-shot para reportar el trade-off calidad/velocidad de forma empírica.
  - **SAM ViT-H** (~636M parámetros, ~2.4 GB de pesos): calidad de referencia, inference lenta en M4 (~2-4 seg por tile), no se fine-tunea.
  - **MobileSAM** (~9.66M parámetros, ~40 MB): mismo prompt encoder + mask decoder que SAM, image encoder destilado a ViT-Tiny. Inference sub-segundo en M4, fine-tunable localmente.
- Estrategia de prompting: seed points automáticos en grid (16×16 por tile).
- Métricas por modelo: mean IoU, boundary F1, tiempo de inference por tile, huella de memoria.
- Análisis del gap: se documenta cuánto pierde MobileSAM en calidad y cuánto gana en velocidad. Resultado directamente útil para el proyecto HOT (¿qué modelo desplegar en producción?).
- Análisis crítico esperado: se anticipa que SAM (en cualquiera de sus dos variantes) falle en bosque nativo cerrado de coihue y ñire (la propia propuesta HOT lo señala como "honest caveat" en la página 14). Este failure mode es material valioso para el análisis del TP.
- Opcional si el tiempo lo permite: sumar **SAM2** como tercer punto de comparación.

**Actividades.**
1. Instalar y cargar OpenCLIP y MobileSAM.
2. Escribir `src/models/clip_zeroshot.py` con la lógica de scoring.
3. Escribir `src/models/sam_zeroshot.py` con la lógica de prompting automático, parametrizable para elegir el checkpoint (MobileSAM, SAM ViT-H, opcionalmente SAM2).
4. Correr experimentos sobre el split de test.
5. Loguear métricas en MLflow.
6. Generar visualizaciones: para CLIP, matriz de confusión + ejemplos de éxito y fallo. Para SAM, overlays de máscaras predichas vs ground truth.

**Deliverables.**
- Módulos `clip_zeroshot.py` y `sam_zeroshot.py`.
- Métricas registradas en MLflow.
- Notebook o script de visualización con ejemplos.
- Escritura preliminar de las secciones correspondientes del informe.

**Criterio de done.** Existen métricas cuantitativas y visualizaciones para ambos modelos sobre el split de test.

**Riesgos y mitigación.**
- Si CLIP zero-shot rinde muy alto (>95%), la comparación fine-tuning aporta poco. Se dedica ese tiempo a mejorar el análisis por régimen.
- Si MobileSAM falla catastróficamente en canopy patagónico, ese resultado es en sí una contribución (confirma el caveat de la propuesta HOT).

---

### Fase 4 — Baseline CNN + DETR/ViT

**Objetivo.** Establecer un baseline sólido con YOLOv8 y comparar contra DETR (con backbone ViT y/o ResNet-50) sobre las tareas de detección de objetos (Zonas 1 y 2 del protocolo).

**Por qué es necesaria.** Es la rama más pesada computacionalmente pero también la que responde de forma más directa la pregunta central de investigación (sesgo inductivo con datasets pequeños). Sin esta fase, el TP pierde su columna vertebral argumentativa.

**Ideas principales.**

*Baseline YOLOv8n:*
- Se elige la variante `n` (nano, 3.2M parámetros) por ser entrenable en M4 en tiempo razonable.
- Fine-tuning sobre los samples de El Mallín, partiendo de pesos pre-entrenados en COCO.
- Métricas: mAP@0.5, mAP@0.5:0.95, per clase.

*DETR con backbone ViT:*
- Se emplea la implementación de HuggingFace `transformers` (facebook/detr-resnet-50 como punto de partida) y opcionalmente una variante con backbone ViT-Small.
- Fine-tuning en Colab gratuito (T4 GPU) para caber en 12 horas.
- Métricas comparables: mismas que YOLO.
- Análisis del entrenamiento: curvas de loss, learning rate schedule, cantidad de queries.

*Análisis comparativo:*
- Tablas per clase, per régimen (urban vs forest fringe).
- Análisis explícito del sesgo inductivo: ¿DETR requiere más samples que YOLO para converger? ¿La diferencia se cierra al aumentar el dataset?
- Referencia obligada a la slide 17 de la Clase 1.
- **Benchmark comparativo externo**: se reportan métricas de los modelos entrenados sobre UAVid (dataset benchmark de segmentación aérea) para dar comparabilidad con literatura reciente. Este ejercicio se realiza sobre las clases coincidentes (edificios, árboles) y sirve además para medir el gap de dominio entre UAVid (urbano europeo) y El Mallín (interfaz forestal patagónica).

**Actividades.**
1. Escribir `src/models/yolo_baseline.py` con setup, training loop y evaluación.
2. Escribir `src/models/detr_finetune.py` idem.
3. Escribir el notebook de entrenamiento de DETR para Colab.
4. Entrenar YOLOv8n localmente en M4.
5. Entrenar DETR en Colab.
6. Correr evaluación sobre test.
7. Generar tablas y curvas comparativas.

**Deliverables.**
- Modelos entrenados guardados en `models/`.
- Métricas comparativas en MLflow y en tablas del informe.
- Análisis por régimen y por clase.

**Criterio de done.** Existen métricas comparables (mismo split, mismas clases) para YOLOv8n y DETR, con al menos un análisis por régimen.

**Riesgos y mitigación.**
- Si DETR fine-tuning no converge en 12 h de Colab, plan B: usar DETR pre-entrenado en COCO evaluado zero-shot sobre las clases coincidentes (persona, car → estructuras approx). La comparación sigue siendo válida como "cuánto pierde ViT sin fine-tune vs YOLO con fine-tune".
- Si Colab desconecta a mitad del entrenamiento, guardar checkpoints cada época.

---

### Fase 5 — Integración y compliance index

**Objetivo.** Unificar los outputs de los cuatro modelos (CLIP, SAM, YOLO, DETR) en el compliance index per estructura definido por el protocolo municipal de Villa La Angostura.

**Por qué es necesaria.** Sin esta fase, el TP es una colección de tres experimentos independientes. Con esta fase, es un pipeline coherente que resuelve un problema real y demuestra el valor de cada arquitectura en el contexto del sistema completo.

**Ideas principales.**
- Se implementa el compliance index tal como se define en la página 7 de la propuesta HOT: para cada estructura detectada, se generan buffers geodésicos de 2 m (Zona 1), 10 m (Zona 2) y 30 m (Zona 3), y se chequean reglas geométricas contra las demás detecciones y máscaras.
- Reglas evaluadas:
  - Zona 1: ninguna copa de árbol a menos de 2 m de la estructura, no hay tanques de gas ni leña en la zona.
  - Zona 2: al menos 3 m entre copas, no hay tanques ni leña.
  - Zona 3: distancia mínima a canopy continuo cerrado.
- Se implementa como un módulo independiente que consume las detecciones sin importar qué modelo las produjo (importante para comparar impacto de cada modelo en el índice final).
- Se analiza propagación de error: ¿un mAP 5% más bajo en detección se traduce en cuánta diferencia en el índice final?

**Actividades.**
1. Escribir `src/analysis/compliance.py` con las reglas geométricas.
2. Calcular el índice usando los outputs de cada combinación de modelos.
3. Generar visualizaciones: mapas de compliance por régimen, comparativas de índice según modelo usado.
4. Análisis cuantitativo de propagación de error.

**Deliverables.**
- Módulo `compliance.py` documentado.
- Tablas y mapas de compliance.
- Análisis de sensibilidad.

**Criterio de done.** Se puede correr `python -m src.analysis.compliance --model detr` (o cualquier otro) y obtener un CSV con el índice per estructura + mapas de visualización.

**Riesgos y mitigación.**
- Si los tiempos aprietan, se puede acotar el análisis a un solo modelo (el mejor) y dejar la comparación de sensibilidad como trabajo futuro.

---

### Fase 6 — Informe técnico y README final

**Objetivo.** Producir el informe PDF y el README del repositorio con la calidad exigida por la rúbrica.

**Por qué es necesaria.** El informe es 50% de la entrega. Un buen resultado técnico mal comunicado se percibe como un mal resultado. Media semana completa de escritura y polish no es lujo, es requisito.

**Ideas principales.**

*Estructura del informe (siguiendo Clase 1 slide 4):*
1. Objetivo del proyecto.
2. Arquitectura general (diagrama de flujo + descripción de componentes).
3. Implementación técnica (herramientas, módulos clave).
4. Evaluación (métricas de desempeño de modelos).
5. Resultados y ejemplos.
6. Conclusiones y mejoras futuras.
7. Planificación (tabla con tareas, responsables y estado).

*Estilo de redacción:*
- Voz impersonal a lo largo de todo el documento: "se implementó", "se comparó", "se encontró", "los resultados muestran". Nunca "hicimos", "encontramos", "nuestros resultados".
- Referencias bibliográficas mínimas: Vaswani et al. 2017, Dosovitskiy et al. 2020, Carion et al. 2020, Radford et al. 2021, Kirillov et al. 2023.

*Referencia al contexto real:*
- La sección "aplicación en contexto real" (que la Clase 1 pide explícitamente) se apoya en la propuesta HOT ya escrita. Se resume la problemática de Villa La Angostura, el estándar de defensible space y el rol operacional del pipeline.

**Actividades.**
1. Redactar el informe en LaTeX o Markdown → PDF (pandoc).
2. Generar todas las figuras finales.
3. Revisar el README del repositorio.
4. Completar la tabla de planificación con estados finales.
5. Revisión final: consistencia de números, tipos y unidades, voz impersonal, ortografía.

**Deliverables.**
- `reports/informe_tp_final.pdf`.
- `README.md` actualizado.
- Repositorio en estado entregable.

**Criterio de done.** El informe cubre las 7 secciones exigidas, todas las figuras están numeradas y referenciadas, la voz impersonal se mantiene en todo el documento, y no hay tareas técnicas pendientes en el repo.

**Riesgos y mitigación.**
- La escritura siempre lleva más que lo estimado. La segunda mitad de la semana 6 se reserva exclusivamente para redacción, sin experimentos nuevos.

---

### Fase 7 — Presentación de Clase 8

**Objetivo.** Comunicar los resultados en 15 minutos con foco en métricas, visualizaciones y aplicación real.

**Por qué es necesaria.** La Clase 1 lo pide explícitamente y forma parte de la evaluación. Además, es la oportunidad de defender ante los docentes las decisiones tomadas y responder preguntas.

**Ideas principales.**
- 15 minutos son pocos: aproximadamente 8-10 slides efectivas.
- Foco pedido por la rúbrica: análisis de resultados con énfasis en métricas, visualizaciones del modelo, y explicación de cómo el modelo se aplica en un contexto real.
- Se sugiere estructura: problema (2 min), pipeline (3 min), resultados por modelo (6 min), análisis integrado + compliance index (3 min), aplicación real y trabajo futuro (1 min).

**Actividades.**
1. Diseñar el deck.
2. Preparar visualizaciones finales (mapas de compliance, ejemplos de detección, curvas comparativas).
3. Escribir el guion.
4. Ensayar timing (al menos dos ensayos completos).
5. Preparar respuestas a preguntas anticipables.

**Deliverables.**
- Deck en PDF.
- Guion escrito.

**Criterio de done.** El deck se recorre en 15 minutos ± 1, con transiciones claras entre secciones.

---

## 7. Métricas de evaluación

| Tarea | Modelo | Métrica principal | Métrica secundaria |
|---|---|---|---|
| Clasificación de tiles (canopy Zone 3) | CLIP | Accuracy | F1, matriz de confusión |
| Segmentación de canopy | SAM | mean IoU | Boundary F1, análisis de failure modes |
| Detección de objetos (Zonas 1 y 2) | YOLOv8n / DETR | mAP@0.5 | mAP@0.5:0.95 per clase |
| Compliance index integrado | Pipeline completo | % estructuras compliant per régimen | Sensibilidad al error de detección |

Todas las métricas se reportan además desagregadas por régimen (urban matrix vs forest fringe), siguiendo la hipótesis dual-regime de la propuesta HOT (página 9).

---

## 8. Bibliografía mínima

Los papers listados a continuación son de lectura obligada para el desarrollo del TP y para las referencias del informe:

- Vaswani, A. et al. (2017). *Attention Is All You Need.* NeurIPS.
- Dosovitskiy, A. et al. (2020). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR.
- Carion, N. et al. (2020). *End-to-End Object Detection with Transformers.* ECCV. (DETR)
- Radford, A. et al. (2021). *Learning Transferable Visual Models From Natural Language Supervision.* ICML. (CLIP)
- Kirillov, A. et al. (2023). *Segment Anything.* ICCV. (SAM)
- Zhang, C. et al. (2023). *Faster Segment Anything: Towards Lightweight SAM for Mobile Applications.* (MobileSAM)
- Zhang, H. et al. (2023). *Understanding Why ViT Trains Badly on Small Datasets: An Intuitive Perspective.* arXiv.

---

## 9. Referencias del contexto HOT

- Propuesta LAWAL HOT Open Call 2026: `../LAWAL_HOT_Proposal.pdf`.
- Convocatoria HOT "Map your city with fAIr": https://hotosm.github.io/fAIrswipe-call-2026/
- Repositorio fAIr: https://github.com/hotosm/fAIr
- Imagen de El Mallín en OpenAerialMap: (URL a completar al comienzo de Fase 2)

---

## 10. Estado del plan

Documento vivo. Se actualiza al cerrar cada fase con: hallazgos, desviaciones respecto al plan original, decisiones tomadas.

| Fase | Estado | Fecha de cierre |
|---|---|---|
| 1 | Pendiente | — |
| 2 | Trabajo preparatorio en curso | — |
| 3 | Pendiente | — |
| 4 | Pendiente | — |
| 5 | Pendiente | — |
| 6 | Pendiente | — |
| 7 | Pendiente | — |

### Trabajo preparatorio ya realizado (previo a la aprobación de Fase 0)

- **Sondeo del API de fAIr**: se confirmó existencia de tres datasets etiquetados de Villa La Angostura (382, 383, 393) con modelo YOLOv8 asociado y accuracy oficial. Endpoints funcionales en `api-prod.fair.hotosm.org`.
- **Descarga de bibliografía**: 13 papers arxiv descargados y organizados en `references/` con orden de lectura pedagógico.
- **Descarga de datasets externos**:
  - UAVid: completo, 4.0 GB, 9 parquet files en `data/external/uavid/`.
  - Aeroscapes: descarga parcial (rate-limit HTTP 429 tras 496 archivos), retry en curso con menos workers.
- **Ambiente Python**: venv creado en `.venv/` con `huggingface_hub` instalado.

