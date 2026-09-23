# Referencias bibliográficas — TP VpC III

Los 13 PDFs de esta carpeta corresponden a la bibliografía relevante para el proyecto: 10 papers de arXiv listados por la cátedra en la Clase 1, más 3 papers adicionales incorporados por el reencuadre del TP (CLIP, SAM, MobileSAM).

El prefijo numérico (`01_`, `02_`, ...) indica el **orden sugerido de lectura**, no cronológico, sino pedagógico: primero los fundamentos y la pregunta central, después las variantes, y al final las arquitecturas específicas usadas por el TP.

## Orden de lectura

### Bloque 1 · Fundamentos (leer sí o sí antes de tocar código)

| # | Archivo | Por qué acá |
|---|---|---|
| 01 | `01_vaswani_2017_attention_is_all_you_need.pdf` | Paper fundacional. Define scaled dot-product attention y multi-head attention. Todo lo demás asume esto. |
| 02 | `02_dosovitskiy_2020_vit.pdf` | Primera aplicación seria de transformers a imágenes. Introduce el concepto de imagen como secuencia de patches. |
| 03 | `03_zhang_2023_why_vit_trains_badly_small_datasets.pdf` | Analiza exactamente la pregunta central del TP y de la Clase 1 slide 17: por qué el sesgo inductivo débil de ViT lo hace frágil con datasets pequeños. |

### Bloque 2 · Variantes de ViT (leer al menos DeiT y Swin)

| # | Archivo | Por qué acá |
|---|---|---|
| 04 | `04_touvron_2021_deit.pdf` | Continuación directa del paper de Zhang: DeiT propone estrategias de training data-efficient para ViT. Muy relevante para el TP. |
| 05 | `05_liu_2021_swin_transformer.pdf` | La variante ViT más influyente. Introduce jerarquía + shifted windows. Backbone de muchos modelos posteriores. |
| 06 | `06_wang_2021_pyramid_vision_transformer.pdf` | Otro enfoque piramidal, alternativa a Swin. |
| 07 | `07_yuan_2021_t2t_vit.pdf` | Otro intento de mejorar training desde cero en datasets no gigantes. |
| 08 | `08_wu_2021_cvt.pdf` | Hibrida convoluciones con transformers, recupera parte del sesgo inductivo. |
| 09 | `09_mehta_2021_mobilevit.pdf` | Variante liviana, especialmente relevante por la restricción de M4 en el TP. |

### Bloque 3 · Modelos específicos usados en el TP

| # | Archivo | Fase del TP |
|---|---|---|
| 10 | `10_carion_2020_detr.pdf` | Fase 4 · detección end-to-end con transformers. Modelo central de la comparación arquitectural. |
| 11 | `11_radford_2021_clip.pdf` | Fase 3 · clasificación zero-shot de tiles. Base de la Clase 5. |
| 12 | `12_kirillov_2023_sam.pdf` | Fase 3 · segmentación promptable. Base de la Clase 6. |
| 13 | `13_zhang_2023_mobilesam.pdf` | Fase 3 · variante liviana de SAM, elegida para correr en MacBook M4. |

## Referencias no descargables mencionadas por la cátedra

Ambas están detrás de paywall. Se citan en el informe si es necesario, pero no son requeridas.

- **Rothman, D. (2024).** *Transformers for Natural Language Processing and Computer Vision.* Packt Publishing, 3rd edition. Libro completo.
- **"Transformers and Visual Transformers"** — capítulo del libro *Neuromethods* vol. 197. Springer.

## Cómo se propone leer esto

Con seis semanas de cronograma y ninguna semana holgada, la sugerencia realista es:

- **Semana 1** (mientras se hace setup): 01, 02, 03. Son los tres imprescindibles.
- **Semana 2** (mientras se prepara data): 04 y 05 (DeiT + Swin). Los otros del bloque 2 se hojean.
- **Semana 3** (mientras se corre CLIP + SAM zero-shot): 11 y 12 (CLIP + SAM). 13 se lee en paralelo si se elige MobileSAM.
- **Semana 4** (mientras se entrena DETR): 10 (DETR) en profundidad.
- **Semanas 5 y 6**: consulta puntual para redacción del informe.
