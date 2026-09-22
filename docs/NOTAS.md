# Notas para la presentacion

Piezas didacticas que se van acumulando durante el desarrollo. Sirven de base para armar el deck de Clase 8 y para explicaciones puntuales durante la defensa.

---

## Fotos del dron, ortomosaico y tiles: son tres cosas distintas

**Fotos originales del dron.** Cientos de tomas individuales que hace el dron durante el vuelo. Cada foto tiene su angulo, su posicion GPS y su perspectiva. No estan corregidas geometricamente.

**Ortomosaico.** El TIF que devuelve Drone-TM despues de procesar todas esas fotos. Es una sola imagen grande, stitched, con la perspectiva corregida y proyectada al plano. Cada pixel esta ubicado en su coordenada geografica real. Vista cenital pura, como si la camara hubiera estado exactamente arriba de cada punto del terreno.

**Tiles.** Recortes cuadrados del ortomosaico (512x512 pixeles en este proyecto). Cada tile es una porcion georreferenciada de la imagen grande. Se usan para alimentar los modelos.

### Por que tilear

- Los ortomosaicos pesan cientos de MB y no entran completos en memoria del modelo.
- Los modelos de vision esperan input chico y consistente.
- Los tiles conservan la georreferencia, asi que despues se pueden medir distancias reales en metros.

### Por que no usar las fotos originales

- No hay consistencia geometrica entre ellas (perspectiva variable).
- No tienen georreferencia por pixel.
- Un modelo entrenado sobre ellas no generaliza a ortomosaicos.

---

## Como se eligio el tamano de tile (1024x1024 a 4 cm/px)

Los ortomosaicos vienen a 4 cm/px de resolucion nativa. Cada tile es de 1024x1024 pixeles con 20% de overlap.

### Que representa cada tile en el mundo real

Un tile de 1024 pixeles a 4 cm/px representa un cuadrado de 40x40 metros en el terreno.

### Por que 40x40 m y no otro tamano

**El compliance municipal se calcula por vivienda.** El protocolo define tres zonas concentricas alrededor de cada casa: Zona 1 (0 a 2 m), Zona 2 (2 a 10 m) y Zona 3 (10 a 30 m). Para que un modelo pueda inferir compliance sobre una casa, el tile debe capturar al menos la casa completa mas las zonas cercanas alrededor.

Comparacion de tamanos evaluados:

| Tile | Area | Que se ve | Descartado por |
|---|---|---|---|
| 256x256 | 10x10 m | Fragmento de casa | No entra ni una casa entera |
| 512x512 | 20x20 m | Casa justa o cortada | Al testear, se vio que muchas casas de VLA no entran completas |
| **1024x1024** | **40x40 m** | **Casa completa + Zonas 1, 2 y parte de 3** | Elegido |
| 1024x1024 a 8 cm/px | 80x80 m | Casa + Zonas 1, 2, 3 completas + vecinos | Se pierde detalle fino de fuel objects |

### Sesgo hacia estandares de la industria

1024x1024 es tambien el input nativo de SAM, uno de los foundation models que usamos zero-shot. No hay resize durante inferencia, se procesa directo. Ver `references/12_kirillov_2023_sam.pdf`.

### Overlap del 20%

Sin overlap, un objeto que cae en el borde de un tile queda cortado en dos y potencialmente perdido. Con 20%, cada objeto aparece entero en al menos uno de los tiles vecinos.

### Cifras finales del tileado sobre Correntoso-Arauco

- 25 ortofotos originales de Drone-TM (3.6 GB en `data/raw/correntoso-arauco/`).
- 2056 tiles utiles generados (1932 mas descartados por transparencia mayor a 50 por ciento).
- 6.9 GB en `data/interim/tiles/correntoso-arauco/`.
- Tiempo de tileado sobre las 25 ortofotos: 1 minuto y 22 segundos.

