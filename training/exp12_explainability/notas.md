# EXP-12: método y límites

Checkpoint: `training/checkpoints/detector_best.pth`; reutilizado: True.
Los ejemplos son los primeros dos reales y los primeros dos de cada generador
presentes en el test fijo, sin selección por confianza o por aspecto del mapa.
Original = vista del archivo a 128 px; entrada = imagen tras el resize compartido.
La superposición se alinea con esta última, que es la imagen evaluada.

Grad-CAM manual con forward hook en `spatial.layer4` y `autograd.grad`:
promedio espacial del gradiente, suma ponderada de activaciones y ReLU.
Objetivo: logit de la clase predicha (z para falsa, -z para real), sin sigmoid
para evitar saturación. La frecuencia permanece fija. No se aplica Grad-CAM
a la rama FFT. Método de [Selvaraju et al., ICCV 2017](https://openaccess.thecvf.com/content_iccv_2017/html/Selvaraju_Grad-CAM_Visual_Explanations_ICCV_2017_paper.html).

Mapas nativos de 4×4, interpolados a 128×128 y normalizados individualmente:
no permiten delimitar estructuras anatómicas finas ni comparar intensidad
absoluta entre imágenes. Todos los mapas elegidos son finitos y no constantes;
sus rangos, desviaciones y normas del gradiente están en `metadata.json`.
Esto verifica variación numérica, no validez causal de la explicación.

Frecuencia: gradiente local del **logit falsa** respecto a cada uno de los 32
bins normalizados que recibe el MLP. Media del valor absoluto en las 89
imágenes de test; también se conservan gradientes individuales y medias con signo.
Se elige gradiente directo para medir sensibilidad local sin introducir una
referencia arbitraria para integrated gradients. No mide contribución acumulada,
ni fracción de la decisión, ni el gradiente respecto a píxeles/FFT sin normalizar.
Los bins con mayor media absoluta en esta ejecución (índices 0–31) son [19, 17, 16, 25, 1].

## Observaciones automáticas reproducibles

- `data/real/real_0006.png`: P(falsa)=0.1243; máximo interpolado en (x=79, y=48); std del mapa nativo=0.035044.
- `data/real/real_0009.png`: P(falsa)=0.0252; máximo interpolado en (x=47, y=48); std del mapa nativo=0.0669053.
- `data/fake/fake_0177.png`: P(falsa)=0.9792; máximo interpolado en (x=79, y=79); std del mapa nativo=0.0854703.
- `data/fake/fake_0224.png`: P(falsa)=0.9802; máximo interpolado en (x=80, y=48); std del mapa nativo=0.103781.

La revisión visual específica de esta corrida se documenta en `../exp0912_notas.md`.

## Revisión visual de la ejecución DCGAN (2026-09-18)

En real_0006 se destaca una región amplia superior/central del tórax; en
real_0009, una zona central/izquierda. En ambas el fondo negro superior
presenta menor intensidad. En fake_0177 domina una zona central; fake_0224
muestra una región más extensa hacia la mitad derecha y el borde superior.
No hay concentración exclusiva en background. Las regiones se solapan con
anatomía y textura visibles, pero la resolución nativa de 4×4 no permite
identificar una estructura anatómica o artefacto generativo concreto como
causa de la decisión. No se concluye que el detector reconozca anatomía.

Las 32 barras no son uniformes. Destacan los bins 19,17,16,25 y también el 1;
la sensibilidad no se concentra exclusivamente en las frecuencias más altas.
No se compara esta magnitud con la contribución de la rama espacial.
