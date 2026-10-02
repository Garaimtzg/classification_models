# classification_models

Pruebas de modelos de clasificación y de decisión de Hugging Face.

---

## CLM: modelos que deciden en vez de conversar

- Modelo: [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (licencia libre Apache 2.0, salió el 23/09/2026)
- Código: [github.com/Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM). Se instala con `pip install contrastive-lm`
- Lo han hecho investigadores de Stanford y NVIDIA (Jacky Kwok, Azalia Mirhoseini, Christopher Ré y otros)
- Notebook para probarlo: [`notebooks/clm_v0_1_8b_colab.ipynb`](notebooks/clm_v0_1_8b_colab.ipynb) (Google Colab Pro con GPU L4 o A100)

### Qué es

Un LLM normal, como ChatGPT o Claude, **escribe** respuestas. Un CLM no escribe nada: **elige**.

Le das dos cosas:

1. Una situación y una pregunta. Por ejemplo: *"Un cliente dice que le han cobrado dos veces. ¿Qué departamento lo lleva?"*
2. Una lista de opciones. Por ejemplo: *facturación, soporte técnico, ventas*.

Y te devuelve el porcentaje que le da a cada opción. Por ejemplo: *facturación 94 %, soporte técnico 4 %, ventas 2 %*.

**Cómo funciona por dentro, en pocas palabras:**

- Usa Qwen3-8B, un LLM conocido, pero solo para **convertir textos en números** (un "embedding", es decir, un resumen numérico del significado del texto). No lo usa para escribir.
- Encima tiene dos piezas pequeñas (75 MB en total): una para la situación y otra para las opciones.
- Se entrenaron para que la situación y la opción correcta "se parezcan" en números y las incorrectas no. Así, elegir es tan simple como mirar qué opción se parece más.
- Para entrenarlo usaron decenas de millones de preguntas con su respuesta correcta, ejemplos de respuestas incorrectas que parecen correctas (para que aprenda a no dejarse engañar) y cerca de un millón de ejemplos de agentes de IA haciendo tareas.

**Tipos de pregunta que acepta:**

| Tipo | Para qué sirve | Ejemplo de respuesta |
|---|---|---|
| `Noul` | Preguntas de sí o no | "¿Es urgente?" → 0,41, es decir, un 41 % de que sí |
| `Choice` | Elegir una opción entre varias | "¿Qué departamento?" → facturación |
| `Score` | Puntuar en una escala | "¿Cómo de enfadado está?" de 0 a 2 → 1,98 |
| `rank` | Ordenar una lista de opciones de mejor a peor | Varias soluciones de código ordenadas de más a menos correcta |

### En qué se diferencia de un LLM normal

| | LLM normal (ChatGPT, Claude, Llama…) | CLM |
|---|---|---|
| Qué hace | Escribe texto | Da un porcentaje a cada opción que le pasas |
| Velocidad | Lento: escribe palabra a palabra | Muy rápido: una sola pasada |
| Muchas opciones | Si le das 1.000 opciones, el mensaje es enorme y lento | Recuerda las opciones que ya ha visto, así que 1.000 opciones cuestan casi lo mismo que 3 |
| Fiabilidad del formato | Puede inventarse una opción que no existe o contestar con otro formato | Nunca se sale de las opciones que le das |
| Qué no puede hacer | — | No explica por qué elige, no razona paso a paso y no puede proponer opciones nuevas |
| Adaptarlo a tu problema | Caro | Muy barato: solo se reentrenan las dos piezas pequeñas |

Se parece más a un **clasificador** que a un chat. Encaja bien para:

- Repartir tickets o correos entre departamentos.
- Que un agente de IA elija qué herramienta usar.
- Elegir la mejor de varias respuestas generadas por otro modelo.
- Decidir el siguiente movimiento en un juego o en una pantalla, en tiempo real.
- Clasificar textos con etiquetas descritas en lenguaje normal.

### Los distintos modelos de la familia

| Modelo | Para qué es | Comentario |
|---|---|---|
| [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | El modelo general | Es el principal. Funciona sin entrenarlo para tu tarea. Solo inglés |
| [deepswe-clm-heads-8k](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k) | Versión especializada en revisar soluciones de programación | Sale del anterior, reentrenado para esa tarea concreta. No sirve para lo demás |
| `clm-raw` | Versión de prueba que trae el paquete | Es el modelo **sin** las dos piezas entrenadas. Sirve para ver cuánto mejoran esas piezas |
| [CLM-Qwen3.5-2B](https://huggingface.co/jayavibhav/clm-qwen3.5-2b) | Versión hecha por la comunidad (no oficial) | Mismo método con un modelo 4 veces más pequeño. Acepta textos mucho más largos. Pensado para clasificar documentos |
| CLM-35B | Versión grande, anunciada | También entenderá imágenes. Todavía no ha salido |

### ¿Son buenos?

**Lo que dicen sus autores**:

- Sin entrenamiento extra, acierta más o menos lo mismo que Jev (otro modelo de decisión, que es con el que se comparan), pero es **hasta 9 veces más rápido**. Lo probaron en manejar un ordenador, jugar a videojuegos y elegir herramientas.
- Con 1.000 opciones, es **13 veces más rápido** que Jev.
- Reentrenado para revisar código, consigue los **mejores resultados publicados** en dos pruebas de programación: DeepSWE (81,6 %) y Terminal-Bench 2.1 (87,6 %).

**Lo que hay que tener en cuenta:**

- Los mejores resultados son de **versiones reentrenadas** para cada tarea, no del modelo general.
- Las pruebas son **pequeñas**. El 81,6 % de DeepSWE significa acertar 31 tareas de 38. Quedarse directamente con la primera solución ya acertaba 28, así que la mejora real es de 3 tareas.
- Aunque las piezas propias pesen poco, **necesita Qwen3-8B funcionando**, y eso pide una tarjeta gráfica con unos 16 GB de memoria.
- Está entrenado en **inglés**. En español puede funcionar peor; el notebook incluye una prueba.
- **Siempre elige alguna opción.** Si la correcta no está en la lista, elegirá la menos mala.
- Es una primera versión (v0.1).

**Resumen:** cuando hay que **elegir entre opciones**, es mucho más rápido y barato que preguntarle a un LLM normal. Para escribir, explicar o razonar, sigue haciendo falta un LLM normal.

### Cómo probarlo

**En Google Colab Pro (lo recomendado):** abre [`notebooks/clm_v0_1_8b_colab.ipynb`](notebooks/clm_v0_1_8b_colab.ipynb) y elige una GPU **L4 o A100**. Con la T4 gratuita no hay memoria suficiente. El notebook:

1. Instala todo y pone en marcha el modelo.
2. Hace preguntas de sí/no, de elegir y de puntuar, en inglés y en español.
3. Ordena opciones: qué herramienta usar y cuál de varias soluciones de código es la buena.
4. Hace una pequeña prueba de clasificación de 400 noticias en 4 temas. Compara CLM con su versión sin piezas entrenadas (`clm-raw`) y con BART-MNLI, un clasificador clásico. Mide aciertos y velocidad.
5. Comprueba cuánto se gana porque recuerda las opciones ya vistas, usando 1.000 opciones.
6. Opcionalmente, abre una página web para probarlo a mano.

**En tu propio ordenador** necesitas una tarjeta gráfica con al menos 24 GB de memoria:

```bash
pip install contrastive-lm
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &
clm-serve   # página de prueba en http://localhost:8700/
```

### Enlaces

- Ficha del modelo: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- Código: https://github.com/Contrastive-LM/CLM
- Blog de los autores: https://contrastive-lm.notion.site
- Anuncio de Azalia Mirhoseini: https://x.com/Azaliamirh/status/2102958024323428362
- Anuncio de Jacky Kwok: https://x.com/jackyk02/status/2102905335925424285
- Explicación sencilla: https://blog.dailydoseofds.com/p/contrastive-language-model-clearly

---

## Entrenar un LLM desde cero: train-llm-from-scratch

- Repo: [FareedKhan-dev/train-llm-from-scratch](https://github.com/FareedKhan-dev/train-llm-from-scratch) (licencia MIT)
- Notebook para probarlo: [`notebooks/train_llm_from_scratch_colab.ipynb`](notebooks/train_llm_from_scratch_colab.ipynb) (Google Colab Pro, mejor con A100)

### Qué es

Es un tutorial con código para **construir y entrenar un modelo tipo LLM desde cero**, en miniatura.
Todo está escrito a mano en PyTorch, sin usar librerías que lo den hecho (como `transformers`), así que se puede leer y entender cada pieza.

Recorre las mismas fases que siguen los modelos de verdad:

| Fase | Qué hace, en pocas palabras |
|---|---|
| **1. Preparar datos** | Descarga mucho texto en inglés (The Pile) y lo convierte en números, que es lo único que entiende el modelo |
| **2. Preentrenamiento** | El modelo lee el texto y aprende a adivinar la siguiente palabra. Aquí aprende el idioma |
| **3. SFT** | Le enseñamos a contestar preguntas con ejemplos de pregunta y respuesta. Aquí pasa de "continuar texto" a "contestar" |
| **4. Modelo de recompensa** | Un modelo aparte aprende a puntuar qué respuesta es mejor |
| **5. DPO / PPO** | El modelo aprende a preferir las respuestas buenas frente a las malas |
| **6. GRPO** | Aprendizaje por refuerzo con problemas de matemáticas: si acierta el resultado, premio. Es la técnica de DeepSeek-R1 |
| **7. Evaluar y chatear** | Mide cuántos problemas de matemáticas resuelve y deja hablar con él |

### ¿Es factible?

**Sí, pero en pequeño.** Depende mucho de qué tamaño de modelo quieras:

| Tamaño | Dónde | Tiempo aproximado | Resultado esperado |
|---|---|---|---|
| 13 millones de parámetros | Colab (incluso la T4 gratuita) | Minutos | Palabras reales y frases con algo de gramática, pero sin sentido |
| **77 millones** (el del notebook) | Colab Pro con A100 o L4 | Unas 1–3 horas en total | Frases bastante naturales, pero se inventa todo. Puede seguir el formato de chat |
| 400 millones | 1–2 GPUs grandes (A100/H100) | Muchas horas o días | Algo mejor, todavía muy lejos de un modelo útil |
| 1.000 millones o más | Varias GPUs grandes | Semanas | El repo dice que "cabe" en una A100, pero caber no es lo mismo que entrenarlo bien: harían falta miles de millones de palabras de texto |

Lo he probado en pequeño (un modelo de 13 millones, solo con procesador): **todas las fases funcionan de principio a fin**, y las pruebas automáticas del repo pasan.
Sin tarjeta gráfica va a unos 1.300 tokens por segundo, demasiado lento para algo más que comprobar que el código funciona.

### ¿Merece la pena?

**Para aprender, mucho.** Es de lo más completo que hay: en un solo sitio y con código sencillo se ve todo el proceso, desde el texto en bruto hasta el aprendizaje por refuerzo.

**Para conseguir un modelo útil, no.** Hay que tener claras las expectativas:

- El autor entrenó el modelo de 77 millones con 2 GPUs L40 en 2.000 pasos. La pérdida bajó de 11,1 a 3,8, que está bien para ese tamaño.
- Su modelo de recompensa y su DPO aciertan un **57 %** al elegir la respuesta buena, cuando al azar sería un 50 %. Es decir, mejoran muy poco.
- **No publica resultados de matemáticas** (GSM8K). Con un modelo tan pequeño lo esperable es acertar un **0–2 %**. Por comparación, los modelos de 7.000–8.000 millones aciertan entre el 50 % y el 90 %.
- Como referencia, GPT-2 pequeño (2019) tenía 124 millones de parámetros y se entrenó con unos 10.000 millones de tokens. Aquí usamos 77 millones de parámetros y unos 200 millones de tokens.
- El código prioriza que se entienda sobre la velocidad. Por ejemplo, la atención se calcula cabeza a cabeza en lugar de usar las versiones optimizadas de PyTorch, así que entrena más lento de lo que podría.

**Si lo que quieres es un modelo propio que funcione bien**, es mucho mejor partir de un modelo ya preentrenado (por ejemplo Qwen o Llama) y ajustarlo con tus datos.

### Qué hace el notebook

1. Detecta la GPU y ajusta solo el tamaño del modelo y del lote.
2. Instala el repo y pasa sus pruebas automáticas.
3. Descarga y prepara unos 250 millones de tokens de The Pile.
4. Preentrena el modelo de 77 millones (2.000 pasos) y dibuja la gráfica de la pérdida.
5. Prueba el modelo base: solo sabe continuar texto.
6. Hace el SFT con unos 75.000 ejemplos de pregunta y respuesta y prueba a chatear.
7. Opcional: DPO y GRPO.
8. Mide los aciertos en problemas de matemáticas en cada fase y deja chatear con el modelo final.

Los modelos entrenados se pueden guardar en Google Drive para no perderlos si Colab se desconecta.

### Enlaces

- Repo: https://github.com/FareedKhan-dev/train-llm-from-scratch
- Documentación del autor: https://fareedkhan-dev.github.io/train-llm-from-scratch/

---

## Jev: el original de TypeSafe

- Empresa: [TypeSafe AI](https://typesafe.ai/) (San Francisco, fundada en 2026)
- Documentación: [docs.typesafe.ai](https://docs.typesafe.ai/) · Cuenta y clave: [console.typesafe.ai](https://console.typesafe.ai/)
- Notebook para probarlo: [`notebooks/jev_typesafe_colab.ipynb`](notebooks/jev_typesafe_colab.ipynb) (no necesita GPU, vale el Colab gratis)

### Qué es

Jev es el modelo con el que TypeSafe lanzó la idea de los **modelos "System One"**: modelos que no conversan, sino que **toman decisiones rápidas** para que las use un programa.
CLM y el resto de **alternativas libres** que están saliendo copian su forma de trabajar.

- Le mandas un texto (el "estado") y preguntas de tipo `Noul` (sí/no), `Choice` (elegir) o `Score` (puntuar).
- Te devuelve porcentajes y, además, **la confianza**: lo seguro que está. La idea es que el programa actúe solo cuando la confianza es alta y mande a una persona lo dudoso.
- TypeSafe dice que lo entrena con un método propio ("RLCD") pensado para que esos porcentajes sean fiables.

### Jev frente a las alternativas libres

| | Jev (TypeSafe) | CLM (libre) |
|---|---|---|
| Cómo se usa | Por internet (API), con cuenta y clave | Lo descargas y lo ejecutas tú |
| Qué necesitas | Nada especial, vale cualquier ordenador | Una tarjeta gráfica de unos 24 GB |
| Precio | 0,042 $ por millón de tokens de entrada (la salida es gratis) | Gratis, pero pagas tu GPU |
| Tus datos | Salen de tu ordenador hacia TypeSafe (dicen que no entrenan con ellos) | No salen de tu máquina |
| Adaptarlo a tu problema | No se puede reentrenar: solo se ajusta escribiendo bien las preguntas | Se pueden reentrenar sus piezas pequeñas |
| Confianza | Sí, para cada respuesta | Solo los porcentajes |
| Textos largos | Hasta 32.000 tokens de texto | 2.048 tokens por defecto |
| Idiomas | Mejor en inglés; otros funcionan peor | Solo inglés |

### ¿Se puede probar gratis?

**Ahora mismo, no.** Según varias webs que lo siguen, al abrir el registro (20/09/2026) regalaban 5 $ de saldo, pero lo pararon por exceso de demanda
y desde el 27/09/2026 las cuentas nuevas no reciben crédito gratis. Lo mejor es comprobarlo en [console.typesafe.ai](https://console.typesafe.ai/).

Eso sí, **probarlo cuesta muy poco**: con 1 $ se procesan unos 24 millones de tokens. Se calcula que el notebook completo (unas 400 llamadas) gasta **menos de 1 céntimo**.
La consola también tiene un [Playground](https://console.typesafe.ai/playground) para probarlo a mano desde el navegador.

### Lo que dicen ellos y lo que reconocen que hace mal

**Lo que dicen** (datos de su web):
- **193 veces más rápido** y **444 veces más barato** que un LLM normal en este tipo de tareas (0,11 s frente a 8,6 s por decisión).
- **Cero alucinaciones**, porque siempre responde con una de las opciones que le das y con su confianza.

**Lo que reconocen que hace mal** (lo publican ellos mismos en la [lista de puntos débiles de Jev 1.13](https://docs.typesafe.ai/model-jaggedness/jev-1.13)):
- **No sabe contar ni hacer cuentas.** Recomiendan hacer las cuentas en el código.
- **No compara bien fechas.**
- Se lía con **negaciones dobles** y preguntas enrevesadas.
- Se lee las preguntas **al pie de la letra**: responde a lo que escribes, no a lo que querías decir.
- Si el texto tiene mucha **información que no viene al caso**, acierta menos.
- Se le puede **engañar** con textos escritos a propósito para despistarle.
- Preguntar lo mismo de dos formas (por ejemplo, "¿pide un reembolso?" y "¿pide algo que no sea un reembolso?") puede dar resultados que no cuadran entre sí.

### Qué hace el notebook

1. Instala el SDK oficial (`typesafe-sdk`) y lee tu clave de los Secretos de Colab (con el nombre `TYPESAFE_API_KEY`).
2. Repite las pruebas del notebook de CLM: el ticket de soporte en inglés y en español, elegir herramienta y elegir la solución de código correcta.
3. Clasifica **las mismas 400 noticias** de AG News que el notebook de CLM y mide aciertos, velocidad y coste, para poder comparar los dos.
4. Comprueba si su **confianza sirve**: cuántas noticias se podrían automatizar con cada nivel de confianza y cuánto acierta en ellas.
5. Pone a prueba sus **puntos débiles** con casos sencillos de contar, hacer cuentas, comparar fechas y negaciones.
6. Lleva la cuenta de lo gastado y guarda los resultados en `jev_resultados.json`.

### Enlaces

- Web de TypeSafe: https://typesafe.ai/
- Documentación: https://docs.typesafe.ai/
- Precios y modelos: https://docs.typesafe.ai/models
- Puntos débiles de Jev 1.13: https://docs.typesafe.ai/model-jaggedness/jev-1.13
- SDK de Python: https://github.com/typesafe-ai/typesafe-sdk-python
- ¿Es gratis Jev?: https://www.layer3labs.io/guides/is-jev-free
