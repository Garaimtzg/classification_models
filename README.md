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

**Lo que dicen sus autores** (nadie lo ha comprobado aún por su cuenta):

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

En este equipo (WSL con 3,7 GB de RAM y sin tarjeta gráfica) no se puede ejecutar.

### Enlaces

- Ficha del modelo: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- Código: https://github.com/Contrastive-LM/CLM
- Blog de los autores: https://contrastive-lm.notion.site
- Anuncio de Azalia Mirhoseini: https://x.com/Azaliamirh/status/2102958024323428362
- Anuncio de Jacky Kwok: https://x.com/jackyk02/status/2102905335925424285
- Explicación sencilla: https://blog.dailydoseofds.com/p/contrastive-language-model-clearly
