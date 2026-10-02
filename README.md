# classification_models

Pruebas de modelos de clasificación y de decisión de Hugging Face.

---

## Contrastive Language Models (CLM): modelos de lenguaje *decisionales*

- Modelo: [Contrastive-LM/CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) (Apache 2.0, publicado el 23/09/2026)
- Código: [github.com/Contrastive-LM/CLM](https://github.com/Contrastive-LM/CLM) · paquete `pip install contrastive-lm`
- Autores: Jacky Kwok, Hangoo Kang, Tarun Suresh, Jon Saad-Falcon, Marco Pavone, Christopher Ré, Azalia Mirhoseini (Stanford / NVIDIA)
- Notebook para probarlo: [`notebooks/clm_v0_1_8b_colab.ipynb`](notebooks/clm_v0_1_8b_colab.ipynb) (Google Colab Pro, GPU L4 o A100)

### Qué es

Un CLM es un modelo **"System One"**: no conversa ni genera texto, sino que **toma decisiones entre opciones cerradas**.
Le das un *estado* (contexto + pregunta) y una lista de *candidatos* (acciones, respuestas, herramientas, etiquetas…)
y devuelve una distribución de probabilidad sobre esos candidatos.

Arquitectura de CLM-v0.1-8B:

```
estado ──► Qwen3-8B (congelado, embedding del último token) ──► cabeza de estado  ──┐
                                                                                    ├─► coseno × escala ─► softmax
candidato ► Qwen3-8B (congelado, embedding del último token) ──► cabeza de acción ──┘
```

- El encoder es **Qwen3-8B sin modificar**. Lo único entrenado son dos MLP pequeñas (~20M parámetros, fichero de 75 MB).
- Entrenamiento contrastivo (**InfoNCE bidireccional**, como CLIP pero estado↔acción) en tres fases:
  pre-entrenamiento con ~60M pares pregunta/respuesta de Nemotron, mid-training con ~30M negativos difíciles
  sintéticos y post-entrenamiento con ~1M trayectorias de agentes.
- API de preguntas tipadas (compatible con TypeSafe):
  - `Noul`: sí/no → probabilidad de que sea cierto.
  - `Choice`: elegir una opción entre varias.
  - `Score`: escala ordenada → nivel esperado.
  - `rank`: ordenar candidatos libres (best-of-N, herramientas, siguiente movimiento).

### En qué se diferencian de un LLM conversacional

| | LLM conversacional (Qwen, Llama, Claude…) | CLM |
|---|---|---|
| Salida | Texto libre, token a token | Probabilidades sobre **los candidatos que tú le das** |
| Coste por decisión | Prompt + varios tokens generados (decodificación autoregresiva) | 1 pasada del encoder sobre el estado + 1 producto escalar por candidato |
| Candidatos | Tienen que caber en el prompt; 1.000 opciones = prompt enorme | Se codifican **por separado y se cachean**; 1.000 opciones cuestan casi lo mismo que 3 |
| Calibración | Hay que parsear la respuesta o mirar logprobs | La salida *es* una distribución (softmax) |
| Alucinaciones de formato | Puede inventar una opción que no existe | Imposible: solo puntúa lo que le pasas |
| Flexibilidad | Puede razonar, explicar, proponer opciones nuevas | No razona en voz alta ni genera; no puede decir "ninguna" si no se lo ofreces |
| Ajuste fino | Caro (o LoRA) | Muy barato: solo se entrenan las cabezas sobre embeddings precalculados |

Se parece más a un **reranker o a un clasificador zero-shot** (tipo BART-MNLI o un bi-encoder) que a un chat, pero
entrenado específicamente para *decidir acciones* en lugar de para similitud semántica. La ablación `clm-raw`
(coseno directo en el espacio de Qwen3-8B, sin cabezas) que trae el paquete permite medir qué aportan las cabezas.

Casos de uso naturales: enrutado de tickets, selección de herramientas en agentes, verificador en best-of-N,
agentes en tiempo real (juegos, uso del ordenador) y clasificación con etiquetas descritas en lenguaje natural.

### Diferencias entre los modelos de la familia

| Modelo | Encoder | Qué es | Notas |
|---|---|---|---|
| [CLM-v0.1-8B](https://huggingface.co/Contrastive-LM/CLM-v0.1-8B) | Qwen3-8B | Checkpoint general (cabezas de 75 MB) | El de referencia; zero-shot. Contexto de 2048 tokens por defecto. Solo inglés |
| [deepswe-clm-heads-8k](https://huggingface.co/Contrastive-LM/deepswe-clm-heads-8k) | Qwen3-8B | Cabezas ajustadas a partir del anterior como **verificador de trayectorias de código** | Es el que da el 81,6 % de DeepSWE; especializado, no generalista |
| `clm-raw` (en el paquete) | Qwen3-8B | Ablación sin cabezas | Línea base para comparar |
| [CLM-Qwen3.5-2B](https://huggingface.co/jayavibhav/clm-qwen3.5-2b) (comunidad) | Qwen3.5-2B | Misma receta sobre un encoder 4× más pequeño | 131K de contexto; lee el estado en la "posición de respuesta" (formato chat). Orientado a clasificación de documentos. No es oficial |
| CLM-35B (anunciado) | — | Multimodal, más datos y parámetros | Anunciado para principios de octubre de 2026; aún no publicado |

### ¿Son buenos?

Lo que reportan los autores (sin verificación independiente todavía):

- **Zero-shot**: rendimiento similar a Jev (el modelo de decisión de TypeSafe) en uso del ordenador, juegos
  (T-Rex, Super Mario, WikiRacing) y llamadas a herramientas (BFCL v4), con **hasta 9× menos latencia**.
  Con ~1.000 candidatos, 13× más rápido gracias a la caché de acciones.
- **Como verificador ajustado**: SOTA en **DeepSWE (81,6 %)** y **Terminal-Bench 2.1 (87,6 %)**, 4–6× más rápido que Jev.

Matices importantes:

- Los números SOTA son de **cabezas ajustadas a cada tarea**, no del checkpoint general en zero-shot.
- Las muestras son pequeñas: el 81,6 % de DeepSWE son **31/38 tareas**, frente a un pass@1 de 28/38 (73,7 %)
  y un oráculo de 34/38 (89,5 %). Es decir, el verificador acierta 3 tareas más que coger la primera solución.
  Terminal-Bench usa 30 tareas.
- Está **atado a Qwen3-8B**: las cabezas solo funcionan con los embeddings de ese modelo exacto (último token),
  así que necesitas servir un 8B (~16 GB de VRAM en bf16) aunque las cabezas pesen 75 MB.
- Entrenado en **inglés**; en español hay que probarlo (el notebook incluye un ejemplo).
- Las probabilidades son **relativas al conjunto** de candidatos: si la respuesta correcta no está, elegirá la menos mala.
- Es la v0.1: muy prometedor para decisiones rápidas entre opciones cerradas, pero conviene validarlo en tu propio caso.

**Conclusión:** para *elegir entre opciones* (clasificar, enrutar, verificar, escoger herramienta) es una alternativa
mucho más rápida y barata que preguntar a un LLM, y el ajuste fino es trivial. Para cualquier cosa que requiera
generar, explicar o razonar en varios pasos, sigue haciendo falta un LLM.

### Cómo lanzarlo

**Google Colab Pro (recomendado):** abre [`notebooks/clm_v0_1_8b_colab.ipynb`](notebooks/clm_v0_1_8b_colab.ipynb)
con GPU **L4 o A100**. Una T4 (16 GB) no basta para Qwen3-8B en bf16. El notebook:

1. Instala `contrastive-lm` y arranca Qwen3-8B como servidor de embeddings con vLLM.
2. Hace preguntas tipadas (`noul` / `choice` / `score`), en inglés y en español.
3. Ordena candidatos libres (selección de herramienta y best-of-N de código).
4. Ejecuta un mini-benchmark zero-shot en AG News (400 ejemplos) comparando **CLM**, **`clm-raw`** y **BART-large-MNLI**
   en precisión y latencia.
5. Mide el efecto de la caché de acciones con 1.000 candidatos.
6. Opcionalmente, abre el playground web de `clm-serve`.

**En local** hace falta una GPU con ≥24 GB:

```bash
pip install contrastive-lm
vllm serve Qwen/Qwen3-8B --served-model-name qwen3-8b --runner pooling --max-model-len 2048 --port 8090 &
clm-serve   # API + playground en http://localhost:8700/
```

En este equipo (WSL con 3,7 GB de RAM y sin GPU) no es viable con vLLM. La alternativa sería servir un GGUF de
Qwen3-8B con `llama-server --embeddings --pooling last`, pero sería lento y los embeddings no serían idénticos
a aquellos con los que se entrenaron las cabezas.

### Referencias

- Ficha del modelo: https://huggingface.co/Contrastive-LM/CLM-v0.1-8B
- Código y API: https://github.com/Contrastive-LM/CLM
- Blog: https://contrastive-lm.notion.site
- Anuncio de Azalia Mirhoseini: https://x.com/Azaliamirh/status/2102958024323428362
- Anuncio de Jacky Kwok: https://x.com/jackyk02/status/2102905335925424285
- Explicación divulgativa: https://blog.dailydoseofds.com/p/contrastive-language-model-clearly
