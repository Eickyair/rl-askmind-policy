# AskMind — Política conversacional `ask` / `respond` (Baselines MLP)

Proyecto final de **Aprendizaje por Refuerzo**. Entrena baselines tipo **MLP** para
una política conversacional binaria que decide, en cada turno, si el sistema debe:

- **`ask`** — pedir una aclaración adicional al usuario, o
- **`respond`** — responder con el contexto disponible.

El proyecto se construye sobre la dimensión **AskMind** del benchmark
[**AskBench**](https://arxiv.org/abs/2602.11199) (*When and What to Ask: AskBench and
Rubric-Guided RLVR for LLM Clarification*). A partir de preguntas "degradadas"
(con información faltante o ambigua) y sus *required points*, se modela el problema
como un MDP de decisión por turno y se entrenan baselines de **policy gradient** y
**Q-learning** sobre embeddings TF-IDF + SVD del estado conversacional.

---

## 1. Contenido del repositorio

| Archivo | Descripción |
| --- | --- |
| [`askmind_mlp_baselines.ipynb`](askmind_mlp_baselines.ipynb) | Notebook principal **autocontenido**: descarga el dataset, construye el dataset tabular, formaliza el MDP, entrena y evalúa los baselines y reporta la tabla final en test. |
| [`requirements.txt`](requirements.txt) | Dependencias con versiones exactas. |
| [`README.md`](README.md) | Esta guía. |
| `.gitignore` | Excluye datos, artefactos y archivos no relevantes. |

### Qué produce el notebook (entregables)

1. **Formulación formal del problema como MDP** (sección en inglés, lista para el paper):
   estado, espacio de acciones `ASK/ANSWER`, dinámica de transición y función de recompensa.
2. **Split de 3 vías** train/validation/test (60/20/20) **agrupado por `ori_question`**, para
   reportar métricas en un **test held-out con etiquetas** (el `test.jsonl` oficial no las trae).
3. **Seis sistemas comparables**: `Always ASK`, `Always ANSWER`, `Random`, `Supervised MLP`,
   `MLP Policy Gradient` y `MLP Q-learning`.
4. **Tabla final de sistemas** en validation y test con `Accuracy`, `Macro F1`, `Ask rate` y
   `Avg reward`.
5. **Ablación OFAT**: efecto del costo de preguntar (bajo/medio/alto) sobre `ask_rate` y `reward`.
6. **Error analysis** con 5 ejemplos (respondió antes de aclarar / preguntó de más / correcto).

> **Nota:** el dataset (`askmind_data/`) **no** se versiona en git. El notebook lo
> **descarga automáticamente** desde Hugging Face la primera vez que se ejecuta
> (ver §5). Así el repositorio queda ligero y reproducible.

---

## 2. Requisitos

- **Python 3.12** (probado con 3.12.3). Versiones 3.10–3.12 deberían funcionar.
- **pip ≥ 23** y `venv` (incluido en Python).
- ~2 GB de espacio (la rueda de PyTorch es la dependencia más pesada).
- Conexión a internet en la **primera** ejecución del notebook (descarga ~17 MB de datos).
- GPU NVIDIA **opcional** (acelera el entrenamiento; no es necesaria).

Versiones exactas (ver [`requirements.txt`](requirements.txt)):

```text
numpy==2.4.5        pandas==3.0.3        scipy==1.17.1
scikit-learn==1.8.0 torch==2.12.0        ipython==9.13.0     ipykernel==7.2.0
```

---

## 3. Instalación por sistema operativo

El flujo es el mismo en todos los SO: **(1)** crear un entorno virtual, **(2)** instalar
PyTorch con la rueda adecuada (CPU o GPU), **(3)** instalar el resto con `requirements.txt`.

### 🐧 Linux

```bash
# 1) Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) PyTorch
#    a) CPU (recomendado si no tienes GPU NVIDIA):
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
#    b) GPU NVIDIA con CUDA 13.0 (alternativa):
# pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

# 3) Resto de dependencias
pip install -r requirements.txt
```

### 🍎 macOS (Intel y Apple Silicon)

En macOS no hay CUDA; la rueda de PyPI ya es CPU/MPS, así que basta:

```bash
# 1) Entorno virtual
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) + 3) Todo de una vez (torch viene de PyPI, soporta MPS en Apple Silicon)
pip install -r requirements.txt
```

> En Apple Silicon, PyTorch usa el backend **MPS** automáticamente si está disponible.
> El notebook detecta el dispositivo (`cuda`/`cpu`) por sí solo; para forzar MPS, edita
> `BaselineConfig.device` a `"mps"`.

### 🪟 Windows

**PowerShell:**

```powershell
# 1) Entorno virtual
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) PyTorch (CPU)
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
#    GPU NVIDIA (CUDA 13.0):
# pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

# 3) Resto
pip install -r requirements.txt
```

**CMD (`cmd.exe`):** idéntico, pero activa con:

```bat
.\.venv\Scripts\activate.bat
```

> Si `Activate.ps1` falla por política de ejecución, abre PowerShell como usuario y ejecuta:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### 🐍 (Opcional) Conda — cualquier SO

```bash
conda create -n askmind python=3.12 -y
conda activate askmind
pip install -r requirements.txt   # o instala torch con su índice como arriba
```

---

## 4. Registrar el kernel de Jupyter

Para que el notebook use exactamente este entorno:

```bash
python -m ipykernel install --user --name askmind --display-name "AskMind (.venv 3.12)"
```

En **VS Code** basta con seleccionar el intérprete `.venv` en la esquina superior
derecha del notebook; no hace falta registrar el kernel manualmente.

---

## 5. Dataset: descarga automática (notebook autocontenido)

No necesitas descargar nada a mano. La celda **"0. Descarga automática del dataset"**
del notebook ejecuta `ensure_askmind_dataset(...)`, que:

1. Comprueba si ya existen `askmind_data/train.jsonl` y `askmind_data/test.jsonl`.
   Si están presentes, **omite** la descarga.
2. Si faltan, descarga los archivos crudos de AskBench desde Hugging Face:
   - **train** → [`jialeuuz/askbench_train` → `mind.jsonl`](https://huggingface.co/datasets/jialeuuz/askbench_train)
   - **test** → [`jialeuuz/askbench_bench` → `ask_bench_data/ask_mind.jsonl`](https://huggingface.co/datasets/jialeuuz/askbench_bench)
3. Aplica el **mismo preprocesamiento** del proyecto: elimina filas con caracteres
   **Han/CJK**, descarta líneas vacías o inválidas y normaliza cada registro como una
   línea JSON.
4. Escribe `askmind_data/train.jsonl` (**5830** ejemplos), `askmind_data/test.jsonl`
   (**399** ejemplos) y un `MANIFEST.csv`.

> El conteo y el contenido resultantes son **idénticos** al dataset original del proyecto
> (verificado registro a registro). Solo usa `stdlib` (`urllib`), sin dependencias extra.

Para **forzar** una redescarga, en esa celda llama:
`ensure_askmind_dataset(config.data_dir, force=True)`.

### Campos del dataset

- `degraded_question`: pregunta incompleta que ve el modelo (entrada de la política).
- `degraded_info`: descripción de la información removida o ambigua.
- `required_points`: checkpoints que el modelo debe cubrir preguntando.
- `conversation_history`: trayectoria multi-turno (usuario/asistente).
- `ori_question` / `expected_answer`: referencia para evaluación (no son entrada de la política).

---

## 6. Cómo ejecutar el proyecto

### Opción A — VS Code (recomendada)

1. Abre la carpeta del proyecto en VS Code.
2. Abre [`askmind_mlp_baselines.ipynb`](askmind_mlp_baselines.ipynb).
3. Selecciona el intérprete/kernel `.venv` (3.12).
4. **Run All**. La primera ejecución descargará el dataset automáticamente.

### Opción B — JupyterLab / Jupyter Notebook

```bash
pip install jupyterlab          # si aún no lo tienes
jupyter lab                     # o: jupyter notebook
# abre askmind_mlp_baselines.ipynb y ejecuta todas las celdas
```

### Opción C — ejecución headless (sin abrir la UI)

Ejecuta el notebook de principio a fin desde la terminal:

```bash
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace askmind_mlp_baselines.ipynb
```

> Esto corre todas las celdas (incluida la descarga del dataset) y guarda las salidas
> en el propio `.ipynb`. Útil para CI o para verificar reproducibilidad.

---

## 7. Configuración de hiperparámetros

Todos los parámetros viven en la dataclass `BaselineConfig` del notebook (celda de
*imports/config*). Los más relevantes:

| Parámetro | Valor por defecto | Significado |
| --- | --- | --- |
| `data_dir` | `askmind_data` | Carpeta del dataset descargado. |
| `validation_fraction` | `0.20` | Fracción de grupos reservada a validación (split por `ori_question`). |
| `test_fraction` | `0.20` | Fracción de grupos reservada al test held-out (split por `ori_question`). |
| `ask_cost_levels` | `(low 0.0, medium 0.3, high 0.6)` | Niveles de costo de preguntar para la ablación OFAT. |
| `random_seed` | `42` | Semilla global (numpy + torch). |
| `tfidf_max_features` | `4096` | Vocabulario máximo del TF-IDF. |
| `embedding_dim` | `256` | Dimensión objetivo del SVD sobre TF-IDF. |
| `batch_size` | `128` | Tamaño de lote. |
| `hidden_dims` | `(256, 128)` | Capas ocultas del MLP. |
| `epochs` | `5` | Épocas de entrenamiento. |
| `device` | auto (`cuda`/`cpu`) | Dispositivo de cómputo. |
| `precomputed_embeddings` | `None` | Ruta opcional a un `.npz` para reemplazar TF-IDF+SVD. |

---

## 8. Reproducibilidad

- Semilla fija (`random_seed=42`) sobre numpy y torch.
- El split train/validación/**test** se hace **agrupando por `ori_question`** para evitar fuga
  entre turnos y variantes de una misma conversación; así el test held-out es comparable a
  validación y conserva etiquetas de acción.
- El preprocesamiento del dataset es determinista (filtro Han + normalización JSON).

---

## 9. Solución de problemas

| Síntoma | Causa / solución |
| --- | --- |
| `ModuleNotFoundError: torch` | No instalaste PyTorch o activaste otro entorno. Reactiva `.venv` e instala con el índice correcto (§3). |
| Descarga de torch enorme / lenta en CPU | Usa el índice CPU: `--index-url https://download.pytorch.org/whl/cpu`. |
| `URLError` / timeout al descargar el dataset | Revisa tu conexión; reintenta la celda. Si tienes los `.jsonl`, colócalos en `askmind_data/` y la descarga se omite. |
| El notebook no encuentra el kernel | Registra el kernel (§4) o selecciona el intérprete `.venv` en VS Code. |
| `Activate.ps1 cannot be loaded` (Windows) | Ajusta la política de ejecución (ver nota en §3 → Windows). |

---

## 10. Referencias

- **Paper:** Zhao, Fang, Cheng. *When and What to Ask: AskBench and Rubric-Guided RLVR
  for LLM Clarification.* arXiv:[2602.11199](https://arxiv.org/abs/2602.11199) (2026).
- **Dataset (train):** [`jialeuuz/askbench_train`](https://huggingface.co/datasets/jialeuuz/askbench_train)
- **Dataset (benchmark):** [`jialeuuz/askbench_bench`](https://huggingface.co/datasets/jialeuuz/askbench_bench)
