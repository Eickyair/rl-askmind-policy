# AskMind — Conversational `ask` / `respond` policy (MLP baselines)

Final project for **Reinforcement Learning**. It trains **MLP** baselines for a binary
conversational policy that decides, at each turn, whether the system should:

- **`ask`** — request an additional clarification from the user, or
- **`respond`** — answer with the available context.

The project builds on the **AskMind** dimension of the
[**AskBench**](https://arxiv.org/abs/2602.11199) benchmark (*When and What to Ask: AskBench and
Rubric-Guided RLVR for LLM Clarification*). Starting from "degraded" questions
(with missing or ambiguous information) and their *required points*, the problem is modeled
as a per-turn decision MDP, and **policy gradient** and **Q-learning** baselines are trained
over TF-IDF + SVD embeddings of the conversational state.

---

## 1. Repository contents

| File | Description |
| --- | --- |
| [`askmind_mlp_baselines.ipynb`](askmind_mlp_baselines.ipynb) | Main **self-contained** notebook: downloads the dataset, builds the tabular dataset, formalizes the MDP, trains and evaluates the baselines, and reports the final test table. |
| [`requirements.txt`](requirements.txt) | Dependencies with exact versions. |
| [`README.md`](README.md) | This guide. |
| `.gitignore` | Excludes data, artifacts and non-relevant files. |

### What the notebook produces (deliverables)

1. **Formal MDP problem statement** (paper-ready section, in English): state, `ASK/ANSWER`
   action space, transition dynamics and reward function.
2. **Three-way split** train/validation/test (60/20/20) **grouped by `ori_question`**, to
   report metrics on a **held-out test with labels** (the official `test.jsonl` has none).
3. **Six comparable systems**: `Always ASK`, `Always ANSWER`, `Random`, `Supervised MLP`,
   `MLP Policy Gradient` and `MLP Q-learning`.
4. **Final systems table** on validation and test with `Accuracy`, `Macro F1`, `Ask rate` and
   `Avg reward`.
5. **OFAT ablation**: effect of the cost of asking (low/medium/high) on `ask_rate` and `reward`.
6. **Error analysis** with 5 examples (answered before clarifying / asked too much / correct).

> **Note:** the dataset (`askmind_data/`) is **not** versioned in git. The notebook
> **downloads it automatically** from Hugging Face the first time it runs
> (see §5). This keeps the repository light and reproducible.

---

## 2. Requirements

- **Python 3.12** (tested with 3.12.3). Versions 3.10–3.12 should work.
- **pip ≥ 23** and `venv` (included with Python).
- ~2 GB of disk space (the PyTorch wheel is the heaviest dependency).
- Internet access on the **first** notebook run (downloads ~17 MB of data).
- NVIDIA GPU **optional** (speeds up training; not required).

Exact versions (see [`requirements.txt`](requirements.txt)):

```text
numpy==2.4.5        pandas==3.0.3        scipy==1.17.1
scikit-learn==1.8.0 torch==2.12.0        ipython==9.13.0     ipykernel==7.2.0
```

---

## 3. Installation per operating system

The flow is the same on every OS: **(1)** create a virtual environment, **(2)** install
PyTorch with the right wheel (CPU or GPU), **(3)** install the rest with `requirements.txt`.

### Linux

```bash
# 1) Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) PyTorch
#    a) CPU (recommended if you don't have an NVIDIA GPU):
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
#    b) NVIDIA GPU with CUDA 13.0 (alternative):
# pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

# 3) Remaining dependencies
pip install -r requirements.txt
```

### macOS (Intel and Apple Silicon)

There is no CUDA on macOS; the PyPI wheel is already CPU/MPS, so this is enough:

```bash
# 1) Virtual environment
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip

# 2) + 3) All at once (torch comes from PyPI, supports MPS on Apple Silicon)
pip install -r requirements.txt
```

> On Apple Silicon, PyTorch uses the **MPS** backend automatically if available.
> The notebook detects the device (`cuda`/`cpu`) by itself; to force MPS, edit
> `BaselineConfig.device` to `"mps"`.

### Windows

**PowerShell:**

```powershell
# 1) Virtual environment
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip

# 2) PyTorch (CPU)
pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cpu
#    NVIDIA GPU (CUDA 13.0):
# pip install torch==2.12.0 --index-url https://download.pytorch.org/whl/cu130

# 3) Rest
pip install -r requirements.txt
```

**CMD (`cmd.exe`):** identical, but activate with:

```bat
.\.venv\Scripts\activate.bat
```

> If `Activate.ps1` fails due to the execution policy, open PowerShell as your user and run:
> `Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned`.

### (Optional) Conda — any OS

```bash
conda create -n askmind python=3.12 -y
conda activate askmind
pip install -r requirements.txt   # or install torch with its index as above
```

---

## 4. Register the Jupyter kernel

So the notebook uses exactly this environment:

```bash
python -m ipykernel install --user --name askmind --display-name "AskMind (.venv 3.12)"
```

In **VS Code** it is enough to select the `.venv` interpreter in the top-right corner of
the notebook; you do not need to register the kernel manually.

---

## 5. Dataset: automatic download (self-contained notebook)

You don't need to download anything by hand. The **"0. Automatic dataset download"** cell
of the notebook runs `ensure_askmind_dataset(...)`, which:

1. Checks whether `askmind_data/train.jsonl` and `askmind_data/test.jsonl` already exist.
   If present, it **skips** the download.
2. If missing, it downloads the raw AskBench files from Hugging Face:
   - **train** → [`jialeuuz/askbench_train` → `mind.jsonl`](https://huggingface.co/datasets/jialeuuz/askbench_train)
   - **test** → [`jialeuuz/askbench_bench` → `ask_bench_data/ask_mind.jsonl`](https://huggingface.co/datasets/jialeuuz/askbench_bench)
3. Applies the **same preprocessing** as the project: removes rows with
   **Han/CJK** characters, drops empty or invalid lines and normalizes each record as a
   single JSON line.
4. Writes `askmind_data/train.jsonl` (**5830** examples), `askmind_data/test.jsonl`
   (**399** examples) and a `MANIFEST.csv`.

> The resulting count and content are **identical** to the project's original dataset
> (verified record by record). It only uses `stdlib` (`urllib`), with no extra dependencies.

To **force** a re-download, in that cell call:
`ensure_askmind_dataset(config.data_dir, force=True)`.

### Dataset fields

- `degraded_question`: incomplete question seen by the model (policy input).
- `degraded_info`: description of the removed or ambiguous information.
- `required_points`: checkpoints the model should cover by asking.
- `conversation_history`: multi-turn trajectory (user/assistant).
- `ori_question` / `expected_answer`: evaluation reference (not policy inputs).

---

## 6. How to run the project

### Option A — VS Code (recommended)

1. Open the project folder in VS Code.
2. Open [`askmind_mlp_baselines.ipynb`](askmind_mlp_baselines.ipynb).
3. Select the `.venv` interpreter/kernel (3.12).
4. **Run All**. The first run will download the dataset automatically.

### Option B — JupyterLab / Jupyter Notebook

```bash
pip install jupyterlab          # if you don't have it yet
jupyter lab                     # or: jupyter notebook
# open askmind_mlp_baselines.ipynb and run all cells
```

### Option C — headless execution (without opening the UI)

Run the notebook end to end from the terminal:

```bash
pip install jupyter
jupyter nbconvert --to notebook --execute --inplace askmind_mlp_baselines.ipynb
```

> This runs all cells (including the dataset download) and stores the outputs
> in the `.ipynb` itself. Useful for CI or to verify reproducibility.

---

## 7. Hyperparameter configuration

All parameters live in the `BaselineConfig` dataclass of the notebook (the
*imports/config* cell). The most relevant ones:

| Parameter | Default value | Meaning |
| --- | --- | --- |
| `data_dir` | `askmind_data` | Folder of the downloaded dataset. |
| `validation_fraction` | `0.20` | Fraction of groups reserved for validation (split by `ori_question`). |
| `test_fraction` | `0.20` | Fraction of groups reserved for the held-out test (split by `ori_question`). |
| `ask_cost_levels` | `(low 0.0, medium 0.3, high 0.6)` | Cost-of-asking levels for the OFAT ablation. |
| `random_seed` | `42` | Global seed (numpy + torch). |
| `tfidf_max_features` | `4096` | Maximum TF-IDF vocabulary. |
| `embedding_dim` | `256` | Target SVD dimension over TF-IDF. |
| `batch_size` | `128` | Batch size. |
| `hidden_dims` | `(256, 128)` | MLP hidden layers. |
| `epochs` | `5` | Training epochs. |
| `device` | auto (`cuda`/`cpu`) | Compute device. |
| `precomputed_embeddings` | `None` | Optional path to a `.npz` to replace TF-IDF+SVD. |

---

## 8. Reproducibility

- Fixed seed (`random_seed=42`) over numpy and torch, reset before each trained model.
- The train/validation/**test** split is done by **grouping on `ori_question`** to avoid
  leakage between turns and variants of the same conversation; this makes the held-out test
  comparable to validation and keeps action labels.
- The dataset preprocessing is deterministic (Han filter + JSON normalization).

---

## 9. Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| `ModuleNotFoundError: torch` | You didn't install PyTorch or activated another environment. Reactivate `.venv` and install with the correct index (§3). |
| Huge / slow torch download on CPU | Use the CPU index: `--index-url https://download.pytorch.org/whl/cpu`. |
| `URLError` / timeout while downloading the dataset | Check your connection; retry the cell. If you already have the `.jsonl` files, place them in `askmind_data/` and the download is skipped. |
| The notebook can't find the kernel | Register the kernel (§4) or select the `.venv` interpreter in VS Code. |
| `Activate.ps1 cannot be loaded` (Windows) | Adjust the execution policy (see the note in §3 → Windows). |

---

## 10. References

- **Paper:** Zhao, Fang, Cheng. *When and What to Ask: AskBench and Rubric-Guided RLVR
  for LLM Clarification.* arXiv:[2602.11199](https://arxiv.org/abs/2602.11199) (2026).
- **Dataset (train):** [`jialeuuz/askbench_train`](https://huggingface.co/datasets/jialeuuz/askbench_train)
- **Dataset (benchmark):** [`jialeuuz/askbench_bench`](https://huggingface.co/datasets/jialeuuz/askbench_bench)
