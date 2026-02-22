# QNN Design for Supervised Learning (Bachelor Thesis)

This repository contains the code and experiment setup I used for my bachelor's thesis on **Quantum Neural Network (QNN) / Parameterized Quantum Circuit (PQC) design** in a **supervised-learning** setting.

The work focuses on how different circuit design choices (encoding, entangling structure, depth, etc.) affect **trainability** and **function-approximation performance** on **synthetic regression targets**.

---

## Thesis Abstract
Parameterized Quantum Circuits (PQCs) are a practical approach for applying quantum computers to quantum machine learning (QML) tasks. 
The design of a QML model is crucial for enabling efficient training and achieving accurate predictions. Therefore, depending on the specific task, 
it is important to understand the properties and capabilities of PQCs in order to select an effective and efficient model. 
While a large body of research studies practical applications of PQCs, knowledge about principled and task-appropriate circuit design remains limited.

PQCs can be represented by partial Fourier series, where the data encoding determines the accessible frequencies 
and the coefficients are induced by the parameterized ansatz. Substantial work has investigated the encoding component of PQCs 
by analyzing the generated frequency spectra, whereas the parameterized component responsible for learning suitable coefficients is comparatively less explored.

The goal of this thesis is to systematically evaluate a variety of PQC ansätze to identify ansatz fragments that lead to improved performance. 
For comparison, the considered models are trained on a truncated Fourier series of degree 10 and a non-differentiable target function, 
after which they are evaluated with respect to function-fitting capability and convergence speed across different circuit depths. 

Across the tested setups, entangling layers using controlled rotation gates yielded the best results, while all-to-all connectivity was advantageous only in some configurations. 
Serial architectures performed markedly better in the 1D case, whereas with additional input features the performance trend shifted towards parallel architectures. 

---

## What this project does
- Trains and compares multiple QNN/PQC architectures for **1D** and **2D** regression tasks
- Uses **JAX + Optax** for training loops and **PennyLane** for differentiable quantum circuits
- Logs and saves:
  - predictions + **R² scores**
  - training/test losses
  - gradient statistics (via `GradientLogger`)
  - circuit drawings and fit plots

---

## Repository structure (high level)
- `main.py` – entry point (runs 1D or 2D experiments depending on flags)
- `model_training.py` / `model_training_2D.py` – training loops for 1D and 2D settings
- `models/` – circuit building blocks (encodings, entanglers, model builder)
- `target_functions/` – synthetic target functions (1D + 2D)
- `eval/` – notebooks for post-hoc evaluation and plotting
- `plots/` – generated figures (fits, R², circuit diagrams, etc.)
- `preds_and_r2/` – serialized experiment outputs (pickles)
- `trained_models/` – serialized trained weights / saved data splits (pickles)
- `IBM QC/`, `lrz_qc/` – optional notebooks for experiments on quantum hardware/backends
- `script.sh` – example SLURM job script

---

## Installation
Recommended: Python 3.10+.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

### Notes on JAX
Depending on your machine (CPU/GPU), you may want to install a specific JAX build. Examples (optional):

```bash
# CPU-only
pip install -U "jax[cpu]"

# CUDA (example; choose the version that matches your system)
pip install -U "jax[cuda12_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

---

## Quickstart (run experiments)

### 1D (univariate) experiment
```bash
python main.py --num_input_features 1 --group standard --seed 42
```

### 2D (multivariate) experiment
```bash
python main.py --num_input_features 2 --group exponential --seed 42
```

Parameters:
- `--num_input_features`: `1` or `2` input features
- `--group`: `standard` or `exponential` encoding
- `--seed`: random seed for train/test split and initialization

---

## Outputs / where results are saved
During training, artifacts are written into:
- `plots/`
  - circuit diagrams (via `qml.draw_mpl`)
  - fit plots with R²
- `preds_and_r2/`
  - predictions and R² (train/test)
  - gradients / loss curves
- `trained_models/`
  - trained weights and saved splits (pickled)

(See the exact subpaths inside `model_training.py` and `model_training_2D.py`.)

---

## Evaluation / visualization
Use the notebooks in `eval/` to load and visualize results:
- `eval/evaluation.ipynb`
- `eval/evaluation_2D.ipynb`
- `eval/evaluation_final.ipynb`

---

## Hardware / external backends (optional)
The folders `IBM QC/` and `lrz_qc/` contain notebooks related to running on external backends, such as the IQM Q-Exa at LRZ in Munich/Garching.

---

## Disclaimer
This is research/experimental code created in the context of a university thesis. It is provided as a portfolio snapshot and may require small adjustments depending on environment (Python/JAX versions, hardware, etc.).