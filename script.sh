#!/bin/bash
#
#SBATCH --job-name=PQC_multivariate_ising
#SBATCH --comment="Training verschiedener Quantum ML Architekturen für meine BA"
#SBATCH --mail-type=ALL
#SBATCH --mail-user=
#SBATCH --chdir=/home/qnn_design_for_supervised_learning
#SBATCH --output=/home/qnn_design_for_supervised_learning/slurm.%j.%N.out
#SBATCH --qos=abaki
#SBATCH --ntasks=1
#SBATCH --no-requeue

set -euo pipefail
mkdir -p logs

# Writable tmp (nodes complain otherwise)
export TMPDIR="${SLURM_TMPDIR:-$PWD/.tmp}"
mkdir -p "$TMPDIR"

# point to your system CUDA (12.8 per your node)
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

VENV_DIR="$PWD/.venv"
PY="$VENV_DIR/bin/python"

# create venv (exact Debian way)
if [[ ! -x "$PY" ]]; then
  echo "[setup] creating venv at $VENV_DIR"
  python3 -m venv "$VENV_DIR"          # requires python3-venv (python3-full)
fi

# use the venv's python/pip for installs
#"$PY" -m pip install --upgrade pip setuptools wheel
#"$PY" -m pip --version

# install the CUDA-bundled JAX (includes cuDNN)
# "$PY" -m pip install -U "jax[cuda12_pip]==0.4.33" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
# Install all deps into THIS venv
# "$PY" -m pip install -r requirements.txt

# Log driver/GPU
# nvidia-smi || true

# Sanity: confirm correct CUDA toolchain picked up
which ptxas || true
ptxas --version || true
echo "CUDA_HOME=$CUDA_HOME"

# Sanity checks: show where Python is and that key packages import


# Headless plotting
export MPLBACKEND=Agg

# Run training with the venv python
exec "$PY" -u main.py --num_input_features 2 --group exponential --seed "$1"