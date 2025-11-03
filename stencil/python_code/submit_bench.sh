#!/bin/bash
#SBATCH --job-name="bench_stencil"
#SBATCH --output="bench_stencil.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --exclusive
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 00:30:00

set -euo pipefail

module purge
module load cpu/0.17.3b
module load gcc/10.2.0
module load openmpi/4.1.1
module load anaconda3/2021.05
module load slurm

# --- Conda initialization & activation ---------------------------------
# Replace with your conda env name if different
CONDA_ENV="stencil"

echo "Initializing conda for shell..."
if ! eval "$(conda shell.bash hook)" >/dev/null 2>&1; then
    # Fallback: try a common conda.sh location
    if [ -f "/cm/shared/apps/anaconda3/etc/profile.d/conda.sh" ]; then
        . "/cm/shared/apps/anaconda3/etc/profile.d/conda.sh"
    elif [ -f "${HOME}/anaconda3/etc/profile.d/conda.sh" ]; then
        . "${HOME}/anaconda3/etc/profile.d/conda.sh"
    fi
fi

echo "Activating conda env: ${CONDA_ENV}"
if ! conda activate "${CONDA_ENV}" >/dev/null 2>&1; then
    echo "ERROR: failed to activate conda env '${CONDA_ENV}'." >&2
    echo "Which conda: $(which conda 2>/dev/null || echo 'none')" >&2
    conda --version || true
    exit 1
fi

echo "Conda activated: $(which python) -- $(python --version 2>&1)"

# Helpful exports for threaded runs
export SLURM_CPUS_PER_TASK=${SLURM_CPUS_PER_TASK:-128}
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# --- Scratch setup and rsync --------------------------------------------
export SCRATCH_DIR=/scratch/$USER/job_$SLURM_JOB_ID
echo "Using scratch dir: ${SCRATCH_DIR}"
mkdir -p "$SCRATCH_DIR"

echo "Rsync from SLURM_SUBMIT_DIR='$SLURM_SUBMIT_DIR' to '$SCRATCH_DIR/workdir/'"
rsync -av --exclude='*.o' --exclude='results' "$SLURM_SUBMIT_DIR/" "$SCRATCH_DIR/workdir/"

# --- Locate the benchmark script and cd there ---------------------------
TARGET_SCRIPT="bench_stencil2d_pthreads.py"
if [ -f "$SCRATCH_DIR/workdir/stencil/python_code/${TARGET_SCRIPT}" ]; then
    CD_DIR="$SCRATCH_DIR/workdir/stencil/python_code"
elif [ -f "$SCRATCH_DIR/workdir/${TARGET_SCRIPT}" ]; then
    CD_DIR="$SCRATCH_DIR/workdir"
else
    FOUND=$(find "$SCRATCH_DIR/workdir" -name "${TARGET_SCRIPT}" -print -quit || true)
    if [ -n "$FOUND" ]; then
        CD_DIR=$(dirname "$FOUND")
    else
        echo "ERROR: cannot find ${TARGET_SCRIPT} under $SCRATCH_DIR/workdir" >&2
        ls -al "$SCRATCH_DIR/workdir" || true
        exit 1
    fi
fi

echo "Changing directory to: ${CD_DIR}"
cd "${CD_DIR}" || { echo "cd failed to ${CD_DIR}"; exit 1; }
echo "Directory contents:"; ls -al

# --- Run the benchmark under srun so Slurm binds CPUs -------------------
echo "Launching benchmark via srun ..."
srun --cpu_bind=cores -n1 python3 "${TARGET_SCRIPT}" --use_scratch --remove_scratch \
    --base_results "$SLURM_SUBMIT_DIR/results" \
    --N1 1024 --N2 8192 --num_Ns 8 --P_start 1 --P_step 16 --P_max 128 --I1 100 --I2 500 --Istep 100 \
    --warmup 1 --trials 4 --timeout_sec 600

echo "Job finished. Results (if any) will be copied back by the Python script to $SLURM_SUBMIT_DIR/results/<exp_name>"
