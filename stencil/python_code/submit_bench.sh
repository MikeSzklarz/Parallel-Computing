#!/bin/bash
#SBATCH --job-name="bench_stencil"
#SBATCH --output="bench_stencil.%j.%N.out"
#SBATCH --partition=compute
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --mem=0
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 00:30:00

module purge
module load cpu/0.17.3b
module load gcc/10.2.0
module load openmpi/4.1.1
module load anaconda3/2021.05
module load slurm

conda activate stencil

export SCRATCH_DIR=/scratch/$USER/job_$SLURM_JOB_ID
mkdir -p "$SCRATCH_DIR"

# Copy the submission directory (SLURM_SUBMIT_DIR) to node-local scratch
rsync -av --exclude='*.o' --exclude='results' "$SLURM_SUBMIT_DIR/" "$SCRATCH_DIR/workdir/"

cd "$SCRATCH_DIR/workdir/stencil/python_code"

# Example invocation: adjust arguments as needed. Use --use_scratch so the
# benchmark writes outputs to node-local scratch and then copies them back.
# The --base_results should point to a directory on the shared filesystem
# (for example a 'results' directory under the submission directory).
python3 bench_stencil2d_pthreads.py --use_scratch --remove_scratch \
    --base_results "$SLURM_SUBMIT_DIR/results" \
    --N1 1024 --N2 8192 --num_Ns 8 --P_start 1 --P_step 16 --P_max 128 --I1 100 --I2 500 --Istep 100 \
    --warmup 1 --trials 4 --timeout_sec 600

# After the job finishes, results will be copied back to $SLURM_SUBMIT_DIR/results/<exp_name>
