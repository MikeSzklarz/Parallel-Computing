#!/bin/bash
#SBATCH --job-name="mpi_sweep_debug"
#SBATCH --output="mpi_sweep_debug.%j.out"
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --mem=0
#SBATCH --account=ccu108
#SBATCH --export=ALL
#SBATCH -t 00:30:00

module purge
module load cpu/0.17.3b
module load gcc/10.2.0
module load openmpi/4.1.1
module load anaconda3/2021.05

conda activate sweep_trap

A=10
B=20
N1=100000000
N2=2000000000
N_INC=100000000
P1=1
P2=128
P_INC=1

python3 mpi_trap_sweep.py \
    $A $B $N1 $N2 $N_INC $P1 $P2 $P_INC \
    --mpirun "srun" \
    --exe ./mpi_trap_modified