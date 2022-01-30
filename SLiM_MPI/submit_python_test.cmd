#!/bin/bash
#SBATCH -p debug
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=30
#SBATCH -J SLiM
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err

module load python
srun -n 32 -c 2 python Dispersion_list_MPI.py