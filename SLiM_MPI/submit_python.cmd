#!/bin/bash
#SBATCH -p regular
#SBATCH --constraint=haswell
#SBATCH --nodes=80
#SBATCH --time=1200
#SBATCH -J SLiM
#SBATCH -o ./%x.%j.out
#SBATCH -e ./%x.%j.err

module load python
srun -n 2560 -c 2 python MTMDispersion_list_Calc_MPI.py