#!/bin/bash
#SBATCH -p amd_256      
#SBATCH -N 1                 
#SBATCH --ntasks-per-node=64   
#SBATCH -n 64                   
#SBATCH --cpus-per-task=1     
#SBATCH -o logs/slurm-%j.out    
#SBATCH -e logs/slurm-%j.err    

# submit job
# 1. cd /public3/home/sc72581/zzk/VMC_pennylane
# 2. sbatch submit_Heisenberg.sh

# load env
source ~/.bashrc
conda activate zzk

# mpi
source /public3/soft/modules/module.sh
# module avail mpi 
module load mpi/openmpi/3.1.4-gcc-10.2.0-public3  # best


# run
mpirun -n 64 python VMC_Heisenberg.py

# 10 qubit: -N 2  --ntasks-per-node=64  -n 128  --cpus-per-task=1 
# 10 qubit (parameter-shift): -N 10  --ntasks-per-node=64  -n 640  --cpus-per-task=1 

# 12 qubit: -N 4  --ntasks-per-node=64  -n 256  --cpus-per-task=1 
# 12 qubit (parameter-shift): -N 15  --ntasks-per-node=64  -n 960  --cpus-per-task=1 

