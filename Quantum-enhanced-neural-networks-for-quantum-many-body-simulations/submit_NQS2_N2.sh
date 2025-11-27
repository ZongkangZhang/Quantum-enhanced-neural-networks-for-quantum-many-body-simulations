#!/bin/bash
#SBATCH -p amd_256       
#SBATCH -N 4                   
#SBATCH --ntasks-per-node=64   
#SBATCH -n 256              
#SBATCH --cpus-per-task=1       
#SBATCH -o logs/slurm-%j.out    
#SBATCH -e logs/slurm-%j.err    

# submit job：
# 1. cd /public3/home/sc72581/zzk/VMC_pennylane
# 2. sbatch submit_NQS2_N2.sh

# load env
source ~/.bashrc
conda activate zzk

# mpi
source /public3/soft/modules/module.sh
# module avail mpi 
module load mpi/openmpi/3.1.4-gcc-10.2.0-public3  # best

# run
mpirun -n 256 python NQS2_N2.py
