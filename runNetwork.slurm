#!/bin/bash
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:05:00

module purge
module load GCC/5.4.0 OpenMPI/1.10.3 TensorFlow/0.10.0 CUDA/7.5.18 Python/3.5.2

srun python3 NNetwork.py
