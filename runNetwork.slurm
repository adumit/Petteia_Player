#!/bin/bash
#SBATCH --partition=commons
#SBATCH --account=commons
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=05:00:00

module purge
module load GCC/5.4.0 OpenMPI/1.10.3 CUDA/7.5.18 TensorFlow/0.10.0

srun python NNetwork.py
