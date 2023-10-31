#!/bin/bash
#SBATCH --account=def-selouani
#SBATCH --gres=gpu:1       # Request GPU "generic resources"
#SBATCH --mem=32G       # Memory proportional to GPUs: 32000 Cedar, 64000 Graham.
#SBATCH --time=0-24:00
#SBATCH --output=output/%N-%j.out

module load python/3.8 scipy-stack gcc/9.3.0 cuda/11.4 opencv
source ENV/bin/activate


python ssl_training.py
