#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=48:00:00
#SBATCH --partition=main
#SBATCH --account=aoberai_286

module purge
module load gcc/11.3.0
module load cuda/11.8.0
eval "$(conda shell.bash hook)"
conda activate pytorch_env

python main.py \
        --idx=1 \
        --nsamples=10000 \
