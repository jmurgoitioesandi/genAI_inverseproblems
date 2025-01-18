#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=1              # 1 process
#SBATCH --cpus-per-task=4       # 4 CPUs
#SBATCH --mem=128GB              # 32 GB of memory
#SBATCH --time=48:00:00         # 1 hour run time
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1        # 1 node with 1 K40 GPU
#SBATCH --account=aoberai_286   # Account to charge resources to

module purge
module load gcc/13.3.0
module load cuda/12.6.3
eval "$(conda shell.bash hook)"
conda activate pytorch_env

python main.py  --saving_dir=exps/011825 \
                --problem=HeatEqDoubleSource \
                --xtype=vector \
                --noiselvl=0.4