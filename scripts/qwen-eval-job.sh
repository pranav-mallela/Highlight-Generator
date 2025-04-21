#!/bin/sh
# COMMENT: #SBATCH directives that convey submission options:
#SBATCH --job-name=eval_qwen
#SBATCH --mail-type=BEGIN,END,FAILED
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=8:00:00
#SBATCH --account=eecs545w25_class
#SBATCH --partition=spgpu
#SBATCH --output=/home/%u/logs/%x-%j.log
# COMMENT:The application(s) to execute along with its input arguments and options:

eval "$(conda shell.bash hook)"
conda activate qwen2
ml cuda/12
ml gcc/13.2.0
export LD_PRELOAD=/sw/pkgs/arc/gcc/13.2.0/lib64/libstdc++.so.6
export HF_HOME=/scratch/eecs545w25_class_root/eecs545w25_class/highlights/cache

python /home/naveenu/545-project/evaluation/qwen_eval.py