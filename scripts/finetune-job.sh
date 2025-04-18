#!/bin/sh
# COMMENT: #SBATCH directives that convey submission options:
#SBATCH --job-name=finetune_qwen
#SBATCH --mail-type=BEGIN,END,FAILED
#SBATCH --parsable
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-cpu=32GB
#SBATCH --time=8:00:00
#SBATCH --account=eecs545w25_class
#SBATCH --partition=spgpu,gpu
#SBATCH --output=/home/%u/logs/%x-%j.log
# COMMENT:The application(s) to execute along with its input arguments and options:

eval "$(conda shell.bash hook)"
conda activate qwen2
ml cuda/12
ml gcc/13.2.0
export LD_PRELOAD=/sw/pkgs/arc/gcc/13.2.0/lib64/libstdc++.so.6
export PYTHONPATH=$PYTHONPATH:/home/naveenu/Qwen2-VL-Finetune/src
bash $HOME/Qwen2-VL-Finetune/scripts/finetune_highlights.sh
