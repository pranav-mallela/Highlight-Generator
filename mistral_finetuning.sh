#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccerhighlightsmistral
#SBATCH --account=eecs568s001w25_class
#SBATCH --partition=spgpu,gpu_mig40
#SBATCH --gpus=1
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=4g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=out.txt

eval "$(conda shell.bash hook)"
conda activate llm-ft

python3 mistral_finetuning.py