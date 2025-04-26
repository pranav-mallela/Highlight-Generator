#!/bin/bash
# The interpreter used to execute the script

#“#SBATCH” directives that convey submission options:

#SBATCH --job-name=soccerhighlightsmistral
#SBATCH --account=eecs568s001w25_class
#SBATCH --partition=spgpu,gpu_mig40
#SBATCH --gpus=1
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32g
#SBATCH --mail-type=BEGIN,END
#SBATCH --output=out.txt

eval "$(conda shell.bash hook)"
conda activate rob535

python3 mistral_finetuning.py