#!/bin/sh
# COMMENT: #SBATCH directives that convey submission options:
#SBATCH --job-name=soccernet_download
#SBATCH --mail-type=BEGIN,END,FAILED
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=100m
#SBATCH --time=8:00:00
#SBATCH --account=eecs545w25_class
#SBATCH --partition=standard
#SBATCH --output=/home/%u/logs/%x-%j.log
# COMMENT:The application(s) to execute along with its input arguments and options:

$HOME/545-project/scripts/soccernet-download/venv/bin/python $HOME/545-project/scripts/soccernet-download/dataset.py
