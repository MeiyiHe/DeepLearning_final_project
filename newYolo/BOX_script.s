#!/bin/bash
#
#SBATCH --job-name=yolo01
#SBATCH --output=yolo01.out
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --mem=16GB
#SBATCH --gres=gpu:p40:1
#SBATCH --mail-user=mh5275@nyu.edu
#SBATCH --mail-type=ALL


. ~/.bashrc
module load anaconda3/5.3.1

conda activate pytorch
conda install -n pytorch nb_conda_kernels

python trainYolo.py
