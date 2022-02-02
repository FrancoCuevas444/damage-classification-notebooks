#!/bin/bash
#SBATCH --job-name=fcuevas-job
#SBATCH --ntasks=1
#SBATCH --mem=8192
#SBATCH --time=00:15:00
#SBATCH --partition=besteffort
#SBATCH --qos=besteffort_gpu
#SBATCH --gres=gpu:1
#SBATCH -o salida.out

export PATH=$PATH:/usr/local/cuda/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64

cd ~/tesis-jupyters-2
conda activate tesis
python train_script.py
