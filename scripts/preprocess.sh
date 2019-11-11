#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
source /home/$USER/p3.6/bin/activate
module load cuda/9.0
module load cudnn/7-cuda-9.0
python /home/priyansh.agrawal/ire_final_project/scripts/lstm.py
