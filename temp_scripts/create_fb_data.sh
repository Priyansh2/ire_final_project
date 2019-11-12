#!/bin/bash
#SBATCH -A research
#SBATCH -n 10
#SBATCH --qos=medium
#SBATCH -p long
#SBATCH --gres=gpu:0
#SBATCH --mem-per-cpu=3G
#SBATCH --time=4-00:00:00
source /home/$USER/p3.6/bin/activate
python /home/priyansh.agrawal/ire_final_project/temp_scripts/create_fb_data.py
