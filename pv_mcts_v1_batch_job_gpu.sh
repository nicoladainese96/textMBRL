#!/bin/bash
#SBATCH --account=project_2001281
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1
#SBATCH --mem-per-gpu=16G 
#SBATCH --mail-user=nicola.dainese@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

# email research-support@csc.fi for help
module load pytorch
cd ~/textMBRL
python run_PVMCTS_v1.py $*
