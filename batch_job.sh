#!/bin/bash
#SBATCH --account=project_2001281
#SBATCH --partition=small
#SBATCH --mem=16G 
#SBATCH --mail-user=nicola.dainese@aalto.fi
#SBATCH --mail-type=FAIL,REQUEUE,TIME_LIMIT_80

# email research-support@csc.fi for help
module load pytorch
python run_VMCTS.py $*
