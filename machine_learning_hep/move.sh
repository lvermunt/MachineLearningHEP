#!/bin/bash

#SBATCH --output=slurm-%J.out
#SBATCH --error=slurm-%J.out

function die
{
    echo $1
    exit
}

JOBIDX="-0"
[ -n "${SLURM_ARRAY_TASK_ID}" ] && JOBIDX="-${SLURM_ARRAY_TASK_ID}"
export JOBIDX

unset DISPLAY
export MLPBACKEND=pdf
mv /data/TTree/D0DsLckINT7HighMultwithJets/vAN-20190922_ROOT6-1/* /mnt/temp/MovedForTemporarySpace/TTree_D0DsLckINT7HighMultwithJets_vAN-20190922_ROOT6-1/

