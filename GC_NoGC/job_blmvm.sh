#!/bin/bash
#PBS -N NoGC_BLMVM
#PBS -P xd2
#PBS -q normalsr
#PBS -l walltime=24:00:00
#PBS -l mem=500GB
#PBS -l ncpus=48
#PBS -l jobfs=400GB
#PBS -l storage=scratch/xd2+gdata/xd2+gdata/fp50
#PBS -l wd
#PBS -W umask=0022

### Submit from Stage_1 directory with:
###
### qsub -v RUN_ID=1,MAX_RUNS=4 ../job.sh 

### RUN_ID and MAX_RUNS are passed through qsub -v flags
if [[ "${RUN_ID}" -lt "${MAX_RUNS}" ]]; then
    next_run=$(( "${RUN_ID}" + 1 ))
    ### Create next run directory
    mkdir "../Stage_${next_run}"
    pushd "../Stage_${next_run}"
    ### Submit next job (will only run if/when current job completes successfully)
    qsub -W depend=afterok:${PBS_JOBID} -v RUN_ID=${next_run},MAX_RUNS=${MAX_RUNS} ../job_blmvm.sh
    popd
fi

prev_run=$(( "${RUN_ID}" - 1 ))

### Read run params from run_info.txt
read id restore_it max_it alpha_T alpha_u alpha_m alpha_d alpha_s checkpoint_restore < <( sed -n ${RUN_ID}p ../run_info.txt )
export ID=$id
export RESTORE_IT=$restore_it
export MAX_IT=$max_it
export ALPHA_T=$alpha_T
export ALPHA_U=$alpha_u
export ALPHA_M=$alpha_m
export ALPHA_D=$alpha_d
export ALPHA_S=$alpha_s
export CHECKPOINT_RESTORE=$checkpoint_restore

export PYTHONPATH=/home/135/rb0141/install
module use /g/data/fp50/modules
export MY_GADOPT=/g/data/xd2/rad552/FIREDRAKE_GIT/g-adopt
module load firedrake/main-20260312
module load gcc/14.2.0

# This is to make sure we only compile on rank 0
export OMPI_MCA_io="ompio"

# Run your script
### inverse.py has been modified to read TIME and SIM_END environment variables
mpiexec -np $PBS_NCPUS python ../mpl_wrapper.py ../inverse_BLMVM.py &> output_BLMVM.dat
next_run=$((RUN_ID + 1))
src="../optimisation_checkpoint_${RUN_ID}/${MAX_IT}"
dest="../optimisation_checkpoint_${next_run}"
# Ensure destination directory exists:
mkdir -p "$dest"
# Check and copy
if [ -d "$src" ]; then
    cp -r "$src" "$dest/"
else
    echo "WARNING: Source directory '$src' does not exist. Skipping copy." >&2
fi



