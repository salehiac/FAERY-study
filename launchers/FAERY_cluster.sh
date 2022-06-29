#!/bin/bash

#SBATCH --job-name=FAERY
#SBATCH --cpus-per-task=32
#SBATCH --array=1
#SBATCH --nodelist=cpu24
#SBATCH --partition=cpu-1semaine
#SBATCH --output=/home/yannis.elrharbi/reports/report_%A_%a.out

# Runs FAERY on a cluster in singularity container with specified parameters file

# Path to parameter file, additionnal parameters
PARAMS_PATH=~/launchers/parameters/base_assembly.params
PARAMS_OTHER="
--inner_algorithm QD
--steps_after_solved 0
"

SOURCE_SINGULARITY=faery.sif

# Creating the output directory
OUTPUT_PATH=~/results/$SLURM_JOB_NAME\_$1\_$SLURM_ARRAY_JOB_ID\_$SLURM_ARRAY_TASK_ID
mkdir $OUTPUT_PATH

# Storing used parameters
cd $OUTPUT_PATH
echo "Parameters at" $PARAMS_PATH ":" > params.out
echo $(< $PARAMS_PATH) >> params.out
echo >> params.out
echo "Additionnal parameters:" >> params.out
echo $PARAMS_OTHER >> params.out

# Making the scratch (work) directory
SCRATCH=$SLURM_TMPDIR/$SLURM_ARRAY_TASK_ID
mkdir -p $SCRATCH/tmp_prog

# Moving the source to the scratch directory
cp $SLURM_SUBMIT_DIR/singularity/$SOURCE_SINGULARITY $SCRATCH
cd $SCRATCH

# Launching the program
echo "Start time:", $(date) > main.out
srun singularity exec --bind $SCRATCH:$HOME $SOURCE_SINGULARITY python3 -B -m scoop --hosts localhost -n 32 /src/FAERY/main.py $(< $PARAMS_PATH) $PARAMS_OTHER --top_level_log tmp_prog >> main.out
echo "End time:", $(date) >> main.out

# Retrieving the output
cp main.out $OUTPUT_PATH
cp -r $SCRATCH/tmp_prog/* $OUTPUT_PATH

# Removing work files
cd
rm -rf $SCRATCH
