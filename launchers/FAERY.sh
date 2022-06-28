#!/bin/bash

# Path to parameter file, additionnal parameters
PARAMS_PATH=/launchers/parameters/base_${1:-test}.params
PARAMS_OTHER="
--inner_algorithm QD
--steps_after_solved 0
"

# Creating the output directory
OUTPUT_PATH=/results/$1_$date
mkdir -p $OUTPUT_PATH

# Storing used parameters
cd $OUTPUT_PATH
echo "Parameters at" $PARAMS_PATH ":" > params.out
echo $(< $PARAMS_PATH) >> params.out
echo >> params.out
echo "Additionnal parameters:" >> params.out
echo $PARAMS_OTHER >> params.out

mkdir tmp_prog

# Launching the program
echo "Start time:", $(date) > main.out
python3 -B -m scoop --hosts localhost -n ${2:-32} main.py $(< $PARAMS_PATH) $PARAMS_OTHER --top_level_log tmp_prog >> main.out
echo "End time:", $(date) >> main.out

# Exit
exit 0
