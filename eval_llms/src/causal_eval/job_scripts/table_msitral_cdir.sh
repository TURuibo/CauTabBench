#!/bin/env bash

#SBATCH -A NAISS2024-22-984      # find your project with the "projinfo" command
#SBATCH -t 0-10:00:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J table_mistral_v2.0          # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# Specify the path to the container
CONTAINER=/mimer/NOBACKUP/groups/naiss2024-22-565/huggingFace/llama.sif

# Print the PyTorch version then exit
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 10  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 1  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 2  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 3  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 4  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 5  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 6  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 7  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 8  --input_type table --max_table_rows 50 --batch_size 1 &&
singularity exec $CONTAINER python llm_table_cdir.py --llm mistral  --max_new_tokens 10 --sim_seed 9  --input_type table --max_table_rows 50 --batch_size 1