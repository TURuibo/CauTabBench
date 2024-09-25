#!/bin/env bash

#SBATCH -A NAISS2024-22-984      # find your project with the "projinfo" command
#SBATCH -t 0-00:30:00          # how long time it will take to run
#SBATCH --gpus-per-node=A40:1  # choosing no. GPUs and their type
#SBATCH -J graph_cdir_llama_v1.1           # the jobname (not necessary)

# Make sure to remove any already loaded modules
module purge

# Specify the path to the container
CONTAINER=/mimer/NOBACKUP/groups/naiss2024-22-565/huggingFace/llama.sif

# Print the PyTorch version then exit
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 1 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 2 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 3 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 4 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 5 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 6 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 7 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 8 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 9 --bt 1 --max_new_tokens 10000 --task_type graph_cdir &&
singularity exec $CONTAINER python llm_graph_cdir.py --llm llama --sim_seed 10 --bt 1 --max_new_tokens 10000 --task_type graph_cdir