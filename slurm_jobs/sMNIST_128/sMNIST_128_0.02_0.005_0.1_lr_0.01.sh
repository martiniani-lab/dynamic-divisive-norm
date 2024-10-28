#!/bin/bash
#SBATCH --nodes=1
#SBATCH --time=15:00:00
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --job-name=sMNIST_128_0.02_0.005_0.1_lr_0.01
#SBATCH --mem=64GB
#SBATCH --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=sr6364@nyu.edu
#SBATCH --output=job.%j.out

singularity exec --nv --overlay /scratch/sr6364/overlay-files/overlay-50G-10M.ext3:ro /scratch/work/public/singularity/cuda11.8.86-cudnn8.7-devel-ubuntu22.04.2.sif /bin/bash -c \
'source /ext3/env.sh; conda activate feed-r-conda; cd /home/sr6364/python_scripts/dynamic-divisive-norm/training_scripts/sMNIST; python train.py \
--MODEL_NAME sMNIST_128_0.02_0.005_0.1_lr_0.01 \
--FOLDER_NAME ../../tb_logs/ortho_init/sMNIST_128 \
--VERSION 0 \
--SEQUENCE_LENGTH 784 \
--dt_tau_max_y 0.02 \
--dt_tau_max_a 0.005 \
--dt_tau_max_b 0.1 \
--learn_tau True \
--HIDDEN_SIZE 128 \
--NUM_EPOCHS 300 \
--LEARNING_RATE 0.01 \
--SCHEDULER_CHANGE_STEP 30 \
--SCHEDULER_GAMMA 0.8 '
