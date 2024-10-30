#!/bin/bash
#SBATCH --job-name=deep_learning_job
#SBATCH --output=output_%j.txt
#SBATCH --error=error_%j.err
#SBATCH --ntasks=1
#SBATCH --time=3-00:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8

#SBATCH --partition=longq7-eng
#SBATCH --nodelist=nodeeng006

# Load necessary modules

module load cuda-9.2.88-gcc-8.3.0-52vvh4g


# Activate virtual environment


# Run your Python script
python epochs_50_subjects_64_channels.py 

