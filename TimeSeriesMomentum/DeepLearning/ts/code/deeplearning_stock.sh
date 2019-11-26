#!/bin/bash
#---------------------------------------------------------------------------------
# Account information

#SBATCH --account=pi-ngp
#---------------------------------------------------------------------------------
# Resources requested

#SBATCH --partition=standard
#SBATCH --cpus-per-task=6
#SBATCH --mem-per-cpu=24G  
#SBATCH --time=6-23:59:59
#---------------------------------------------------------------------------------
# Job specific name (helps organize and track progress of jobs)

#SBATCH --job-name=DL0222_30_1
#---------------------------------------------------------------------------------
# Load necessary modules for the job

module load python/booth/3.6/3.6.3
#---------------------------------------------------------------------------------
# Commands to execute below...

