#!/bin/bash
#SBATCH --job-name=timeTest

#SBATCH --acount=fc_cellmorph
#SBATCH --partition=savio2

module load python/3.6
module load imagemagick

#SBATCH --tasks-per-node=24
#SBATCH --cpus-per-task=1

rclone copy arjit_bdrive:/Cell_Morphology_Research/vit_A_free ./vit_A_free
rclone copy arjit_bdrive:/Cell_Morphology_Research/RD1 ./RD1
rclone copy arjit_bdrive:/Cell_Morphology_Research/RD1-P2X7KO ./RD1-P2X7KO
rclone copy arjit_bdrive:/Cell_Morphology_Research/WT ./WT

python3 cell-analysis.py
