#!/bin/bash
#SBATCH --account=m3229
#SBATCH -q regular
#SBATCH -J miniMD
#SBATCH -d singleton
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH --nodes=64
#SBATCH --mail-type=END

module load openmpi
cd ~/miniMD/ref
date
time mpirun -np 2048 ./miniMD_openmpi -n 320000
date
