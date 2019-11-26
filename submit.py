#!/usr/bin/env python3

from subprocess import call

for run in range(100, 108):
    command = 'sbatch -N 64 -o $HOME/miniMD/results/run%d.out run.sh' % run
    call(command, shell=True)
