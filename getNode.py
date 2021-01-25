#!/usr/bin/env python3
import subprocess
import os

def getNodelist():
    #output = subprocess.check_output('sacct --name=withOSU -n -P -X -o nodelist'.split(' ')).decode('utf-8')
    #output = subprocess.check_output('echo ${SLURM_NODELIST}'.split(' ')).decode('utf-8') # don't work.
    output = os.environ['SLURM_NODELIST']
    print('current nodes:')
    print(output)

getNodelist()

