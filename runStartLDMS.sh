#!/bin/bash
srun -N 1 --nodelist=$2 ./startLDMS.sh $1 $3
