#!/bin/bash
#module unload python/3.6-anaconda-5.2
#module load python/2.7-anaconda-4.4

date

export TOP=/project/projectdirs/m3231/yijia/ldms/ovis/build
export LD_LIBRARY_PATH=$TOP/lib/:$TOP/lib/ovis-ldms:$TOP/sbin:$LD_LIBRARY_PATH
export LDMSD_PLUGIN_LIBPATH=$TOP/lib/ovis-ldms/
export ZAP_LIBPATH=$TOP/lib/ovis-ldms/
export PATH=$TOP/sbin:$TOP/bin:$PATH

# installation of LDMS we are using
export OVIS_HOME=$TOP

# place we want our LDMS data to go (path must exist)
export LDMS_STOREPATH=/project/projectdirs/m3231/yijia/csv
export LDMS_CONTAINER=$1
export LDMS_SOS_RAWDB=/project/projectdirs/m3410/ovis_data/raw
export LDMS_STOREMODE=csv

# script for running LDMS daemons (change only if you copy it)
LDMSSCRIPT=/global/homes/z/zhangyj/allocation/ldmsrun.py

# set the machine name
export Machine=cori

srun -n ${SLURM_NNODES} -m cyclic --mem=4G ${LDMSSCRIPT} sampler &
# allow to start up
sleep 30

date

# run ldms storage daemon only on first node
echo "starting store daemon"
#srun -N 1 --nodelist=$2 --mem=64G ${LDMSSCRIPT} store & # will fail if not on this node.
${LDMSSCRIPT} store &

sleep $3

date

#echo "kill ldmsd"
#srun -n ${SLURM_NNODES} -m cyclic --mem=1G pkill ldmsd # may be delayed by srun problem.

date

