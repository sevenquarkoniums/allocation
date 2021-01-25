#!/bin/bash

module unload darshan/3.1.7
module load perftools-base
module load perftools-lite
cd /global/project/projectdirs/m3231/yijia/cook/mpi
if [[ "$1" == "miniMD" ]]
then
    cd miniMD/miniMD/craypat
    srun -N 32 --ntasks-per-node=68 ./miniMD_cray_intel+pat -n 80000
elif [[ "$1" == "lammps" ]]
then
    cd LAMMPS
    srun -N 32 --ntasks-per-node=68 ./LAMMPS/src/lmp_cori+pat -in in.vacf.2d
elif [[ "$1" == "miniAMR" ]]
then
    cd miniAMR
    srun -N 32 --ntasks-per-node=32 ./miniAMR_ref/miniAMR.x+pat --num_refine 4 --max_blocks 5000 --init_x 1 --init_y 1 --init_z 1 --npx 16 --npy 8 --npz 8 --nx 6 --ny 6 --nz 6 --num_objects 2 --object 2 0 -1.10 -1.10 -1.10 0.030 0.030 0.030 1.5 1.5 1.5 0.0 0.0 0.0 --object 2 0 0.5 0.5 1.76 0.0 0.0 -0.025 0.75 0.75 0.75 0.0 0.0 0.0 --num_tsteps 20 --stages_per_ts 125 --report_perf 4
elif [[ "$1" == "hacc" ]]
then
    cd HACC/HACC_1_7/test
    srun -N 32 --ntasks-per-node=64 ../src/cpu/cori/hacc_tpm+pat indat cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 32 -a final -f refresh -t 16x16x8
elif [[ "$1" == "milc" ]]
then
    cd MILC 
    srun -N 32 --ntasks-per-node=64 ./milc_qcd-7.8.1/ks_imp_dyn/su3_rmd+pat $HOME/milc_qcd-7.8.1/ks_imp_dyn/test/myinput_knl.in
elif [[ "$1" == "hpcg" ]]
then
    cd HPCG 
    srun -N 32 --ntasks-per-node=68 ./hpcg/bin/xhpcg+pat --nx=64 --rt=50
elif [[ "$1" == "qmcpack" ]]
then
    cd QMCPACK/testrun
    export HDF5_USE_FILE_LOCKING=FALSE
    srun -N 32 --ntasks-per-node=68 ../qmcpack_ben/build_bgc/bin/qmcpack+pat simple-H2O.xml
fi
date
