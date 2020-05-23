#!/usr/bin/env python3
from subprocess import call

def main():
    s = submit()
    #s.miniMD(timelimit='00:30:00', isSingle=1, isMail=1)
    #s.withOSU(timelimit='00:30:00', isSingle=1, isMail=1)
    #s.allocation(timelimit='00:30:00', isSingle=1, isMail=1)
    s.fixAllocation(timelimit='01:40:00', isSingle=0, isMail=1)
    #s.CADD(N=193, timelimit='02:00:00', isMail=1)

class submit:
    def miniMD(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        for run in range(0, 2):
            command = 'sbatch -N 32 --account=m888 -q premium -C haswell %s-t %s -J miniMD %s-o $HOME/allocation/results/miniMD_n32_%d.out withOSU.py 5' % (mail, timelimit, singleton, run)
            call(command, shell=True)

    def withOSU(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        for run in range(0, 2):
            command = 'sbatch -N 64 --account=m888 -q premium -C haswell %s-t %s -J withOSU %s-o $HOME/allocation/results/miniMD_n32_%d.out withOSU.py 0' % (mail, timelimit, singleton, run)
            call(command, shell=True)

    def allocation(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        for run in range(27, 28):
            command = 'sbatch -N 96 --account=m3231 -q premium -C haswell %s-t %s -J withOSU %s-o $HOME/allocation/results/alloc_c32_ins5_%d.out withOSU.py 5' % (mail, timelimit, singleton, run)
            call(command, shell=True)

    def fixAllocation(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        command = 'sbatch -N 192 --account=m888 -q premium -C haswell %s-t %s -J lammps %s-o $HOME/allocation/results/fixAlloc_lammps3.out withOSU.py' % (mail, timelimit, singleton)
        call(command, shell=True)

    def CADD(self, N, timelimit, isMail):
        # By default, a job step has access to every CPU allocated to the job.  To ensure that distinct CPUs are allocated to each job step, use the --exclusive option.
        # Check this for multiple srun, https://docs.nersc.gov/jobs/examples/#multiple-parallel-jobs-while-sharing-nodes
        mail = '--mail-type=END ' if isMail else ''
        command = 'sbatch -N %d --account=m3231 -q regular -C haswell %s-t %s -J CADD --exclusive --gres=craynetwork:0 -o $HOME/allocation/CADD.out runwith.sh' % (N, mail, timelimit)
        call(command, shell=True)

if __name__ == '__main__':
    main()
