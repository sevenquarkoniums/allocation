#!/usr/bin/env python3
from subprocess import call

def main():
    s = submit()
    #s.miniMD(timelimit='00:30:00', isSingle=1, isMail=1)
    #s.withOSU(timelimit='00:30:00', isSingle=1, isMail=1)
    #s.allocation(timelimit='00:30:00', isSingle=1, isMail=1)
    #s.fixAllocation(timelimit='01:40:00', isSingle=0, isMail=1)
    #s.CADD(N=151, queue='regular', fname='CADD_lammps_CADDmin151_run60k.out', timelimit='03:00:00', isMail=1)
    s.GPC(N=65, timelimit='00:30:00', isMail=1)
    #for day in range(9, 32):
    #    s.getData(day=day, N=1, timelimit='23:00:00', isMail=0)

class submit:
    def CADD(self, N, queue, fname, timelimit, isMail):
        # By default, a job step has access to every CPU allocated to the job.  To ensure that distinct CPUs are allocated to each job step, use the --exclusive option.
        # Check this for multiple srun, https://docs.nersc.gov/jobs/examples/#multiple-parallel-jobs-while-sharing-nodes
        mail = '--mail-type=END ' if isMail else ''
        command = 'sbatch -N %d --account=m3231 -q %s -C knl %s-t %s -J CADD --exclusive --gres=craynetwork:0 -o $HOME/allocation/%s runwith.sh' % (N, queue, mail, timelimit, fname)
        call(command, shell=True)

    def GPC(self, N, timelimit, isMail):
        mail = '--mail-type=END ' if isMail else ''
        command = 'sbatch -N %d --account=m3231 -q premium -C knl %s-t %s -J GPC --exclusive --gres=craynetwork:0 -o $HOME/allocation/GPC.out runwith.sh' % (N, mail, timelimit)
        call(command, shell=True)

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
        command = 'sbatch -N 192 --account=m3231 -q premium -C haswell %s-t %s -J nekbone %s-o $HOME/allocation/results/fixAlloc_nekbone.out withOSU.py' % (mail, timelimit, singleton)
        call(command, shell=True)

    def getData(self, day, N, timelimit, isMail):
        mail = '--mail-type=END ' if isMail else ''
        sh = 'sh/run%d.sh' % day
        with open(sh, 'w') as f:
            cmd = '#!/bin/bash\nmodule load python/3.6-anaconda-5.2\n'
            for idx in range(48):
                cmd += './main.py %d %d\n' % (day, idx)
            f.write(cmd)
        call('chmod +x %s' % sh, shell=True)
        out = 'out/run%d.out' % day
        command = 'sbatch -N %d --account=m3231 -q regular -C haswell %s-t %s -J getData%d -o $HOME/allocation/%s %s' % (N, mail, timelimit, day, out, sh)
        call(command, shell=True)

if __name__ == '__main__':
    main()
