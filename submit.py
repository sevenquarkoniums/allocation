#!/usr/bin/env python3
from subprocess import call

def main():
    s = submit()
    #s.miniMD()
    #s.withOSU(timelimit='00:30:00', isSingle=1, isMail=1)
    s.allocation(timelimit='00:30:00', isSingle=1, isMail=1)

class submit:
    def miniMD(self):
        for run in range(110, 210):
            command = 'sbatch -N 64 -o $HOME/miniMD/results/run%d.out run.sh' % run
            call(command, shell=True)

    def withOSU(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        for run in range(0, 5):
            #command = 'sbatch -N 64 --account=m888 -q premium -C haswell %s-t %s -J withOSU %s-o $HOME/allocation/OSUresults/Cong_c32_ins8_%d.out withOSU.py 8' % (mail, timelimit, singleton, run)
            command = 'sbatch -N 64 --account=m888 -q premium -C haswell %s-t %s -J withOSU %s-o $HOME/allocation/OSUresults/Cong_c32_ins%d_%d.out withOSU.py %d' % (mail, timelimit, singleton, 5+run, 0, 5+run)
            call(command, shell=True)

    def allocation(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        for run in range(0, 1):
            command = 'sbatch -N 96 --account=m888 -q premium -C haswell %s-t %s -J withOSU %s-o $HOME/allocation/OSUresults/alloc_c32_ins10_%d.out withOSU.py 10' % (mail, timelimit, singleton, run)
            call(command, shell=True)

if __name__ == '__main__':
    main()
