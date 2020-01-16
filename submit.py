#!/usr/bin/env python3
from subprocess import call

def main():
    s = submit()
    #s.miniMD()
    s.withOSU(timelimit='00:30:00', isSingle=1, isMail=0)

class submit:
    def miniMD(self):
        for run in range(110, 210):
            command = 'sbatch -N 64 -o $HOME/miniMD/results/run%d.out run.sh' % run
            call(command, shell=True)

    def withOSU(self, timelimit, isSingle, isMail):
        singleton = '-d singleton ' if isSingle else ''
        mail = '--mail-type=END ' if isMail else ''
        for run in range(0, 20):
            command = 'sbatch -N 64 --account=m3410 -q regular -C haswell %s-t %s -J withOSU %s-o $HOME/allocation/OSUresults/Cong24c_%d.out withOSU.py' % (mail, timelimit, singleton, run)
            call(command, shell=True)

if __name__ == '__main__':
    main()
