#!/usr/bin/env python3
import subprocess
import os
import signal
import time
import sys

def main():
    time.sleep(10)
    w = withOSU()
    #w.testOSU()
    #w.congestion(withCongestor=1, core=32, instance=int(sys.argv[1]))
    w.allocation(instance=int(sys.argv[1])

class withOSU:
    def __init__(self):
        self.getNodelist()

    def parseNodeList(self, nodestr):
        nodelist = []
        if not nodestr.startswith("nid"):
            print("A job without nodes: {}".format(nodestr))
            return nodelist
        nodestr = nodestr.lstrip("nid0*")
        nodestr = nodestr.lstrip("[")
        nodestr = nodestr.rstrip("]")
        tmpstr = nodestr.split(',')
        for s in tmpstr:
            if '-' in s:
                tmpstr2 = s.split('-')
                nodelist += range(int(tmpstr2[0]), int(tmpstr2[1])+1)
            else:
                nodelist += [int(s)]
        return nodelist

    def getNodelist(self):
        output = subprocess.check_output('sacct --name=withOSU -n -P -X -o nodelist'.split(' ')).decode('utf-8')
        self.nodelist = self.parseNodeList(output.rstrip('\n').split('\n')[-1])
        print('current nodes:')
        print(self.nodelist)

    def jumpOne(self, nodelist, N):
        # requires 2N <= len(nodelist).
        return [nodelist[2*x+1] for x in range(N)]

    def abbrev(self, nodelist):
        nodestr = 'nid['
        nodestr += ','.join(['{0:05d}'.format(x) for x in nodelist])
        nodestr += ']'
        return nodestr
        
    def startOSU(self, N, core, instance):
        ntasks = core*N
        msize = 4096
        self.usedNodes = self.jumpOne(self.nodelist, N)
        osulist = self.abbrev(self.usedNodes)
        exe = '/global/homes/z/zhangyj/osu/osu-micro-benchmarks-5.6.2/install/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall'
        taskspernode = '--ntasks-per-node=%d ' % core if core != 32 else ''
        command = 'srun -N %d --ntasks %d --nodelist=%s %s-C haswell %s -m %d:%d -i 90000' % (N, ntasks, osulist, taskspernode, exe, msize, msize)
        procs = []
        for i in range(instance):
            osu = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print('osu %d started.' % i)
            procs.append(osu)
        return procs

    def startApp(self, N, alloc):
        print('Allocation method: %s' % alloc)
        ntasks = 32*N
        unUsedNodes = [x for x in self.nodelist if x not in self.usedNodes]
        if alloc == 'bad':
            appnodeList = unUsedNodes[:N]
        elif alloc == 'good':
            appnodeList = unUsedNodes[N:]
        liststr = ','.join(['nid{0:05d}'.format(y) for y in appnodeList])
        print('app-node list:')
        print(liststr)
        appcmd = 'module load openmpi; cd $HOME/miniMD/ref; '
        appcmd += 'time mpirun -np %d --host %s $HOME/miniMD/ref/miniMD_openmpi -n 160000; ' % (ntasks, liststr) # 160000.
        print('startTime:%s:' % (alloc) + subprocess.check_output(['date']).decode('utf-8'))
        apprun = subprocess.Popen(appcmd, stdout=subprocess.PIPE, shell=True)
        output = apprun.communicate()[0].strip()
        print(output.decode('utf-8'))
        print('endTime:%s:' % (alloc) + subprocess.check_output(['date']).decode('utf-8'))
        print()

    def congestion(self, withCongestor, core, instance):
        if withCongestor:
            procs = self.startOSU(N=32, core=core, instance=instance)
        else:
            print('osu not started due to setting.')

        self.startApp(N=32, alloc='bad')

        if withCongestor:
            print('is osu running:')
            for i in range(instance):
                osu = procs.pop()
                print(osu.poll() == None)
                os.killpg(os.getpgid(osu.pid), signal.SIGTERM)

    def allocation(self, instance, core=32):
        procs = self.startOSU(N=32, core=core, instance=instance)
        self.startApp(N=32, alloc='good')
        self.startApp(N=32, alloc='bad')

        print('is osu running:')
        for i in range(instance):
            osu = procs.pop()
            print(osu.poll() == None)
            os.killpg(os.getpgid(osu.pid), signal.SIGTERM)


if __name__ == '__main__':
    main()
