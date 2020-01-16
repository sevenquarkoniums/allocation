#!/usr/bin/env python3
import subprocess
import os
import signal
import time

def main():
    time.sleep(5)
    w = withOSU()
    #w.testOSU()
    w.congestion(withCongestor=1, core=24)

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
        
    def testOSU(self):
        N = 32
        ntasks = 32*N
        msize = 4096
        osulist = self.abbrev(self.jumpOne(self.nodelist, N))
        exe = '/global/homes/z/zhangyj/osu/osu-micro-benchmarks-5.6.2/install/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall'
        command = 'srun -N %d --ntasks %d --nodelist=%s -C haswell %s -m %d:%d -i 100\n' % (N, ntasks, osulist, exe, msize, msize)
        print(command)
        print(subprocess.check_output(['date']).decode('utf-8'))
        output = subprocess.check_output(command.split(' ')).decode('utf-8')
        #run = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
        #output = run.communicate()[0].strip()
        print(output)
        print(subprocess.check_output(['date']).decode('utf-8'))
        #osu = subprocess.Popen(command, preexec_fn=os.setsid, shell=True)

    def congestion(self, withCongestor, core=32):
        N = 32
        ntasks = core*N
        msize = 4096
        osulist = self.abbrev(self.jumpOne(self.nodelist, N))
        exe = '/global/homes/z/zhangyj/osu/osu-micro-benchmarks-5.6.2/install/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall'
        taskspernode = '--ntasks-per-node=%d ' % core if core != 32 else ''
        command = 'srun -N %d --ntasks %d --nodelist=%s %s-C haswell %s -m %d:%d -i 90000' % (N, ntasks, osulist, taskspernode, exe, msize, msize)
            # iter 1000 for 32 nodes is approx 20 min.
        if withCongestor:
            osu = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print('osu started.')
            #stdout, stderr = osu.communicate()
            #print(stdout)
            #print(stderr)
        else:
            print('osu not started.')

        N = 32
        ntasks = 32*N
        appnodeList = [x for x in self.nodelist if x not in self.jumpOne(self.nodelist, N)]
        liststr = ','.join(['nid{0:05d}'.format(y) for y in appnodeList])
        print('app-node list:')
        print(liststr)
        appcmd = 'module load openmpi; cd $HOME/miniMD/ref; '
        appcmd += 'time mpirun -np %d --host %s $HOME/miniMD/ref/miniMD_openmpi -n 160000; ' % (ntasks, liststr) # 160000.
        print('startTime: ' + subprocess.check_output(['date']).decode('utf-8'))
        apprun = subprocess.Popen(appcmd, stdout=subprocess.PIPE, shell=True)
        output = apprun.communicate()[0].strip()
        print(output.decode('utf-8'))
        print('endTime: ' + subprocess.check_output(['date']).decode('utf-8'))

        if withCongestor:
            print('is osu running:')
            print(osu.poll() == None)
            os.killpg(os.getpgid(osu.pid), signal.SIGTERM)

if __name__ == '__main__':
    main()
