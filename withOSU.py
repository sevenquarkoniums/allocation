#!/usr/bin/env python3
import subprocess

def main():
    w = withOSU()
    w.testOSU()
    #w.together()

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
        output = subprocess.check_output('sacct --name=withOSU -s R -n -P -X -o nodelist'.split(' ')).decode('utf-8')
        self.nodelist = self.parseNodeList(output.rstrip('\n'))

    def jumpOne(self, nodelist, N):
        # requires 2N <= len(nodelist).
        return [nodelist[2*x+1] for x in range(N)]

    def abbrev(self, nodelist):
        nodestr = 'nid0['
        nodestr += ','.join(['{0:04d}'.format(x) for x in nodelist])
        nodestr += ']'
        return nodestr
        
    def testOSU(self):
        N = 32
        ntasks = 32*N
        msize = 4096
        osulist = self.abbrev(self.jumpOne(self.nodelist, N))
        exe = '/global/homes/z/zhangyj/osu/osu-micro-benchmarks-5.6.2/install/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall'
        command = 'time srun -N %d --ntasks %d --nodelist=%s -C haswell %s -m %d:%d -i 1000\n' % (N, ntasks, osulist, exe, msize, msize)
        print(command)
        output = subprocess.check_output(command.split(' ')).decode('utf-8')
        print(output)
        #osu = subprocess.Popen(command, preexec_fn=os.setsid, shell=True)

    def together(self):
        exe = '/global/homes/z/zhangyj/osu/osu-micro-benchmarks-5.6.2/install/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall'
        command = 'time srun -N 32 --ntasks-per-node=32 -C haswell %s -m 4096:4096 -i 1000\n' % exe
        osu = subprocess.Popen(command, preexec_fn=os.setsid, shell=True)

if __name__ == '__main__':
    main()
