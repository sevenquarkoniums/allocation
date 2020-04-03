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
    #w.congestion(withCongestor=0, core=32, instance=int(sys.argv[1]))
    #w.allocation(instance=int(sys.argv[1]))
    w.fixAllocation(appName='milc', iteration=1, instance=5)

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
        #output = subprocess.check_output('sacct --name=withOSU -n -P -X -o nodelist'.split(' ')).decode('utf-8') # only work for one job.
        #output = subprocess.check_output('echo $SLURM_NODELIST'.split(' ')).decode('utf-8') # don't work.
        output = os.environ['SLURM_NODELIST']
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
        
    def startOSU(self, N, core, instance, nodes):
        ntasks = core*N
        msize = 4096
        osulist = self.abbrev(nodes)
        exe = '/global/homes/z/zhangyj/osu/osu-micro-benchmarks-5.6.2/install/libexec/osu-micro-benchmarks/mpi/collective/osu_alltoall'
        taskspernode = '--ntasks-per-node=%d ' % core if core != 32 else ''
        command = 'srun -N %d --ntasks %d --nodelist=%s %s-C haswell %s -m %d:%d -i 90000' % (N, ntasks, osulist, taskspernode, exe, msize, msize)
        procs = []
        for i in range(instance):
            osu = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print('osu %d started.' % i)
            procs.append(osu)
        return procs

    def startGPC(self, N, instance, nodes):
        ntasks = 32*N
        procs = []
        gpclist = self.abbrev(nodes)
        for i in range(instance):
            command = 'srun -N %d --ntasks %d --nodelist=%s --ntasks-per-node=32 -C haswell /global/homes/z/zhangyj/GPCNET/network_load_test > results/gpc_%s_N%d_run%d.out' % (N, ntasks, gpclist, self.appName, N, self.gpcRun)
            gpc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print('gpc %d started.' % i)
            procs.append(gpc)
            self.gpcRun += 1
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
        appcmd += 'time mpirun -np %d --host %s $HOME/miniMD/ref/miniMD_openmpi -n 160000; ' % (ntasks, liststr)
        print('startTime:%s:' % (alloc) + subprocess.check_output(['date']).decode('utf-8'))
        apprun = subprocess.Popen(appcmd, stdout=subprocess.PIPE, shell=True)
        output = apprun.communicate()[0].strip()
        print(output.decode('utf-8'))
        print('endTime:%s:' % (alloc) + subprocess.check_output(['date']).decode('utf-8'))
        print()

    def appOnNodes(self, app, N, nodes):
        '''
        Run app on some nodes.
        app: nekbone, miniMD, lammps, miniamr, hacc.
        '''
        ntasks = 32*N
        liststr = ','.join(['nid{0:05d}'.format(y) for y in nodes])
        print('app-node list:')
        print(nodes)
        appcmd = 'module load openmpi; '
        if app == 'miniMD':
            appcmd += 'cd $HOME/miniMD/ref; '
            appcmd += 'mpirun -np %d --host %s $HOME/miniMD/ref/miniMD_openmpi -n 160000; ' % (ntasks, liststr)
        elif app == 'nekbone':
            appcmd += 'cd $HOME/allocation/nekbone/Nekbone/test/example1; '
            appcmd += 'mpirun -np %d --host %s ./nekbone ex1; ' % (ntasks, liststr)
        elif app == 'lammps':
            appcmd += 'cd $HOME/allocation/lammps/testrun; '
            appcmd += 'srun -N %d --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/LAMMPS/src/LAMMPS -in in.vacf.2d; ' % (N, liststr)
        elif app == 'miniamr':
            appcmd += 'cd $HOME/allocation/miniamr/testrun; '
            appcmd += 'srun -N %d --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/miniAMR_1.0_all/miniAMR_ref/miniAMR.x --num_refine 4 --max_blocks 5000 --init_x 1 --init_y 1 --init_z 1             --npx 16 --npy 16 --npz 8 --nx 6 --ny 6 --nz 6 --num_objects 2             --object 2 0 -1.10 -1.10 -1.10 0.030 0.030 0.030 1.5 1.5 1.5 0.0 0.0 0.0             --object 2 0 0.5 0.5 1.76 0.0 0.0 -0.025 0.75 0.75 0.75 0.0 0.0 0.0             --num_tsteps 10 --stages_per_ts 125 --report_perf 4; ' % (N, liststr)
        elif app == 'hacc':
            appcmd += 'cd $HOME/allocation/hacc/testrun; '
            appcmd += 'srun -N %d --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/HACC_1_7/HACC indat cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 64 -a final -f refresh -t 16x16x8; ' % (N, liststr)
        elif app == 'graph500':
            appcmd += 'cd $HOME/allocation/graph500/src; export SKIP_VALIDATION=1; '
            appcmd += 'mpirun -np %d --host %s ./graph500_reference_bfs_sssp 26; ' % (ntasks, liststr)
        elif app == 'milc':
            appcmd += 'cd $HOME/milc_qcd-7.8.1/ks_imp_dyn/test; '
            appcmd += 'srun -N %d --ntasks-per-node=32 --nodelist=%s ../su3_rmd myinput.in; ' % (N, liststr)
        print('startTime:' + subprocess.check_output(['date']).decode('utf-8'))
        apprun = subprocess.Popen(appcmd, stdout=subprocess.PIPE, shell=True)
        output = apprun.communicate()[0].strip()
        print(output.decode('utf-8'))
        print('endTime:' + subprocess.check_output(['date']).decode('utf-8'))
        print()

    def congestion(self, withCongestor, core, instance):
        N = 32
        self.usedNodes = self.jumpOne(self.nodelist, N)
        if withCongestor:
            procs = self.startOSU(N=N, core=core, instance=instance, nodes=self.usedNodes)
        else:
            print('osu not started due to setting.')

        self.startApp(N=N, alloc='bad')

        if withCongestor:
            print('is osu running:')
            for i in range(instance):
                osu = procs.pop()
                print(osu.poll() == None)
                os.killpg(os.getpgid(osu.pid), signal.SIGTERM)

    def allocation(self, instance, core=32):
        N = 32
        self.usedNodes = self.jumpOne(self.nodelist, N)
        procs = self.startOSU(N=N, core=core, instance=instance, nodes=self.usedNodes)
        self.startApp(N=N, alloc='good')
        self.startApp(N=N, alloc='bad')

        print('is osu running:')
        for i in range(instance):
            osu = procs.pop()
            print(osu.poll() == None)
            os.killpg(os.getpgid(osu.pid), signal.SIGTERM)

    def fixAllocation(self, appName, iteration, instance):
        self.gpcRun = 0
        self.appName = appName
        N = 64
        rotate = 3*N // iteration
        for i in range(iteration):
            print('iteration:%d' % i)
            allNodes = self.nodelist[i*rotate:] + self.nodelist[:i*rotate]
            congestNodes = [allNodes[2*x+1] for x in range(N)]
            greenNodes = allNodes[-N:]
            yellowNodes = [allNodes[2*x] for x in range(N)]

            # run without congestor.
            self.appOnNodes(app=appName, N=N, nodes=greenNodes)
            self.appOnNodes(app=appName, N=N, nodes=yellowNodes)

            # run with congestor.
            procs = self.startGPC(N=N, instance=instance, nodes=congestNodes)
            #procs = self.startOSU(N=N, core=32, instance=instance, nodes=congestNodes)
            self.appOnNodes(app=appName, N=N, nodes=greenNodes)
            self.appOnNodes(app=appName, N=N, nodes=yellowNodes)
            print('is congestor running:')
            for j in range(instance):
                cong = procs.pop()
                print(cong.poll() == None)
                if cong.poll() == None:
                    os.killpg(os.getpgid(cong.pid), signal.SIGTERM)


if __name__ == '__main__':
    main()
