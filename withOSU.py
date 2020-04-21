#!/usr/bin/env python3
import subprocess
import os
import signal
import time
import sys
import random
import multiprocessing
import pandas as pd

def main():
    time.sleep(10)
    w = withOSU()
    #w.testOSU()
    #w.congestion(withCongestor=0, core=32, instance=int(sys.argv[1]))
    #w.allocation(instance=int(sys.argv[1]))
    #w.fixAllocation(appName='milc', iteration=10, instance=5)
    w.CADD(appName='miniMD', iteration=10)

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

    def startGPC(self, instance, nodes):
        N = len(nodes)
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
            procs = self.startGPC(instance=instance, nodes=congestNodes)
            #procs = self.startOSU(N=N, core=32, instance=instance, nodes=congestNodes)
            self.appOnNodes(app=appName, N=N, nodes=greenNodes)
            self.appOnNodes(app=appName, N=N, nodes=yellowNodes)
            print('is congestor running:')
            for j in range(instance):
                cong = procs.pop()
                print(cong.poll() == None)
                if cong.poll() == None:
                    os.killpg(os.getpgid(cong.pid), signal.SIGTERM)

    def startContGPC(self, nodes):
        print('start GPC checker.')
        self.GPCnodes = nodes
        self.q = multiprocessing.Queue()
        self.GPCchecker = multiprocessing.Process(target=self.continualGPC, args=(self.q,))
        self.GPCchecker.start()

    def continualGPC(self, q):
        '''
        Only 1 instace.
        '''
        print(self.GPCnodes)
        N = len(self.GPCnodes)
        ntasks = 32*N
        gpclist = self.abbrev(self.GPCnodes)
        command = 'srun -N %d --ntasks %d --nodelist=%s --ntasks-per-node=32 -C haswell /global/homes/z/zhangyj/GPCNET/network_load_test > results/continualGPC_.out' % (N, ntasks, gpclist)
        print(command)
        GPCproc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        print('continual GPC started.')
        while 1:
            time.sleep(0.5)
            if not q.empty():
                message = q.get()
                if message == 'stop':
                    if GPCproc.poll() == None: # if not finished.
                        os.killpg(os.getpgid(GPCproc.pid), signal.SIGTERM)
                        print('GPC killed in continualGPC().')
                    break
            if GPCproc.poll() != None: # if finished.
                print('restarting GPC..')
                #GPCproc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

    def stopGPC(self):
        '''
        may not run due to srun problem.
        '''
        self.q.put('stop')
        #self.GPCchecker.terminate()
        self.GPCchecker.join()
        print('continualGPC() finished.')
        self.q.close()
        self.q.join_thread()
        print('GPC killed.')

    def runLDMS(self, foldername, seconds):
        '''
        Additional 30s to start sampler is not counted.
        '''
        print('starting LDMS..')
        proc = subprocess.call('./startLDMS.sh %s %d' % (foldername, seconds), shell=True)
        print('LDMS ended.')
        self.ldmsdir = '/project/projectdirs/m3231/yijia/csv/%s' % foldername
        print('runLDMS() finish.')

    def sortCongestion(self):
        print('Reading cray csv..')
        df = pd.read_csv('%s/cray_aries_r' % self.ldmsdir)
        print('%.1f seconds collected in total.' % (df.iloc[-1]['#Time'] - df.iloc[0]['#Time']))
        print('Calculating stall sum..')
        df['stalled_sum'] = df[['stalled_%03d (ns)' % x for x in range(48)]].sum(axis=1)
        print('Calculating node stall..')
        nodeCong = []
        selectTime = df[ (df['#Time'] >= self.monitorstart) & (df['#Time'] <= self.monitorend) ]
        for node in self.idleNodes:
            select = selectTime[selectTime['component_id']==node]
            if len(select) == 0:
                print('No data for node %d' % node)
            stall = ( select.iloc[-1]['stalled_sum'] - select.iloc[0]['stalled_sum'] ) / ( select.iloc[-1]['#Time'] - select.iloc[0]['#Time'] )
            nodeCong.append((node, stall))
        print('Sorting nodes..')
        nodeCongPair = sorted(nodeCong, key=lambda x: x[1]) # ascending.
        print(nodeCongPair)
        return nodeCongPair

    def CADD(self, appName, iteration):
        '''
        Experiment for the Congestion-Aware Data-Driven allocation policy.
        '''
        #self.runLDMS(foldername='%s_%d' % (os.environ['SLURM_JOB_ID'], 0), seconds=120)
        #sys.exit(0)
        jobid = os.environ['SLURM_JOB_ID']
        print('jobid: ' + str(jobid))

        for i in range(iteration):
            print('====================')
            print('iteration %d' % i)
            congNodes = random.sample(self.nodelist[1:], 10) # skip the 1st node for LDMS.
            print('Congestor nodes:')
            print(congNodes)
            self.idleNodes = [x for x in self.nodelist[1:] if x not in congNodes]
            self.startContGPC(nodes=congNodes)

            self.monitorstart = int(time.time())
            if i == 0:
                self.runLDMS(foldername='%s_%d' % (jobid, i), seconds=120)
            else:
                time.sleep(120)
            self.monitorend = int(time.time())

            print('Starting sortCongestion()..')
            nodeCongPair = self.sortCongestion() # sort idle nodes from low to high congestion.
            greenNodes = [nodeCongPair[x][0] for x in range(4)]
            yellowNodes = [nodeCongPair[x][0] for x in range(4, 8)]
            print('Run green job.')
            self.appOnNodes(app=appName, N=4, nodes=greenNodes)
            print('Run yellow job.')
            self.appOnNodes(app=appName, N=4, nodes=yellowNodes)
            self.stopGPC()
            time.sleep(5)
            print('Iteration end.')
            print()

if __name__ == '__main__':
    main()
