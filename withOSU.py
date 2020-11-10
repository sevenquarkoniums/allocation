#!/usr/bin/env python3
import subprocess
import os
import signal
import time
import sys
import random
import multiprocessing
from shutil import copyfile

def main():
    time.sleep(10)
    w = withOSU(knl=1)
    #w.testOSU()
    #w.congestion(withCongestor=0, core=32, instance=int(sys.argv[1]))
    #w.allocation(instance=int(sys.argv[1]))
    #w.appOnNodes(app='lammps', N=4, nodes=w.nodelist) # used to test application run.
    #w.fixAllocation(appName='lammps', iteration=10, instance=5)
    w.CADD(appName='lammps', iteration=20, congSize=64, appSize=32, appOut='CADDjob_lammps_knlTile.out')

class withOSU:
    def __init__(self, knl):
        self.knl = knl
        self.getNodelist()
        if knl:
            print('Using knl nodes.')
        else:
            print('Using haswell nodes.')

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
        '''
        Format the nodelist for use in srun command.
        '''
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
        '''
        instance: number to instance of GPCNET to start. Seems 1 instance is enough.
        '''
        N = len(nodes)
        ntasks = 32*N
        procs = []
        gpclist = self.abbrev(nodes)
        for i in range(instance):
            command = 'srun -N %d --mem=100G --ntasks %d --nodelist=%s --ntasks-per-node=32 -C haswell /global/homes/z/zhangyj/GPCNET/network_load_test > results/gpc_%s_N%d_run%d.out' % (N, ntasks, gpclist, self.appName, N, self.gpcRun)
            gpc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            print('gpc %d started.' % i)
            procs.append(gpc)
            self.gpcRun += 1
        return procs

    def startApp(self, N, alloc):
        '''
        obsolete
        '''
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

    def appOnNodes(self, app, N, nodes, writeToFile=None):
        '''
        Run app on some nodes.
        '''
        tasksPerNode = 68 if self.knl else 32
        ntasks = N * tasksPerNode
        liststr = ','.join(['nid{0:05d}'.format(y) for y in nodes]) # nodelist for srun or mpirun format.
        print('app-node list:')
        print(nodes)
        # create bash command for cori.
        appcmd = 'module load openmpi; '
        if app == 'miniMD':
            appcmd += 'cd $HOME/miniMD/ref; '
            appcmd += 'mpirun -np %d --host %s $HOME/miniMD/ref/miniMD_openmpi -n 160000' % (ntasks, liststr) # may trigger LDMS daemon problem by mpirun.
        elif app == 'nekbone':
            appcmd += 'cd $HOME/allocation/nekbone/Nekbone/test/example1; '
            appcmd += 'mpirun -np %d --host %s ./nekbone ex1' % (ntasks, liststr)
        elif app == 'lammps':
            appcmd += 'cd $HOME/allocation/lammps/testrun; '
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/LAMMPS/src/LAMMPS -in in.vacf.2d' % (N, tasksPerNode, liststr)
        elif app == 'miniamr':
            appcmd += 'cd $HOME/allocation/miniamr/testrun; '
            #appcmd += 'sbcast --compress=lz4 /project/projectdirs/m3410/applications/withoutIns/miniAMR_1.0_all/miniAMR_ref/miniAMR.x /tmp/miniAMR.x; '
            appcmd += 'srun -N %d --mem=100G --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/miniAMR_1.0_all/miniAMR_ref/miniAMR.x --num_refine 4 --max_blocks 5000 --init_x 1 --init_y 1 --init_z 1             --npx 16 --npy 16 --npz 8 --nx 6 --ny 6 --nz 6 --num_objects 2             --object 2 0 -1.10 -1.10 -1.10 0.030 0.030 0.030 1.5 1.5 1.5 0.0 0.0 0.0             --object 2 0 0.5 0.5 1.76 0.0 0.0 -0.025 0.75 0.75 0.75 0.0 0.0 0.0             --num_tsteps 10 --stages_per_ts 125 --report_perf 4' % (N, liststr)
        elif app == 'hacc':
            appcmd += 'cd $HOME/allocation/hacc/testrun; '
            #appcmd += 'sbcast --compress=lz4 /project/projectdirs/m3410/applications/withoutIns/HACC_1_7/HACC /tmp/HACC; '
            appcmd += 'srun -N %d --mem=100G --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/HACC_1_7/HACC indat cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N 64 -a final -f refresh -t 16x16x8' % (N, liststr)
        elif app == 'graph500':
            appcmd += 'cd $HOME/allocation/graph500/src; export SKIP_VALIDATION=1; '
            appcmd += 'mpirun -np %d --host %s ./graph500_reference_bfs_sssp 26' % (ntasks, liststr)
        elif app == 'milc':
            appcmd += 'cd $HOME/milc_qcd-7.8.1/ks_imp_dyn/test; '
            appcmd += 'srun -N %d --mem=100G --ntasks-per-node=32 --nodelist=%s ../su3_rmd myinput.in' % (N, liststr)
        elif app == 'hpcg':
            appcmd += 'cd $HOME/allocation/hpcg/testrun; '
            appcmd += 'srun -N %d --mem=100G --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/hpcg/build/bin/HPCG --nx=64 --rt=60' % (N, liststr)
        elif app == 'hpcgDEBUG':
            appcmd += 'cd $HOME/allocation/hpcg/testrun; '
            appcmd += 'srun -N %d --mem=100G -c 1 --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/hpcg/build/bin/HPCG --nx=64 --rt=5' % (N, tasksPerNode, liststr)
        elif app == 'qmcpack':
            appcmd += 'cd $HOME/allocation/qmcpack/testrun; '
            appcmd += 'srun -N %d --mem=100G --ntasks-per-node=32 --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/qmcpack_ben/build_ben/bin/qmcpack simple-H2O.xml' % (N, liststr)
        elif app == 'miniFE':
            appcmd += 'cd $HOME/allocation/miniFE/ref/src; '
            appcmd += 'mpirun -np %d --host %s ./miniFE.x -nx 128 -ny 128 -nz 128' % (ntasks, liststr)
        appcmd += ';'
        #appcmd += ' > %s;' % writeToFile # not needed.
        out = 'startTime:' + subprocess.check_output(['date']).decode('utf-8') + '\n'
        if writeToFile is None:
            print(out)
        else:
            with open(writeToFile, 'a') as o:
                o.write(out)
        apprun = subprocess.Popen(appcmd, stdout=subprocess.PIPE, shell=True) # execute the command.
        output = apprun.communicate()[0].strip() # get execution output.
        out = output.decode('utf-8') + '\n'
        out += 'endTime:' + subprocess.check_output(['date']).decode('utf-8') + '\n\n'
        if writeToFile is None:
            print(out)
        else:
            with open(writeToFile, 'a') as o:
                o.write(out)

    def congestion(self, withCongestor, core, instance):
        '''
        obsolete
        '''
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
        '''
        obsolete
        '''
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
            allNodes = self.nodelist[i*rotate:] + self.nodelist[:i*rotate] # the order of the nodes are rotated in every iteration.
            congestNodes = [allNodes[2*x+1] for x in range(N)] # for running the congestor.
            greenNodes = allNodes[-N:] # selecting a continuous set from allNodes.
            yellowNodes = [allNodes[2*x] for x in range(N)] # selecting every other nodes.

            # run without congestor.
            self.appOnNodes(app=appName, N=N, nodes=greenNodes)
            self.appOnNodes(app=appName, N=N, nodes=yellowNodes)

            # run with congestor.
            procs = self.startGPC(instance=instance, nodes=congestNodes)
            #procs = self.startOSU(N=N, core=32, instance=instance, nodes=congestNodes) # OSU doesn't seem to create congestion.
            self.appOnNodes(app=appName, N=N, nodes=greenNodes)
            self.appOnNodes(app=appName, N=N, nodes=yellowNodes)
            print('is congestor running:')
            for j in range(instance):
                cong = procs.pop()
                print(cong.poll() == None) # print out whether the congestor is still running at the current time.
                if cong.poll() == None:
                    os.killpg(os.getpgid(cong.pid), signal.SIGTERM) # kill it.

    def startContGPC(self, nodes):
        '''
        Start a separate continualGPC() process. Use self.q for communication between the main process and the subprocess.
        '''
        self.GPCnodes = nodes
        self.q = multiprocessing.Queue()
        self.GPCchecker = multiprocessing.Process(target=self.continualGPC, args=(self.q,))
        self.GPCchecker.start()
        print('GPC checker started.')

    def continualGPC(self, q):
        '''
        The process for running GPCNET network congestor.
        Only run 1 instance of congestor.
        '''
        print(self.GPCnodes)
        N = len(self.GPCnodes)
        tpn = 68 if self.knl else 32
        ntasks = tpn*N
        cpu = 'knl' if self.knl else 'haswell'
        gpclist = self.abbrev(self.GPCnodes)
        command = ''
        #command += 'sbcast --compress=lz4 /global/homes/z/zhangyj/GPCNET/network_load_test /tmp/network_load_test; '
        command += 'srun -N %d --mem=30G --ntasks %d --nodelist=%s --ntasks-per-node=%d -C %s /global/homes/z/zhangyj/GPCNET/network_load_test > results/continualGPC_.out; ' % (N, ntasks, gpclist, tpn, cpu)
        GPCproc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        print('continual GPC started.')
        while 1:
            time.sleep(0.5)
            if not q.empty():
                message = q.get()
                if message == 'stop': # stop congestor message received.
                    if GPCproc.poll() == None: # if not finished.
                        os.killpg(os.getpgid(GPCproc.pid), signal.SIGTERM)
                        print('GPC killed in continualGPC().')
                    break
            if GPCproc.poll() != None: # if finished, restart the congestor.
                print('restarting GPC..')
                GPCproc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)

    def stopGPC(self):
        '''
        Stop the GPCNET congestor.
        may not run successfully due to srun delay problem.
        '''
        self.q.put('stop')
        #self.GPCchecker.terminate()
        self.GPCchecker.join()
        print('continualGPC() finished.')
        self.q.close()
        self.q.join_thread()
        print('GPC killed.')

    def runLDMS(self, foldername, storeNode, seconds):
        '''
        Start LDMS.
        The additional 30s to start sampler in the bash script is not counted.
        '''
        print('starting LDMS..')
        #proc = subprocess.call('./runStartLDMS.sh %s %s %d' % (foldername, storeNode, seconds), shell=True)
        proc = subprocess.call('./startLDMS.sh %s %s %d' % (foldername, storeNode, seconds), shell=True) # storeNode parameter is currently not used in startLDMS.sh.
        self.ldmsdir = '/project/projectdirs/m3231/yijia/csv/%s' % foldername # output folder.
        print('runLDMS() finish.')

    def deleteLastLine(self, f):
        '''
        Delete the last line of a file (may be incomplete).
        '''
        with open(f, 'r+', encoding = "utf-8") as file:
            # Move the pointer (similar to a cursor in a text editor) to the end of the file
            file.seek(0, os.SEEK_END)
            # This code means the following code skips the very last character in the file -
            # i.e. in the case the last line is null we delete the last line
            # and the penultimate one
            pos = file.tell() - 1
            # Read each character in the file one at a time from the penultimate
            # character going backwards, searching for a newline character
            # If we find a new line, exit the search
            while pos > 0 and file.read(1) != "\n":
                pos -= 1
                file.seek(pos, os.SEEK_SET)
            # So long as we're not at the start of the file, delete all the characters ahead
            # of this position
            if pos > 0:
                file.seek(pos, os.SEEK_SET)
                file.truncate()
        print('deleteLastLine() finished.')

    def sortCongestion(self):
        '''
        Sort idle nodes according to their network metrics.
        '''
        import pandas as pd
        import numpy as np
        print('Copying file..')
        copyfile('%s/cray_aries_r' % self.ldmsdir, '%s/temp.csv' % self.ldmsdir)
        print('Removing last line..')
        self.deleteLastLine('%s/temp.csv' % self.ldmsdir)
        print('Reading cray csv..')
        df = pd.read_csv('%s/temp.csv' % self.ldmsdir)
        print('%.1f seconds collected in total.' % (df.iloc[-1]['#Time'] - df.iloc[0]['#Time']))
        print('Calculating stall sum..')
        tiles = ['stalled_%03d (ns)' % x for x in range(48)]
        df['stalled_sum'] = df[tiles].sum(axis=1)
        print('Calculating node stall..')
        nodeCong = []
        selectTime = df[ (df['#Time'] >= self.monitorstart) & (df['#Time'] <= self.monitorend) ]
        for node in self.idleNodes:
            select = selectTime[selectTime['component_id']==node]
            if len(select) == 0:
                print('No data for node %d' % node)
                stallPerTile = -1
            elif len(select) == 1:
                print('Only one timestamp.')
                stallPerTile = -1
            else:
                stall = ( select.iloc[-1]['stalled_sum'] - select.iloc[0]['stalled_sum'] ) / ( select.iloc[-1]['#Time'] - select.iloc[0]['#Time'] )# average over time.
                nonzero = np.count_nonzero( (select.iloc[-1][tiles] - select.iloc[0][tiles]).values )
                if nonzero == 0:
                    print('All tiles have zero stall.')
                    stallPerTile = -1
                else:
                    stallPerTile = stall / nonzero
            nodeCong.append((node, stallPerTile))
        print('Replacing no-value data..')
        positive = [x[1] for x in nodeCong]
        mean = sum(positive) / len(positive)
        if mean == -1:
            pass
        else:
            for idx, pair in enumerate(nodeCong):
                if pair[1] == -1:
                    nodeCong[idx] = (pair[0], mean)
        print('Sorting nodes..')
        nodeCongPair = sorted(nodeCong, key=lambda x: x[1]) # ascending.
        print(nodeCongPair)
        return nodeCongPair

    def CADD(self, appName, iteration, congSize, appSize, appOut):
        '''
        Experiment for the Congestion-Aware Data-Driven allocation policy.
        iteration: number of iteration. In each iteration, the nodes to run congestor is re-selected randomly.
        '''
        print(subprocess.check_output(['date']).decode('utf-8'))
        jobid = os.environ['SLURM_JOB_ID']
        print('jobid: ' + str(jobid))
        #self.runLDMS(foldername='%s_%d' % (jobid, 0), storeNode='nid%05d' % self.nodelist[0], seconds=120) # for debug.
        #self.appOnNodes(app=appName, N=4, nodes=self.nodelist[1:5], writeToFile=appOut) # for debug.
        with open(appOut, 'w') as f:
            f.write('Starting..\n')

        for i in range(iteration):
            print('====================')
            print('iteration %d' % i)
            with open(appOut, 'a') as f:
                f.write('====================\n')
                f.write('iteration %d\n' % i)
            # use 1st node for this python code and LDMS storage; the rest for congestor and app.
            storeNode = 'nid%05d' % self.nodelist[0] # LDMS storage daemon node.
            availNodes = self.nodelist[1:]
            congNodes = random.sample(availNodes, congSize) # randomly selecting nodes for the congestor.
            self.idleNodes = [x for x in availNodes if x not in congNodes]
            numIdle = len(self.idleNodes)

            # get idle nodes that noShare/share with congestor.
            noSharing, sharing, idleNeighbor = [], [], []
            for n in self.idleNodes:
                n4 = (n//4)*4
                router = {n4, n4+1, n4+2, n4+3}
                router.remove(n)
                neighbor = len(router.intersection(set(self.idleNodes))) # 0-3.
                idleNeighbor.append((n,neighbor))
                if not router.intersection(set(congNodes)): # intersection is empty.
                    noSharing.append((n,neighbor))
                else:
                    sharing.append(n)
            noSharing.sort(key=lambda x: x[1], reverse=True)
            idleNeighbor.sort(key=lambda x: x[1], reverse=True)
            # omniscient policy: prioritize nodes without sharing router with congestor, and in that case also prioritize nodes with more neighbors.
            numGood = len(noSharing)
            if numGood >= appSize:
                omniAllocated = [noSharing[x][0] for x in range(appSize)]
            else:
                omniAllocated = [noSharing[x][0] for x in range(numGood)]
                omniAllocated += [sharing[x] for x in range(appSize-numGood)]
            # cori policy: prioritize nodes with more idle neighbors.
            coriAllocated = [idleNeighbor[x][0] for x in range(appSize)]

            print('Congestor nodes:')
            print(congNodes)
            self.startContGPC(nodes=congNodes)

            self.monitorstart = int(time.time())
            if i == 0:
                self.runLDMS(foldername='%s_%d' % (jobid, i), storeNode=storeNode, seconds=60)
            else:
                time.sleep(60) # don't start LDMS again.
            self.monitorend = int(time.time())
            print('Monitor end timestamp: %d' % self.monitorend)

            print('Starting sortCongestion()..')
            nodeCongPair = self.sortCongestion() # sort idle nodes from low to high congestion according to their stall-per-second.
            nodeCongDict = {}
            for pair in nodeCongPair:
                nodeCongDict[pair[0]] = pair[1]

            for policy in ['cadd','anticadd','cori','omni','random']:
                if policy == 'cadd':
                    nodes = [nodeCongPair[x][0] for x in range(appSize)]
                    stall = sum([nodeCongPair[x][1] for x in range(appSize)]) / appSize
                    caddstall = stall
                elif policy == 'anticadd':
                    nodes = [nodeCongPair[x][0] for x in range(numIdle-appSize, numIdle)]
                    stall = sum([nodeCongPair[x][1] for x in range(numIdle-appSize, numIdle)]) / appSize
                elif policy == 'cori':
                    nodes = coriAllocated
                    stall = sum([nodeCongDict[x] for x in nodes]) / appSize
                elif policy == 'omni':
                    nodes = omniAllocated
                    stall = sum([nodeCongDict[x] for x in nodes]) / appSize
                elif policy == 'random':
                    nodes = random.sample(self.idleNodes, appSize)
                    stall = sum([nodeCongDict[x] for x in nodes]) / appSize
                print('%s avg stalls: %.f' % (policy, stall))
                print('Ratio current/cadd: %.3f' % (stall/(caddstall+0.00001)))

                print('Run job in %s policy.' % policy)
                with open(appOut, 'a') as f:
                    f.write('Starting job in %s policy..\n' % policy)
                self.appOnNodes(app=appName, N=appSize, nodes=nodes, writeToFile=appOut)

            self.stopGPC()
            time.sleep(5)

            # run without congestor.
            policy = 'noCong'
            nodes = [nodeCongPair[x][0] for x in range(appSize)]
            print('Run job in %s policy.' % policy)
            with open(appOut, 'a') as f:
                f.write('Starting job in %s policy..\n' % policy)
            self.appOnNodes(app=appName, N=appSize, nodes=nodes, writeToFile=appOut)

            print('Iteration end.')
            print()

if __name__ == '__main__':
    main()
