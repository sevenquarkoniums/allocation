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
    #w.congestor() # My GPCNET takes 5 min on 64 knl nodes.
    #w.allocation(instance=int(sys.argv[1]))
    #w.appOnNodes(app='graph500', N=32, nodes=w.nodelist) # used to test application run.
    #w.fixAllocation(appName='lammps', iteration=10, instance=5)
    #w.NeDD(appName='graph500', iteration=10, congSize=64, appSize=32, appOut='NeDDjob_graph500_201.out')
    w.NeDDTwo(app1='qmcpack', app2='miniMD', iteration=10, congSize=64, appSize=32, out1='NeDDTwojob_qmcpack_miniMD_1.out', out2='NeDDTwojob_qmcpack_miniMD_2.out')
    #w.congestorLDMS()
    #w.testLDMS()

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

    def appOnNodes(self, app, N, nodes, writeToFile=None, waitToEnd=True):
        '''
        Run app on some nodes.
        '''
        tasksPerNode = 68 if self.knl else 32
        if app in ['milc','hacc','graph500']:
            tasksPerNode = 64
        ntasks = N * tasksPerNode
        liststr = ','.join(['nid{0:05d}'.format(y) for y in nodes]) # nodelist for srun or mpirun format.
        print('app-node list:')
        print(nodes)
        # create bash command for cori.
        appcmd = 'module load openmpi; '
        if app == 'miniMD':
            appcmd += 'cd $HOME/miniMD/ref; '
            #appcmd += 'mpirun -np %d --host %s $HOME/miniMD/ref/miniMD_openmpi -n 160000' % (ntasks, liststr) # may trigger LDMS daemon problem by mpirun.
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%d --nodelist=%s $HOME/miniMD/ref/miniMD_openmpi -n 80000' % (N, tasksPerNode, liststr)
        elif app == 'nekbone':
            appcmd += 'cd $HOME/allocation/nekbone/Nekbone/test/example1; '
            appcmd += 'mpirun -np %d --host %s ./nekbone ex1' % (ntasks, liststr)
        elif app == 'lammps':
            appcmd += 'cd $HOME/allocation/lammps/testrun; '
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/LAMMPS/src/LAMMPS -in in.vacf.2d' % (N, tasksPerNode, liststr)
        elif app == 'miniamr':
            appcmd += 'cd $HOME/allocation/miniamr/testrun; '
            #appcmd += 'sbcast --compress=lz4 /project/projectdirs/m3410/applications/withoutIns/miniAMR_1.0_all/miniAMR_ref/miniAMR.x /tmp/miniAMR.x; '
            appcmd += 'srun -N %d --mem=80G --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/miniAMR_1.0_all/miniAMR_ref/miniAMR.x --num_refine 4 --max_blocks 5000 --init_x 1 --init_y 1 --init_z 1             --npx 17 --npy 16 --npz 8 --nx 6 --ny 6 --nz 6 --num_objects 2             --object 2 0 -1.10 -1.10 -1.10 0.030 0.030 0.030 1.5 1.5 1.5 0.0 0.0 0.0             --object 2 0 0.5 0.5 1.76 0.0 0.0 -0.025 0.75 0.75 0.75 0.0 0.0 0.0             --num_tsteps 10 --stages_per_ts 125 --report_perf 4' % (N, tasksPerNode, liststr)# npx * npy * npz better be the same as cores. # memory should be larger.
        elif app == 'hacc':
            appcmd += 'cd $HOME/allocation/hacc/testrun; '
            #appcmd += 'sbcast --compress=lz4 /project/projectdirs/m3410/applications/withoutIns/HACC_1_7/HACC /tmp/HACC; '
            appcmd += 'srun -N %d --mem=80G --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/HACC_1_7/HACC indat cmbM000.tf m000 INIT ALL_TO_ALL -w -R -N %d -a final -f refresh -t 16x16x8' % (N, tasksPerNode, liststr, N)
        elif app == 'graph500':
            appcmd += 'cd $HOME/allocation/graph500/src; export SKIP_VALIDATION=1; '
            #appcmd += 'mpirun -np %d --host %s ./graph500_reference_bfs_sssp 26' % (ntasks, liststr)
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%d --nodelist=%s ./graph500_reference_bfs_sssp 26' % (N, tasksPerNode, liststr) # task need to be power of 2.
        elif app == 'milc':
            appcmd += 'cd $HOME/milc_qcd-7.8.1/ks_imp_dyn/test; '
            if self.knl:
                myinput = 'myinput_knl.in'
            else:
                myinput = 'myinput.in'
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%s --nodelist=%s ../su3_rmd %s' % (N, tasksPerNode, liststr, myinput)
        elif app == 'hpcg':
            appcmd += 'cd $HOME/allocation/hpcg/testrun; '
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/hpcg/build/bin/HPCG --nx=64 --rt=50' % (N, tasksPerNode, liststr)
        elif app == 'hpcgDEBUG':
            appcmd += 'cd $HOME/allocation/hpcg/testrun; '
            appcmd += 'srun -N %d --mem=100G -c 1 --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/hpcg/build/bin/HPCG --nx=64 --rt=5' % (N, tasksPerNode, liststr)
        elif app == 'qmcpack':
            appcmd += 'cd $HOME/allocation/qmcpack/testrun; '
            appcmd += 'srun -N %d --mem=50G --ntasks-per-node=%d --nodelist=%s /project/projectdirs/m3410/applications/withoutIns/qmcpack_ben/build_ben/bin/qmcpack simple-H2O.xml' % (N, tasksPerNode, liststr)
        elif app == 'miniFE':
            appcmd += 'cd $HOME/allocation/miniFE/ref/src; '
            appcmd += 'mpirun -np %d --host %s ./miniFE.x -nx 128 -ny 128 -nz 128' % (ntasks, liststr)
        appcmd += ';'
        #appcmd += ' > %s;' % writeToFile # not needed.
        if waitToEnd:
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
        else:
            apprun = subprocess.Popen(appcmd, stdout=writeToFile, shell=True) # execute the command.
        return apprun

    def congestor(self):
        '''
        Start GPCNET congestor.
        '''
        GPCnodes = self.nodelist
        print(GPCnodes)
        N = len(GPCnodes)
        tpn = 68 if self.knl else 32
        ntasks = tpn*N
        cpu = 'knl' if self.knl else 'haswell'
        gpclist = self.abbrev(GPCnodes)
        command = ''
        command += 'srun -N %d --mem=30G --ntasks %d --nodelist=%s --ntasks-per-node=%d -C %s /global/homes/z/zhangyj/GPCNET/network_load_test > results/testGPC.out; ' % (N, ntasks, gpclist, tpn, cpu)
        GPCproc = subprocess.Popen(command, shell=True, preexec_fn=os.setsid)
        print('GPC started.')
        print(subprocess.check_output(['date']).decode('utf-8'))
        while 1:
            time.sleep(0.5)
            if GPCproc.poll() != None: # if finished, restart the congestor.
                print('GPC finished.')
                print(subprocess.check_output(['date']).decode('utf-8'))
                break

    def congestorLDMS(self):
        '''
        Start congestor with LDMS.
        '''
        print(subprocess.check_output(['date']).decode('utf-8'))
        jobid = os.environ['SLURM_JOB_ID']
        print('jobid: ' + str(jobid))
        print(self.nodelist)
        storeNode = 'nid%05d' % self.nodelist[0] # LDMS storage daemon node.
        self.runLDMS(foldername='%s' % (jobid), storeNode=storeNode, seconds=60)

        GPCnodes = self.nodelist[1:]
        N = len(GPCnodes)
        tpn = 68 if self.knl else 32
        ntasks = tpn*N
        cpu = 'knl' if self.knl else 'haswell'
        gpclist = self.abbrev(GPCnodes)
        GPCproc = []
        for i in range(1):
            command = 'export MPICH_GNI_FMA_SHARING=ENABLED;'
            command += 'srun -N %d --mem=20G --ntasks %d --nodelist=%s --ntasks-per-node=%d -C %s /global/homes/z/zhangyj/GPCNET/network_load_test > results/testGPC%d.out;' % (N, ntasks, gpclist, tpn, cpu, i)
            GPCproc.append( subprocess.Popen(command, shell=True, preexec_fn=os.setsid) )
            print('GPC %d started.' % i)
        print(subprocess.check_output(['date']).decode('utf-8'))
        while 1:
            time.sleep(0.5)
            if GPCproc[-1].poll() != None: # if finished, restart the congestor.
                print('GPC -1 finished.')
                end = int(time.time())
                print('end timestamp: %d' % end)
                time.sleep(120)
                break

    def testLDMS(self):
        print(subprocess.check_output(['date']).decode('utf-8'))
        jobid = os.environ['SLURM_JOB_ID']
        print('jobid: ' + str(jobid))
        print(self.nodelist)
        storeNode = 'nid%05d' % self.nodelist[0] # LDMS storage daemon node.
        self.runLDMS(foldername='%s' % (jobid), storeNode=storeNode, seconds=60)
        time.sleep(60)
        print(subprocess.check_output(['date']).decode('utf-8'))

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
        Each GPC instance needs 10~20G memory.
        '''
        print(self.GPCnodes)
        N = len(self.GPCnodes)
        tpn = 68 if self.knl else 32
        ntasks = tpn*N
        cpu = 'knl' if self.knl else 'haswell'
        gpclist = self.abbrev(self.GPCnodes)
        command = ''
        #command += 'sbcast --compress=lz4 /global/homes/z/zhangyj/GPCNET/network_load_test /tmp/network_load_test; '
        command += 'srun -N %d --mem=20G --ntasks %d --nodelist=%s --ntasks-per-node=%d -C %s /global/homes/z/zhangyj/GPCNET/network_load_test > results/continualGPC_.out; ' % (N, ntasks, gpclist, tpn, cpu)
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

    def NeDD(self, appName, iteration, congSize, appSize, appOut):
        '''
        Experiment for the Network-Data-Driven allocation policy.
        iteration: number of iteration. In each iteration, the nodes to run congestor is re-selected randomly.
        '''
        monitorTime = 120 # 60 may not be enough to pass the network test period.
        print(subprocess.check_output(['date']).decode('utf-8'))
        jobid = os.environ['SLURM_JOB_ID']
        print('jobid: ' + str(jobid))
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

            # start GPCNET.
            print('Congestor nodes:')
            print(congNodes)
            self.startContGPC(nodes=congNodes)

            # monitor LDMS data.
            self.monitorstart = int(time.time())
            if i == 0: # don't start LDMS again.
                self.runLDMS(foldername='%s_%d' % (jobid, i), storeNode=storeNode, seconds=10)
            time.sleep(monitorTime)
            self.monitorend = int(time.time())
            print('Monitor end timestamp: %d' % self.monitorend)

            print('Starting sortCongestion()..')
            nodeCongPair = self.sortCongestion() # sort idle nodes from low to high congestion according to their stall-per-second.
            nodeCongDict = {}
            for pair in nodeCongPair:
                nodeCongDict[pair[0]] = pair[1]
                
            # get idle nodes that noShare/share with congestor.
            nodeInfo = []
            for n in self.idleNodes:
                n4 = (n//4)*4
                router = {n4, n4+1, n4+2, n4+3}
                router.remove(n)
                neighbor = len(router.intersection(set(self.idleNodes))) # idle neighbor number: 0-3.
                congNeighbor = len(router.intersection(set(congNodes)))
                nodeInfo.append( (n,neighbor,congNeighbor,nodeCongDict[n]) )
            nodeInfo2 = nodeInfo.copy()
            nodeInfo3 = nodeInfo.copy()
            
            # NeDD policy: select routers by NIC traffic.
            # so it will first avoid congestor, then choose nodes with more idle neighbors.
            nodeInfo.sort(key=lambda x: x[1], reverse=True) # high to low.
            nodeInfo.sort(key=lambda x: x[2], reverse=False) # low to high. This is prioritized, so sorted later.
            print('nedd nodeInfo:')
            print(nodeInfo)
            neddAlloc = [nodeInfo[x][0] for x in range(appSize)]
            
            # Anti-NeDD.
            antiAlloc = [nodeInfo[x][0] for x in range(numIdle-appSize, numIdle)]
            
            # Fewer switches. Close to Cori's allocation.
            nodeInfo2.sort(key=lambda x: x[1], reverse=True) # high to low.
            fewerAlloc = [nodeInfo2[x][0] for x in range(appSize)]
            
            # Lower router stall count.
            nodeInfo3.sort(key=lambda x: x[3], reverse=False) # low to high.
            lowerAlloc = [nodeInfo3[x][0] for x in range(appSize)]

            for policy in ['nedd','lowerRouterStall','fewerSwitch','random','antinedd']:
                if policy == 'nedd':
                    nodes = neddAlloc
                elif policy == 'antinedd':
                    nodes = antiAlloc
                elif policy == 'fewerSwitch':
                    nodes = fewerAlloc
                elif policy == 'lowerRouterStall':
                    nodes = lowerAlloc
                elif policy == 'random':
                    nodes = random.sample(self.idleNodes, appSize)
                print('Run job in %s policy..' % policy)
                with open(appOut, 'a') as f:
                    f.write('Starting job in %s policy..\n' % policy)
                self.appOnNodes(app=appName, N=appSize, nodes=nodes, writeToFile=appOut)
                print()

            self.stopGPC()
            time.sleep(5)

            # run without congestor.
            #policy = 'neddNoCong'
            #nodes = neddAlloc
            #print('Run job in %s policy..' % policy)
            #with open(appOut, 'a') as f:
            #    f.write('Starting job in %s policy..\n' % policy)
            #self.appOnNodes(app=appName, N=appSize, nodes=nodes, writeToFile=appOut)

            policy = 'fewerNoCong'
            nodes = fewerAlloc
            print('Run job in %s policy.' % policy)
            with open(appOut, 'a') as f:
                f.write('Starting job in %s policy..\n' % policy)
            self.appOnNodes(app=appName, N=appSize, nodes=nodes, writeToFile=appOut)

            print('Iteration end.')
            print()

    def NeDDTwo(self, app1, app2, iteration, congSize, appSize, out1, out2):
        '''
        Two-job Experiment for the Network-Data-Driven allocation policy.
        iteration: number of iteration. In each iteration, the nodes to run congestor is re-selected randomly.
        '''
        monitorTime = 120 # 60 may not be enough to pass the network test period.
        print(subprocess.check_output(['date']).decode('utf-8'))
        jobid = os.environ['SLURM_JOB_ID']
        print('jobid: ' + str(jobid))
        with open(out1, 'w') as f:
            f.write('Starting..\n')
        with open(out2, 'w') as f:
            f.write('Starting..\n')
            
        for i in range(iteration):
            print('====================')
            print('iteration %d' % i)
            with open(out1, 'a') as f:
                f.write('====================\n')
                f.write('iteration %d\n' % i)
            with open(out2, 'a') as f:
                f.write('====================\n')
                f.write('iteration %d\n' % i)
            # use 1st node for this python code and LDMS storage; the rest for congestor and app.
            storeNode = 'nid%05d' % self.nodelist[0] # LDMS storage daemon node.
            availNodes = self.nodelist[1:]
            congNodes = random.sample(availNodes, congSize) # randomly selecting nodes for the congestor.
            self.idleNodes = [x for x in availNodes if x not in congNodes]
            self.numIdle = len(self.idleNodes)

            # start GPCNET.
            print('Congestor nodes:')
            print(congNodes)
            self.startContGPC(nodes=congNodes)

            # monitor LDMS data.
            self.monitorstart = int(time.time())
            if i == 0: # don't start LDMS again.
                self.runLDMS(foldername='%s_%d' % (jobid, i), storeNode=storeNode, seconds=10)
            time.sleep(monitorTime)
            self.monitorend = int(time.time())
            print('Monitor end timestamp: %d' % self.monitorend)

            print('Starting sortCongestion()..')
            nodeCongPair = self.sortCongestion() # sort idle nodes from low to high congestion according to their stall-per-second.
            nodeCongDict = {}
            for pair in nodeCongPair:
                nodeCongDict[pair[0]] = pair[1]
                
            # get idle nodes that noShare/share with congestor.
            nodeInfo = []
            for n in self.idleNodes:
                n4 = (n//4)*4
                router = {n4, n4+1, n4+2, n4+3}
                router.remove(n)
                neighbor = len(router.intersection(set(self.idleNodes))) # idle neighbor number: 0-3.
                congNeighbor = len(router.intersection(set(congNodes)))
                nodeInfo.append( (n,neighbor,congNeighbor,nodeCongDict[n]) )
            nodeInfo2 = nodeInfo.copy()
            nodeInfo3 = nodeInfo.copy()
            
            # NeDD.
            nodeInfo.sort(key=lambda x: x[1], reverse=True) # high to low.
            nodeInfo.sort(key=lambda x: x[2], reverse=False) # low to high. This is prioritized, so sorted later.
            print('nedd nodeInfo:')
            print(nodeInfo)
            #neddAlloc1 = [nodeInfo[x][0] for x in range(appSize)] # for network-sensitive job.
            neddAlloc1 = [nodeInfo[x][0] for x in range(self.numIdle-appSize, self.numIdle)] # for network-insensitive job.
            
            # Anti-NeDD.
            antiAlloc1 = [nodeInfo[x][0] for x in range(self.numIdle-appSize, self.numIdle)]
            
            # Fewer switches. Close to Cori's allocation.
            nodeInfo2.sort(key=lambda x: x[1], reverse=True) # high to low.
            fewerAlloc1 = [nodeInfo2[x][0] for x in range(appSize)]
            
            # Lower router stall count.
            nodeInfo3.sort(key=lambda x: x[3], reverse=False) # low to high.
            lowerAlloc1 = [nodeInfo3[x][0] for x in range(appSize)]
            
            randomAlloc1 = random.sample(self.idleNodes, appSize)

            for policy in ['nedd','lowerRouterStall','fewerSwitch','random','antinedd']:
                if policy == 'nedd':
                    nodes1 = neddAlloc1
                elif policy == 'antinedd':
                    nodes1 = antiAlloc1
                elif policy == 'fewerSwitch':
                    nodes1 = fewerAlloc1
                elif policy == 'lowerRouterStall':
                    nodes1 = lowerAlloc1
                elif policy == 'random':
                    nodes1 = randomAlloc1
                
                # get idle node information after the 1st job is allocated.
                idleNodes = [n for n in self.idleNodes if n not in nodes1]
                numIdle = self.numIdle - appSize
                nodeInfo = []
                for n in idleNodes:
                    n4 = (n//4)*4
                    router = {n4, n4+1, n4+2, n4+3}
                    router.remove(n)
                    neighbor = len(router.intersection(set(idleNodes))) # idle neighbor number: 0-3.
                    congNeighbor = len(router.intersection(set(congNodes)))
                    nodeInfo.append( (n,neighbor,congNeighbor,nodeCongDict[n]) )
                nodeInfo2 = nodeInfo.copy()
                nodeInfo3 = nodeInfo.copy()

                # NeDD.
                nodeInfo.sort(key=lambda x: x[1], reverse=True) # high to low.
                nodeInfo.sort(key=lambda x: x[2], reverse=False) # low to high. This is prioritized, so sorted later.
                neddAlloc2 = [nodeInfo[x][0] for x in range(appSize)]

                # Anti-NeDD.
                antiAlloc2 = [nodeInfo[x][0] for x in range(numIdle-appSize, numIdle)]

                # Fewer switches. Close to Cori's allocation.
                nodeInfo2.sort(key=lambda x: x[1], reverse=True) # high to low.
                fewerAlloc2 = [nodeInfo2[x][0] for x in range(appSize)]

                # Lower router stall count.
                nodeInfo3.sort(key=lambda x: x[3], reverse=False) # low to high.
                lowerAlloc2 = [nodeInfo3[x][0] for x in range(appSize)]

                randomAlloc2 = random.sample(idleNodes, appSize)
                
                if policy == 'nedd':
                    nodes2 = neddAlloc2
                elif policy == 'antinedd':
                    nodes2 = antiAlloc2
                elif policy == 'fewerSwitch':
                    nodes2 = fewerAlloc2
                elif policy == 'lowerRouterStall':
                    nodes2 = lowerAlloc2
                elif policy == 'random':
                    nodes2 = randomAlloc2
                    
                print('Run job %s in %s policy..' % (app1, policy))
                outfile1 = open(out1, 'a')
                outfile1.write('Starting job in %s policy..\n' % policy)
                date1 = 'startTime:' + subprocess.check_output(['date']).decode('utf-8') + '\n'
                outfile1.write(date1)
                outfile1.flush()
                runObj = self.appOnNodes(app=app1, N=appSize, nodes=nodes1, writeToFile=outfile1, waitToEnd=0)

                print('Run job %s in %s policy..' % (app2, policy))
                with open(out2, 'a') as f:
                    f.write('Starting job in %s policy..\n' % policy)
                self.appOnNodes(app=app2, N=appSize, nodes=nodes2, writeToFile=out2, waitToEnd=1)
                print()
                
                while runObj.poll() == None: # if not finished.
                    time.sleep(5)
                outfile1.flush()
                date1 = 'endTime:' + subprocess.check_output(['date']).decode('utf-8') + '\n\n'
                outfile1.write(date1)
                outfile1.close()

            self.stopGPC()
            time.sleep(5)
            print('Iteration end.')
            print()

if __name__ == '__main__':
    main()
