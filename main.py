#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import multiprocessing
import pickle
import datetime
from io import StringIO
import pytz
pacific = pytz.timezone('US/Pacific')
import re

'''
Warning: do not use .replace(tzinfo=pacific) to replace timezone, which has a fault!
'''

def main():
    al = analysis(outputName='temp.csv', logExist=1)
        # do change queryTime when adjusting range.
    #al.runtime()
    #al.process(mode='rtrstall', counterSaved=0, saveFolder='counterOSU')
    #al.analyzeAlloc()
    al.processFix(app='nekbone')
    #al.calcNode()

def getfiles(path):
    # get all the files with full path.
    fileList = []
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            fileList.append(filepath)
    return fileList

def unix2string(unixTime):
    thisTime = pytz.utc.localize(datetime.datetime.utcfromtimestamp(unixTime)).astimezone(tz=pacific)
    timestr = thisTime.strftime('%Y-%m-%d %H:%M:%S')
    return timestr

class ldms:
    def __init__(self, logExist=False):
        self.allrtrmetrics = ['flit','stallvc','stallrowbus']
        nprocs = multiprocessing.cpu_count()
        self.pool = multiprocessing.Pool(nprocs)
        if logExist:
            # needs to have rtr_log_starts.pkl file. Otherwise, use logExist=False.
            print('Reading aggs..')
            with open('rtr_log_starts.pkl', 'rb') as inputFile:
                self.rtr_log_starts = pickle.load(inputFile)
            with open('nic_log_starts.pkl', 'rb') as inputFile:
                self.nic_log_starts = pickle.load(inputFile)
        else:
            print('Building aggs..')
            aggs = self.getAggregatorPaths()
            self.rtr_log_starts, rtr_log_mods = self.buildLDMSFileDicts(aggs, 'metric_set_rtr')
            self.nic_log_starts, nic_log_mods = self.buildLDMSFileDicts(aggs, 'metric_set_nic')
            with open('rtr_log_starts.pkl', 'wb') as output:
                pickle.dump(self.rtr_log_starts, output, pickle.HIGHEST_PROTOCOL)
            with open('nic_log_starts.pkl', 'wb') as output:
                pickle.dump(self.nic_log_starts, output, pickle.HIGHEST_PROTOCOL)

    def getAggregatorPaths(self):
        machine = os.environ.get("NERSC_HOST")
        if machine == "edison":
            basedir = "/scratch1/ovis/csv"
        elif machine == "cori":
            basedir = "/global/cscratch1/ovis/csv"
        elif machine == "gerty":
            basedir = "/global/gscratch1/ovis/"
        else:
            print("getAggregatorPaths(): Didn't recognize $NERSC_HOST.")
            return[]
        return [ os.path.join(basedir, name) for name in os.listdir(basedir) if os.path.isdir(os.path.join(basedir, name)) ]

    def buildLDMSFileDicts(self, aggs, s="metric_set_rtr"):
        '''
        Given a dictionary of LDMS aggregator directories
        Create multi level dictionaries of start times and modification times -> files
        Files contain the string s.
        '''
        #Dictionary which contains a list of files for each aggregator
        files = {}
        for p in self.getAggregatorPaths():
            ftmp = os.listdir(path=p)
            files[p] = [os.path.join(p,f) for f in ftmp if s in f and "HEADER" not in f and os.stat(os.path.join(p,f)).st_size != 0]
                # Empty files are ignored.
        #This uses a two-level dictionary,
        #Dictionary of aggregators -> dictionary of start and stop times for the aggregators files
        #For every file, add it to a dicts of start and stop (modification) times
        log_starts = {}
        log_mods = {}
        for agg in aggs:
            print(str(agg))
            log_starts[agg] = {}
            log_mods[agg] = {}
            for f in files[agg]:
                ts = self.getStartTime(f)
                tm = self.getModTime(f)
                if ts in log_starts[agg]:
                    log_starts[agg][ts].append(f)
                else:
                    log_starts[agg][ts] = [f]
                if tm in log_mods[agg]:
                    log_mods[agg][tm].append(f)
                else:
                    log_mods[agg][tm] = [f]

        return (log_starts, log_mods)

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

    def getCName(self, nid):
        '''
        Get the name of the node. Can be used to see if nodes are on the same router.
        '''
        nid = nid
        if isinstance( nid, str ):
            nid = nid.lstrip("nid0*")
            nid = int(nid)
        elif not isinstance( nid, int ):
            print("Function getCName expects an int or string for input.")
            return None
        rowsize = 12*192
        row = np.floor(nid/rowsize)
        remainder = nid-(row*rowsize)
        column = np.floor(remainder/192)
        cage = np.floor((nid%384)/64)%3
        slot = np.floor(nid/4)%16
        nic = nid%4
        # Just using 0 since 4 node per router
        #Using "a" for node rather than "n" to match counter data.
        return("c{}-{}c{}s{}a{}".format(int(column),int(row),int(cage),int(slot),int(nic)))

    def getStartTime(self, filename):
        '''
        get the (start) Timestamp for a LDMS filename (or full filepath)
        '''
        return pd.Timestamp(int(filename.split(".")[-1]), unit="s", tz="US/Pacific")

    def getModTime(self, filepath):
        '''
        get the last time the file was modified (need full path)
        '''
        return pd.Timestamp(os.path.getmtime(filepath), unit='s', tz="US/Pacific")

    def subsetLinesToDF(self, fname, start_t, end_t):
        #print("Extracting from {}".format(fname))
        #fdate = fname.split(".")[-1]
        #print(pd.datetime.fromtimestamp(int(fdate)))
        with open(fname, 'r') as f:
            sz = os.path.getsize(fname)
            f.seek(0)
            hline = f.readline()
            headerBytes = len(hline)
            bstart = self.bstFileNear(headerBytes, sz, start_t-1, f)
            bend = self.bstFileNear(headerBytes, sz, end_t+1, f)
            if bstart==bend:
                return pd.DataFrame()
            hline = hline[:-1]
            hline = hline.split(',')
            f.seek(bstart)
            tmpstr = f.read(bend-bstart)
            try:
                tmpstr = tmpstr.split('\n', 1)[1]
            except Exception as e:
                print(e)
                print(tmpstr)
                print("No data found in this file {}\n while reading bytes: {} to {}".format(fname, bstart, bend))
                return pd.DataFrame()
            tmpstr = tmpstr.rsplit('\n', 1)[0]
            dftest = pd.read_csv(StringIO(tmpstr), names=hline)
        return dftest

    def bstFileNear(self, left, right, time_t, f, num_bytes=12000, field=0):
        #base case
        f.seek(left)
        #only one sample left to search
        if (right-left)/2 < num_bytes:
            return left
        #sanity check boundary case
        sample = f.read(num_bytes)
        tmp = sample.split("\n")
        if len(tmp) > 1:
            tmp = tmp[1]
        else:
            tmp = tmp[0]
        try:
            tmp = tmp.split(',')[0]
            tmp = float(tmp)
        except Exception as e:
            print(e)
            print(sample)
            print("can't split sample: {}".format(tmp))
        if tmp > time_t:
            #print("Either multi-file or we overshot due to skew")
            return left
        f.seek(right-num_bytes)
        sample = f.read(num_bytes)
        tmp = sample.split("\n")
        if len(tmp) > 1:
            tmp = tmp[1]
        else:
            tmp = tmp[0]
        try:
            tmp = tmp.split(',')[0]
            tmp = float(tmp)
        except Exception as e:
            print(e)
            print(sample)
            print("can't split sample: {}".format(tmp))
        if tmp < time_t:
            #print("May have undershot time due to skew")
            return right
        mid = (right-left)//2+left
        f.seek(mid)
        sample = f.read(num_bytes)
        try:
            sample = sample.split("\n")[1]
        except:
            print("can't split sample: {}".format(sample))
            print("f.seek({})".format(mid))
            return None
        if float(sample.split(',')[0]) >= time_t:
            return self.bstFileNear(left, mid, time_t, f)
        else:
            return self.bstFileNear(mid, right, time_t, f)

    def findRtrStartFilesV2(self, start_time, file_start_dict):
        start_files = []
        for agg in file_start_dict.keys():
            for k in sorted(file_start_dict[agg].keys(), reverse=True):
                #There is a check in here to make sure the file is not too old
                #This is needed for aggregators that are no longer running
                if k <= start_time and k >= start_time-pd.Timedelta("25H") :
                #if k < start_time:
                    #Could be multiple files that started at this time
                    filematches = file_start_dict[agg][k]
                    start_files += filematches
        return start_files

    def findRtrEndFilesV2(self, end_time, file_mod_dict):
        end_files = []
        for agg in file_mod_dict.keys():
            for k in sorted(file_mod_dict[agg].keys()):
                if k >= end_time and k <= end_time+pd.Timedelta("25H"):# 25H is conservative, but working.
                #if k > end_time:
                    #Could be multiple files that started at this time
                    filematches = file_mod_dict[agg][k]
                    end_files += filematches
        return end_files
                
    def fetchData(self, datetimeStart, datetimeEnd, source):
        '''
        Get all the nic or rtr for certain time range.
        '''
        #print('Fetching data..')
        startFiles = self.findRtrStartFilesV2(datetimeStart, self.rtr_log_starts if source=='rtr' else self.nic_log_starts)
        endFiles = self.findRtrEndFilesV2(datetimeEnd, self.rtr_log_starts if source=='rtr' else self.nic_log_starts)
        files = list( set().union(startFiles, endFiles) )
        unixTime = datetimeStart.value // 10**9
        unixEnd = datetimeEnd.value // 10**9
        pargs = [(f, unixTime, unixEnd) for f in files]
        fetched = self.pool.starmap(self.subsetLinesToDF, pargs)
        if len(fetched) == 0:
            print('No data for this time %s.' % str(datetimeStart))
            return None
        queryData = pd.concat(fetched, ignore_index=True)
        return queryData

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict['pool']
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    def selectRouter(self, data, routers):
        '''
        Only select the routers we are interested in.
        '''
        #print('Selecting routers..')
        sel = data[data['aries_rtr_id'].isin(routers)].copy()
        return sel

    def removeDuplicate(self, data):
        '''
        component_id is the producer name.
        Only select the producer with a smaller node id.
        A few routers are only collected by one producer.
        '''
        #print('Removing duplicates..')
        producers = data[['component_id','aries_rtr_id']].copy()
        #print('All producer number: %d' % len(producers['component_id'].unique()))
        selectedProducer = list( producers.groupby(['aries_rtr_id']).min()['component_id'] )
            # all the 'aries_rtr_id' fields end with 'a0'.
        numSelectedProducer = len(selectedProducer)
        prodnum = len(selectedProducer)
        print('Selected producer number: %d' % prodnum)
        rmDup = data[data['component_id'].isin(selectedProducer)].copy()
        jump = 1 if len(selectedProducer)==0 else 0
        if jump:
            print('### No counter data for this run ###')
        return rmDup, jump, numSelectedProducer, prodnum

    def getDiff(self, data):
        #print('Calculating the difference..')
        data['HTime'] = data['#Time'].apply(unix2string)
        #print(data.iloc[0][['#Time','HTime']])
        #data['HTime'] = pd.to_datetime(data['#Time'], unit='s') # this will be converted to utc time, and tz cannot be set.
        nonAccumCol = ['HTime','Time_usec','ProducerName','component_id']
        nonAccum = data[nonAccumCol]
        keep = data[['aries_rtr_id']]
        tobeDiff = data.drop(nonAccumCol, axis=1, inplace=False)
        diffed = tobeDiff.groupby('aries_rtr_id').diff()
        combine = pd.concat([nonAccum, keep, diffed], axis=1)
        diff = combine.dropna(axis=0, how='any', inplace=False)
        return diff

    def regularize(self, data):
        '''
        Scale every metric to be a value for exactly per second.
        '''
        #print('Regularizing the values..')
        noDivCol = ['HTime','Time_usec','ProducerName','component_id','aries_rtr_id']
        metrics = [x for x in data.columns if x not in noDivCol]
        metrics.remove('#Time')# This #Time column is already the differentiated one.
        noDiv = data[noDivCol]
        div = data[metrics].div(data['#Time'], axis=0)
        regu = pd.concat([noDiv, div], axis=1)
        regu['#Time'] = regu['HTime']
        #regu['#Time'] = regu['HTime'].apply(lambda x: x - pd.Timedelta(microseconds=x.microsecond))
        return regu

    def calcNodeStall(self, data, nodes):
        nodevalue = []
        for node in nodes:
            cname = self.getCName(node)
            thisRouter = data[data['aries_rtr_id'] == cname[:-1] + '0']
            if len(thisRouter) > 0:
                thisNodeRatio = thisRouter['AR_NL_PRF_REQ_PTILES_TO_NIC_%s_STALLED' % cname[-1]].mean()
                nodevalue.append(thisNodeRatio)
        if len(nodevalue) > 0:
            avg = sum(nodevalue)/len(nodevalue)
        else:
            avg = -1
        return avg

    def calcRouterStall(self, data):
        '''
        Calculate the avg stall of the 40 tiles in all the routers.
        Averaged on 3 dimensions: time, routers, ports.
        Missing data are replaced by zero.
        '''
        Metrics = []
        for r in range(5):
            for c in range(8):
                #data['VCsum_%d_%d' % (r,c)] = data[['AR_RTR_%d_%d_INQ_PRF_INCOMING_FLIT_VC%d' % (r,c,v) for v in range(8)]].sum(axis=1)
                #Metrics.append('VCsum_%d_%d' % (r, c))
                Metrics.append('AR_RTR_%d_%d_INQ_PRF_ROWBUS_STALL_CNT' % (r,c))
        data.fillna(0, inplace=True)
        data['PortAvgStall'] = data[Metrics].mean(axis=1) # average over ports.
        avgStall = data['PortAvgStall'].mean(axis=0) # average over router and time.
        return avgStall

    def calcRouterFlit(self, data):
        Metrics = []
        for r in range(5):
            for c in range(8):
                data['VCsum_%d_%d' % (r,c)] = data[['AR_RTR_%d_%d_INQ_PRF_INCOMING_FLIT_VC%d' % (r,c,v) for v in range(8)]].sum(axis=1)
                Metrics.append('VCsum_%d_%d' % (r, c))
        data.fillna(0, inplace=True)
        data['PortAvgStall'] = data[Metrics].mean(axis=1) # average over ports.
        avg = data['PortAvgStall'].mean(axis=0) # average over router and time.
        return avg

    def calcRouterAll(self, data):
        Metrics, rowbusMetrics, stallMetrics = [], [], []
        for r in range(5):
            for c in range(8):
                data['VCsum_%d_%d' % (r,c)] = data[['AR_RTR_%d_%d_INQ_PRF_INCOMING_FLIT_VC%d' % (r,c,v) for v in range(8)]].sum(axis=1)
                Metrics.append('VCsum_%d_%d' % (r, c))
                data['stall_%d_%d' % (r,c)] = data[['AR_RTR_%d_%d_INQ_PRF_INCOMING_PKT_VC%d_FILTER_FLIT%d_CNT' % (r,c,v,v) for v in range(8)]].sum(axis=1)
                stallMetrics.append('stall_%d_%d' % (r, c))
                rowbusMetrics.append('AR_RTR_%d_%d_INQ_PRF_ROWBUS_STALL_CNT' % (r,c))
        data.fillna(0, inplace=True)
        data['PortAvgFlit'] = data[Metrics].mean(axis=1) # average over ports.
        flit = data['PortAvgFlit'].mean(axis=0) # average over router and time.
        data['stallvc'] = data[stallMetrics].mean(axis=1) # average over ports.
        stallvc = data['stallvc'].mean(axis=0) # average over router and time.
        data['stallrowbus'] = data[rowbusMetrics].mean(axis=1) # average over ports.
        stallrowbus = data['stallrowbus'].mean(axis=0) # average over router and time.
        return flit, stallvc, stallrowbus

    def calcRouterMetric(self, data):
        '''
        Calculate router stall to flit ratio.
        '''
        filterElec, filterOpt = 5410794, 933 # the median value
            # the filter is necessary to avoid spikes created by a few flits.
        #print('Filter: %s, FilterElec: %d, FilterOpt: %d' % (str(filterOn), filterElec, filterOpt))
        Metrics = []
        for r in range(5):
            for c in range(8):
                data['VCsum_%d_%d' % (r,c)] = data[['AR_RTR_%d_%d_INQ_PRF_INCOMING_FLIT_VC%d' % (r,c,v) for v in range(8)]].sum(axis=1)
                if r==0 and c>=1 and c<=6 or r==1 and c>=2 and c<=5:
                    filterTile = filterOpt
                else:
                    filterTile = filterElec
                data['Ratio_%d_%d' % (r,c)] = data['AR_RTR_%d_%d_INQ_PRF_ROWBUS_STALL_CNT' % (r,c)].apply(lambda x: x if x>filterTile else 0) / data['VCsum_%d_%d' % (r,c)]
                Metrics.append('Ratio_%d_%d' % (r, c))
        data.fillna(0, inplace=True)
        data['Congestion_rtr'] = data[Metrics].mean(axis=1)
        timerouterAvg = data['Congestion_rtr'].mean(axis=0)
        return timerouterAvg

    def calcNodeMetric(self, data, nodes, filterOn=True):
        '''
        Calculate the node congestion by computing for each node the stall/flit.
        '''
        filterREQ = 1019 # the median value over 100 1-second samples is 1099. over 1800 second is 1019.
        filterRSP = 0 # seems have a problem.
            # the filter is necessary to avoid spikes created by a few flits.
        #print('Filter: %s, FilterREQ: %d, FilterRSP: %d' % (str(filterOn), filterREQ, filterRSP))
        for nic in range(4):
            if filterOn:
                firstDiv = data['AR_NL_PRF_REQ_PTILES_TO_NIC_%d_STALLED' % nic].apply(lambda x: x if x>filterREQ else 0) / data['AR_NL_PRF_REQ_PTILES_TO_NIC_%d_FLITS' % nic]
                #secondDiv = data['AR_NL_PRF_RSP_PTILES_TO_NIC_%d_STALLED' % nic].apply(lambda x: x if x>filterRSP else 0) / data['AR_NL_PRF_RSP_PTILES_TO_NIC_%d_FLITS' % nic]
            else:
                firstDiv = data['AR_NL_PRF_REQ_PTILES_TO_NIC_%d_STALLED' % nic] / data['AR_NL_PRF_REQ_PTILES_TO_NIC_%d_FLITS' % nic]
                #secondDiv = data['AR_NL_PRF_RSP_PTILES_TO_NIC_%d_STALLED' % nic] / data['AR_NL_PRF_RSP_PTILES_TO_NIC_%d_FLITS' % nic]
            #firstDiv.fillna(0, inplace=True)
            #secondDiv.fillna(0, inplace=True)
            #data['Congestion_REQ_nic%d' % nic] = firstDiv
            #data['Congestion_RSP_nic%d' % nic] = secondDiv
            data['Congestion_nic%d' % nic] = firstDiv# + secondDiv
        data.fillna(0, inplace=True)
        nodevalue = []
        for node in nodes:
            cname = self.getCName(node)
            thisRouter = data[data['aries_rtr_id'] == cname[:-1] + '0']
            if len(thisRouter) > 0:
                thisNodeRatio = thisRouter['Congestion_nic%s' % cname[-1]].mean()
                nodevalue.append(thisNodeRatio)
        if len(nodevalue) > 0:
            ratio = sum(nodevalue)/len(nodevalue)
        else:
            ratio = -1
        return ratio

    def calcGroup(self, name):
        if name == None:
            return None
        r = re.compile("(c)([0-9]+)(-)([0-9]+)(c)([0-9]+)(s)([0-9]+)")
        m = r.match(name)
        group = int(m.group(4))*6+np.floor(int(m.group(2))/2)
        return int(group)

    def groupsSpanned(self, cnamelist):
        # From a list of cnames calculate the number of groups spanned 
        return list(set([self.calcGroup(name) for name in cnamelist]))

    def nodeToRouter(self, nodelist):
        thisRouters = []
        for node in nodelist:
            cname = self.getCName(node)
            router = cname[:-1] + '0' # replace the nic value by 0.
            if router not in thisRouters:
                thisRouters.append(router)
        return thisRouters

class analysis(ldms):
    def __init__(self, outputName, logExist):
        super().__init__(logExist)
        self.outputName = outputName

    def timeToUnixPST(self, line):
        return pacific.localize(datetime.datetime.strptime(line, '%a %b %d %H:%M:%S PST %Y')).timestamp()

    def timeToUnixPDT(self, line):
        return pacific.localize(datetime.datetime.strptime(line, '%a %b %d %H:%M:%S PDT %Y')).timestamp()

    def parseJob(self):
        fs = getfiles('OSUresults')
        out = open('jobparse.csv', 'w')
        out.write('mode,jobid,execTime,startUnix,t_total,t_force,t_neigh,t_comm,t_other,t_extra\n')
        for f in fs:
            name = f.split('/')[-1]
            with open(f, 'r') as o:
                mode = ''
                for line in o:
                    if line.startswith('startTime: '):
                        thisTime = self.timeToUnixPST(line[11:-1])
                        started = self.jobnodelist[self.jobnodelist['startUnix']<=thisTime]
                        jobid = started.loc[started['startUnix'].idxmax(axis=0)]['JobID']
                    if line.startswith('real'):
                        finish = 1
                        minute = float(line.split()[1].split('m')[0])
                        sec = float(line.split()[1].split('m')[1][:-1])
                        t = minute * 60 + sec
                        if name.startswith('withCong'):
                            mode = 'OSU_32c'
                        elif name.startswith('noCong'):
                            mode = 'no_OSU'
                        elif name.startswith('Cong16c'):
                            mode = 'OSU_16c'
                    if line.startswith('1024 1'):
                        spl = line.split()
                        decompose = [spl[x] for x in [4,5,6,7,8,-1]]
                if mode == '':
                    print('Imcomplete output: %s' % name)
                else:
                    out.write('%s,%d,%f,%d,' % (mode, jobid, t, thisTime) + ','.join(decompose) + '\n')
        out.close()
        self.joblist = pd.read_csv('jobparse.csv', index_col='jobid')

    def common(self, a, b):
        '''
        count the common appearance in sorted list a, b
        '''
        count, i, j = 0, 0, 0
        while i < len(a) and j < len(b):
            if a[i] == b[j]:
                count += 1
                i += 1
                j += 1
            elif a[i] < b[j]:
                i += 1
            elif a[i] > b[j]:
                j += 1
        return count

    def parseTwoJob(self):
        '''
        Build a table of the experiment output info.
        rt is the real time from the time command.
        et is the execution reported by app.
        '''
        N = 32
        fs = getfiles('OSUresults')
        out = open('jobparse.csv', 'w')
        out.write('index,mode,jobid,realTime1,execTime1,startUnix1,routerShare1,groupSpan1,routerSpan1,realTime2,execTime2,startUnix2,routerShare2,groupSpan2,routerSpan2,nodes\n')
        for f in fs:
            name = f.split('/')[-1]
            if not name.startswith('alloc_c32_ins5'):
                continue
            index = int(name.split('.')[0].split('_')[-1])
            with open(f, 'r') as o:
                mode = ''
                startTimeCount, realCount, execCount = 0, 0, 0
                rt, et, thisTime = {}, {}, {}
                for line in o:
                    if line.startswith('startTime:'):
                        startTimeCount += 1
                        thisTime[startTimeCount] = self.timeToUnixPST(line.lstrip('startTime:').lstrip('good:').lstrip('bad:')[:-1])
                        if startTimeCount == 1:
                            started = self.jobnodelist[self.jobnodelist['startUnix']<=thisTime[startTimeCount]]
                            jobid = started.loc[started['startUnix'].idxmax(axis=0)]['JobID']
                    if line.startswith('real'):
                        realCount += 1
                        minute = float(line.split()[1].split('m')[0])
                        sec = float(line.split()[1].split('m')[1][:-1])
                        rt[realCount] = minute * 60 + sec
                        if name.startswith('withCong'):
                            mode = 'OSU_32c'
                        elif name.startswith('noCong'):
                            mode = 'no_OSU'
                        elif name.startswith('Cong16c'):
                            mode = 'OSU_16c'
                        elif name.startswith('alloc'):
                            mode = 'alloc_c32_ins5'
                    if line.startswith('1024 1'):
                        execCount += 1
                        spl = line.split()
                        decompose = [spl[x] for x in [4,5,6,7,8,-1]]
                        et[execCount] = float(decompose[0])
                    if line.startswith('['):
                        # parse the allocation.
                        nodes = [int(x) for x in line[1:-2].split(', ')]
                        alloc = '_'.join([str(x) for x in nodes])
                        grey = [nodes[2*x+1] for x in range(N)]# osu.
                        green = nodes[-32:]# separate.
                        yellow = [nodes[2*x] for x in range(N)]# intertwinned.

                        greyRouter = [n//4*4 for n in grey]
                        greenRouter = [n//4*4 for n in green]
                        yellowRouter = [n//4*4 for n in yellow]
                        gg = self.common(greyRouter, greenRouter)
                        gy = self.common(greyRouter, yellowRouter)
                        ggCoef = gg/len(greenRouter)*100
                        gyCoef = gy/len(yellowRouter)*100

                        gSpan = len(set([n//384 for n in green]))
                        ySpan = len(set([n//384 for n in yellow]))
                        gRSpan = len(set([n//4 for n in green]))
                        yRSpan = len(set([n//4 for n in yellow]))
                if mode == '':
                    print('Imcomplete output: %s' % name)
                else:
                    out.write('%d,%s,%d,%f,%f,%d,%.1f,%d,%d,%f,%f,%d,%.1f,%d,%d,%s\n' % (index, mode, jobid, rt[1], et[1], thisTime[1], ggCoef, gSpan, gRSpan, rt[2], et[2], thisTime[2], gyCoef, ySpan, yRSpan, alloc))
        out.close()
        self.joblist = pd.read_csv('jobparse.csv', index_col='jobid')
        print('jobparse.csv written.')

    def runtime(self):
        '''
        Get the valid run list.
        Get exec time and reported time.
        '''
        fs = getfiles('../miniMD/results')
        self.timeDict, self.t = {}, {}
        for f in fs:
            run = int(f.split('/')[-1].split('.')[0][3:])
            finish = 0
            with open(f, 'r') as o:
                for line in o:
                    if line.startswith('real'):
                        finish = 1
                        minute = float(line.split()[1].split('m')[0])
                        sec = float(line.split()[1].split('m')[1][:-1])
                        t = minute * 60 + sec
                        self.timeDict[run] = t
                    if line.startswith('2048 1'):
                        spl = line.split()
                        self.t[run] = [float(spl[x]) for x in [4,5,6,7,8,-1]]
            if finish == 0:
                print('run %d not finished.' % (run))
        self.validRuns = list(self.timeDict.keys())
        self.validRuns.sort()
        self.queryTime = [self.timeDict[x] for x in self.validRuns]

    def getTimeRange(self, jobid, index):
        '''
        Get the query time start and end, in forms of pandas.Timestamp
        '''
        # currently getting 1 min since app start.
        deltaTime = 60
        queryTime = pd.Timestamp(self.joblist.loc[jobid]['startUnix%d' % index], unit='s', tz='US/Pacific')
        queryEnd = queryTime + pd.Timedelta(seconds=deltaTime)
        #queryEnd = pd.Timestamp(timeStart, unit='s', tz='US/Pacific')
        #queryTime = queryEnd - pd.Timedelta(seconds=deltaTime)
        return queryTime, queryEnd

    def timeToUnix(self, line):
        return pacific.localize(datetime.datetime.strptime(line, '%Y-%m-%dT%H:%M:%S')).timestamp()

    def getRouter(self):
        '''
        Build a dict recording the routers that each job is running on.
        warning: dict index is changed to jobid.
        '''
        self.jobnodelist = pd.read_csv('nodelist_withOSU.csv', sep='|')
        self.routers1, self.routers2, self.nodes, self.cnames1, self.cnames2 = {}, {}, {}, {}, {}

        self.jobnodelist['startUnix'] = self.jobnodelist.apply(lambda x: self.timeToUnix(x['Start']), axis=1)
        self.jobnodelist = self.jobnodelist.astype({'startUnix': int})
        for idx, row in self.jobnodelist.iterrows():
            jobid, nodeString = row[['JobID','NodeList']]
            self.nodes[jobid] = super().parseNodeList(nodeString)
            thisRouters, thisCnames = [], []

            for node in self.nodes[jobid][-32:]:
                cname = super().getCName(node)
                thisCnames.append(cname)
                router = cname[:-1] + '0' # replace the nic value by 0.
                if router not in thisRouters:
                    thisRouters.append(router)
            self.routers1[jobid] = thisRouters
            self.cnames1[jobid] = thisCnames

            for node in [self.nodes[jobid][2*i] for i in range(32)]:
                cname = super().getCName(node)
                thisCnames.append(cname)
                router = cname[:-1] + '0' # replace the nic value by 0.
                if router not in thisRouters:
                    thisRouters.append(router)
            self.routers2[jobid] = thisRouters
            self.cnames2[jobid] = thisCnames

    def analyzeAlloc(self):
        '''
        Analyze the allocation of the runs.
        '''
        self.getRouter()
        self.parseTwoJob()

    def calcNode(self):
        '''
        Calculate the node list for app and congestor used in experiments.
        '''
        N = 64
        iteration = 10
        rotate = 3*N // iteration
        nodelist = [746, 759, 760, 761, 762, 763, 764, 765, 766, 805, 806, 807, 808, 809, 810, 811, 812, 813, 814, 815, 816, 817, 818, 819, 820, 909, 910, 911, 912, 913, 914, 915, 916, 920, 921, 922, 923, 924, 925, 926, 927, 928, 929, 930, 931, 932, 998, 999, 1000, 1001, 1002, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1163, 1164, 1165, 1166, 1167, 1339, 1340, 1341, 1342, 1343, 1369, 1370, 1371, 1372, 1373, 1374, 1375, 1376, 1377, 1378, 1379, 1380, 1381, 1382, 1383, 1384, 1385, 1386, 1387, 1388, 1389, 1390, 1391, 1407, 1463, 1464, 1465, 1466, 1467, 1468, 1469, 1470, 1471, 1472, 1520, 1521, 1522, 1523, 1524, 1525, 1526, 1527, 1528, 1529, 1530, 1531, 1532, 1618, 1619, 1620, 1621, 1622, 1623, 1624, 1625, 1626, 1755, 1756, 1757, 1758, 1763, 1764, 1765, 1766, 1767, 1768, 1769, 1770, 1771, 1772, 1910, 1911, 1912, 1913, 1979, 1980, 2020, 2021, 2022, 2023, 2024, 2025, 2026, 2027, 2028, 2029, 2030, 2169, 2170, 2171, 2172, 2267, 12723, 12753, 12776, 12812, 12813, 12814, 12815, 12816, 12817, 12818, 12819, 12820, 12821, 12822, 12823, 12824, 12825, 12826, 12827, 12828, 12829, 12830, 12831, 12918, 12968, 12992]
        for i in range(iteration):
            print()
            print('===========================')
            print('iteration:%d' % i)
            allNodes = nodelist[i*rotate:] + nodelist[:i*rotate]
            congestNodes = [allNodes[2*x+1] for x in range(N)]
            greenNodes = allNodes[-N:]
            yellowNodes = [allNodes[2*x] for x in range(N)]
            print('congestor:' + str(congestNodes))
            print('green:' + str(greenNodes))
            print('yellow:' + str(yellowNodes))

    def processFix(self, app='nekbone'):
        '''
        app: miniMD, nekbone.
        '''
        run = 10
        task = 2048
        if app == 'miniMD':
            f = 'results/fixAlloc_gpc5.out'
        elif app == 'nekbone':
            f = 'results/fixAlloc_nekbone.out'
        print('Getting exec time..')
        with open(f, 'r') as o:
            et = {}
            count = 0
            for line in o:
                if app == 'miniMD':
                    if line.startswith('%d 1' % task):
                        count += 1
                        spl = line.split()
                        decompose = [spl[x] for x in [4,5,6,7,8,-1]]
                        et[count] = float(decompose[0])
                elif app == 'nekbone':
                    if line.startswith('Avg MFlops'):
                        count += 1
                        et[count] = float(line.split()[3].split('E')[0]) * 10**6
                    #if line.startswith('startTime:'):
                    #    start = self.timeToUnixPDT(line.lstrip('startTime:')[:-1])
                    #if line.startswith('endTime:'):
                    #    count += 1
                    #    end = self.timeToUnixPDT(line.lstrip('endTime:')[:-1])
                    #    et[count] = end - start
        df = pd.DataFrame(columns=['run','green','yellow','greenGPC','yellowGPC'])
        df['run'] = list(range(run))
        df['green'] = [et[x] for x in range(1, 4*run+1, 4)]
        df['yellow'] = [et[x] for x in range(2, 4*run+1, 4)]
        df['greenGPC'] = [et[x] for x in range(3, 4*run+1, 4)]
        df['yellowGPC'] = [et[x] for x in range(4, 4*run+1, 4)]
        if app == 'miniMD':
            df.to_csv('resultFixGPC5.csv', index=False)
        elif app == 'nekbone':
            df.to_csv('resultFix_nekbone.csv', index=False)
        asdf

        print('Getting span data..')
        with open(f, 'r') as o:
            routerSpan = {}
            count = 0
            for line in o:
                if line.startswith('app-node list'):
                    count += 1
                if count >= 1 and line.startswith('['):
                    appnodes = [int(x) for x in line[1:-2].split(', ')]
                    routers = super().nodeToRouter(appnodes)
                    routerSpan[count] = len(routers)

        df = pd.DataFrame(columns=['run','green','yellow','greenGPC','yellowGPC'])
        df['run'] = list(range(run))
        df['green'] = [routerSpan[x] for x in range(1, 4*run+1, 4)]
        df['yellow'] = [routerSpan[x] for x in range(2, 4*run+1, 4)]
        df['greenGPC'] = [routerSpan[x] for x in range(3, 4*run+1, 4)]
        df['yellowGPC'] = [routerSpan[x] for x in range(4, 4*run+1, 4)]
        df.to_csv('fixGPC5_routerSpan.csv', index=False)

        print('Getting router data..')
        with open(f, 'r') as o:
            stall, ratio = {}, {}
            count = 0
            for line in o:
                if line.startswith('app-node list'):
                    count += 1
                if count >= 1 and line.startswith('['):
                    appnodes = [int(x) for x in line[1:-2].split(', ')]
                    routers = super().nodeToRouter(appnodes)
                if line.startswith('startTime:'):
                    deltaTime = 50
                    queryStart = pd.Timestamp(self.timeToUnixPST(line.lstrip('startTime:')[:-1]), unit='s', tz='US/Pacific')
                    queryEnd = queryStart + pd.Timedelta(seconds=deltaTime)
                    qd = super().fetchData(queryStart, queryEnd, 'rtr')
                    if qd is None:
                        stall[count] = -1
                        ratio[count] = -1
                        continue
                    sel = super().selectRouter(qd, routers)
                    rmDup, jump, numSelectedProducer, prodnum = super().removeDuplicate(sel)
                    if jump:
                        stall[count] = -2
                        ratio[count] = -2
                        continue
                    diff = super().getDiff(rmDup)
                    regu = super().regularize(diff)
                    stall[count] = super().calcRouterStall(regu)
                    ratio[count] = super().calcRouterMetric(regu)

        dfStall = pd.DataFrame(columns=['run','green','yellow','greenGPC','yellowGPC'])
        dfStall['run'] = list(range(run))
        dfStall['green'] = [stall[x] for x in range(1, 4*run+1, 4)]
        dfStall['yellow'] = [stall[x] for x in range(2, 4*run+1, 4)]
        dfStall['greenGPC'] = [stall[x] for x in range(3, 4*run+1, 4)]
        dfStall['yellowGPC'] = [stall[x] for x in range(4, 4*run+1, 4)]
        dfStall.to_csv('fixGPC1_stall.csv', index=False)

        dfRatio = pd.DataFrame(columns=['run','green','yellow','greenGPC','yellowGPC'])
        dfRatio['run'] = list(range(run))
        dfRatio['green'] = [ratio[x] for x in range(1, 4*run+1, 4)]
        dfRatio['yellow'] = [ratio[x] for x in range(2, 4*run+1, 4)]
        dfRatio['greenGPC'] = [ratio[x] for x in range(3, 4*run+1, 4)]
        dfRatio['yellowGPC'] = [ratio[x] for x in range(4, 4*run+1, 4)]
        dfRatio.to_csv('fixGPC1_ratio.csv', index=False)

    def process(self, mode, counterSaved, saveFolder):
        '''
        '''
        self.getRouter()
        self.parseTwoJob()
        with open(self.outputName, 'w') as o:
            if mode == 'rtrstall':
                o.write('mode,jobid,producerNum1,execTime1,avgStall1,producerNum2,execTime2,avgStall2\n')
        i = 0
        for idx, row in self.joblist.iterrows():
            jobid, jobmode, execTime1, execTime2 = idx, row['mode'], row['execTime1'], row['execTime2']
            i += 1
            print('Processing run %d, jobid %d..' % (i, jobid))
            prodnum, value = {}, {}
            for index in [1,2]:
                queryTime, queryEnd = self.getTimeRange(jobid, index)
                qd = super().fetchData(queryTime, queryEnd, 'rtr')
                if qd is None:
                    print('No counter data.')
                    break
                if len(qd) == 0:
                    print('Empty counter data.')
                    break
                #print(qd.iloc[0]['#Time'])
                #print(qd.iloc[-1]['#Time'])
                if index == 1:
                    sel = super().selectRouter(qd, self.routers1[jobid])
                else:
                    sel = super().selectRouter(qd, self.routers2[jobid])
                rmDup, jump, numSelectedProducer, prodnum[index] = super().removeDuplicate(sel)
                if not jump:
                    diff = super().getDiff(rmDup)
                    regu = super().regularize(diff)
                    if index == 1:
                        numGroup = len(super().groupsSpanned(self.cnames1[jobid]))
                    else:
                        numGroup = len(super().groupsSpanned(self.cnames2[jobid]))
                    value[index] = super().calcRouterStall(regu)
                    print(str(value[index]))
                else:
                    break
            with open(self.outputName, 'a') as o:
                writeline = '%s,%d,%d,%f,%f,%d,%f,%f\n' % (jobmode, jobid, prodnum[1], execTime1, value[1], prodnum[2], execTime2, value[2])
                o.write(writeline)

    def oldprocess(self, mode, counterSaved, saveFolder):
        '''
        mode: metric, count, nodemetric.
        '''
        self.getRouter()
        self.avgStall = []
        with open(self.outputName, 'w') as o:
            if mode in ['nodestall','rtrstall']:
                name = 'avgStall'
                o.write('run,execTime,%s,avgFlit,t_total,t_force,t_neigh,t_comm,t_other,t_extra\n' % name)
            elif mode == 'rtrmetric':
                name = 'congestion'
                o.write('run,execTime,%s,t_total,t_force,t_neigh,t_comm,t_other,t_extra\n' % name)
            elif mode == 'nodemetric':
                name = 'ratio'
                o.write('run,execTime,%s,t_total,t_force,t_neigh,t_comm,t_other,t_extra\n' % name)
            elif mode == 'rtrall':
                title = 'run,selectedProducer,numRouter,numGroup,'
                title += ','.join(self.allrtrmetrics)
                title += ',execTime,t_total,t_force,t_neigh,t_comm,t_other,t_extra\n'
                o.write(title)
        for run in reversed(self.validRuns):
            if run >= 110:
                continue # jump latest runs without counter data permission.
            print('Processing run %d..' % run)
            if counterSaved:
                if os.path.isfile('%s/run%d.csv' % (saveFolder, run)):
                    sel = pd.read_csv('%s/run%d.csv' % (saveFolder, run))
                else:
                    continue
            else:
                queryTime, queryEnd = self.getTimeRange(run)
                qd = super().fetchData(queryTime, queryEnd, 'rtr')
                if qd is None:
                    continue
                #print(qd.iloc[0]['#Time'])
                #print(qd.iloc[-1]['#Time'])
                sel = super().selectRouter(qd, self.routers[run])
                sel.to_csv('%s/run%d.csv' % (saveFolder, run), index=0)
            rmDup, jump, numSelectedProducer, prodnum = super().removeDuplicate(sel)
            if not jump:
                diff = super().getDiff(rmDup)
                regu = super().regularize(diff)
                numRouter = len(self.routers[run])
                numGroup = len(super().groupsSpanned(self.cnames[run]))
                if mode == 'rtrstall':
                    value = super().calcRouterStall(regu)
                    value2 = super().calcRouterFlit(regu)
                elif mode == 'rtrmetric':
                    value = super().calcRouterMetric(regu)
                elif mode == 'nodestall':
                    value = super().calcNodeStall(regu, self.nodes[run])
                elif mode == 'nodemetric':
                    value = super().calcNodeMetric(regu, self.nodes[run])
                elif mode == 'rtrall':
                    value = super().calcRouterAll(regu)
                print(str(value))
                self.avgStall.append(value)
                with open(self.outputName, 'a') as o:
                    if mode == 'rtrstall':
                        writeline = '%d,%f,%f,%f' % (run, self.timeDict[run], value, value2)
                    elif mode == 'rtrall':
                        writeline = '%d,%d,%d,%d,%f,%f,%f,%f' % (run, numSelectedProducer, numRouter, numGroup, value[0], value[1], value[2], self.timeDict[run])
                    else:
                        writeline = '%d,%f,%f' % (run, self.timeDict[run], value)
                    for i in range(6):
                        writeline = writeline + ',' + str(self.t[run][i])
                    writeline += '\n'
                    o.write(writeline)

if __name__ == '__main__':
    main()

