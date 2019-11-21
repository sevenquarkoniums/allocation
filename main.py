#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
import multiprocessing
import pickle
import datetime
import pytz
from io import StringIO
pacific = pytz.timezone('US/Pacific')

def main():
    al = analysis(outputName='data_-1to0min.csv', logExist=1)
    al.runtime()
    al.process()

def getfiles(path):
    # get all the files with full path.
    fileList = []
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            fileList.append(filepath)
    return fileList

def unix2string(unixTime):
    thisTime = datetime.datetime.utcfromtimestamp(unixTime).replace(tzinfo=datetime.timezone.utc).astimezone(tz=pacific)
    timestr = thisTime.strftime('%Y-%m-%d %H:%M:%S')
    return timestr

class ldms:
    def __init__(self, logExist=False):
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
        queryData = pd.concat(fetched, ignore_index=True)
        if len(queryData) == 0:
            print('No data for this time %s.' % str(datetimeStart))
            return None
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
        print('Selecting routers..')
        sel = data[data['aries_rtr_id'].isin(routers)].copy()
        return sel

    def removeDuplicate(self, data):
        '''
        Only select the producer with a smaller node id.
        A few routers are only collected by one producer.
        '''
        #print('Removing duplicates..')
        producers = data[['component_id','aries_rtr_id']].copy()
        #print('All producer number: %d' % len(producers['component_id'].unique()))
        selectedProducer = list( producers.groupby(['aries_rtr_id']).min()['component_id'] )
            # all the 'aries_rtr_id' fields end with 'a0'.
        print('Selected producer number: %d' % len(selectedProducer))
        rmDup = data[data['component_id'].isin(selectedProducer)].copy()
        jump = 1 if len(selectedProducer)==0 else 0
        if jump:
            print('### No counter data for this run ###')
        return rmDup, jump

    def getDiff(self, data):
        #print('Calculating the difference..')
        data['HTime'] = data['#Time'].apply(unix2string)
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

    def calcRouterStall(self, data):
        '''
        Calculate the avg stall of the 40 tiles in all the routers.
        Averaged on 3 dimensions: time, routers, ports.
        Missing data are replaced by zero.
        '''
        Metrics = []
        for r in range(5):
            for c in range(8):
                data['VCsum_%d_%d' % (r,c)] = data[['AR_RTR_%d_%d_INQ_PRF_INCOMING_FLIT_VC%d' % (r,c,v) for v in range(8)]].sum(axis=1)
                Metrics.append('VCsum_%d_%d' % (r, c))
        data.fillna(0, inplace=True)
        data['PortAvgStall'] = data[Metrics].mean(axis=1)
        avgStall = data['PortAvgStall'].mean(axis=0)
        return avgStall

class analysis(ldms):
    def __init__(self, outputName, logExist):
        super().__init__(logExist)
        self.outputName = outputName

    def runtime(self):
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

    def getTimeRange(self, run):
        '''
        Get the query time start and end, in forms of pandas.Timestamp
        '''
        # currently getting 1 min since app start.
        deltaTime = 60
        with open('../miniMD/results/run%d.out' % run, 'r') as o:
            firstline = o.readline()
        if firstline[:-1].split()[4] == 'PDT':
            timeStart = datetime.datetime.strptime(firstline[:-1], '%a %b %d %H:%M:%S PDT %Y').replace(tzinfo=pacific).timestamp()
        elif firstline[:-1].split()[4] == 'PST':
            timeStart = datetime.datetime.strptime(firstline[:-1], '%a %b %d %H:%M:%S PST %Y').replace(tzinfo=pacific).timestamp()
        #queryTime = pd.Timestamp(timeStart, unit='s', tz='US/Pacific')
        #queryEnd = queryTime + pd.Timedelta(seconds=deltaTime)
        queryEnd = pd.Timestamp(timeStart, unit='s', tz='US/Pacific')
        queryTime = queryEnd - pd.Timedelta(seconds=deltaTime)
        return queryTime, queryEnd

    def getRouter(self):
        '''
        Build a dict recording the routers that each job is running on.
        '''
        nodelist = pd.read_csv('nodelist.csv', sep='|')
        self.routers = {}
        for idx, row in nodelist.iterrows():
            run = idx
            jobid, nodeString = row[['JobID','NodeList']]
            nodes = super().parseNodeList(nodeString)
            thisRouters = []
            for node in nodes:
                router = super().getCName(node)[:-1] + '0' # replace the nic value by 0.
                if router not in thisRouters:
                    thisRouters.append(router)
            self.routers[run] = thisRouters

    def process(self):
        self.getRouter()
        self.avgStall = []
        with open(self.outputName, 'w') as o:
            o.write('run,execTime,avgStall,t_total,t_force,t_neigh,t_comm,t_other,t_extra\n')
        for run in reversed(self.validRuns):
            if run >= 50:
                continue # jump latest runs without counter data.
            print('Processing run %d..' % run)
            queryTime, queryEnd = self.getTimeRange(run)
            qd = super().fetchData(queryTime, queryEnd, 'rtr')
            sel = super().selectRouter(qd, self.routers[run])
            rmDup, jump = super().removeDuplicate(sel)
            if not jump:
                diff = super().getDiff(rmDup)
                regu = super().regularize(diff)
                thisStall = super().calcRouterStall(regu)
                self.avgStall.append(thisStall)
                with open(self.outputName, 'a') as o:
                    writeline = '%d,%f,%f' % (run, self.timeDict[run], thisStall)
                    for i in range(6):
                        writeline = writeline + ',' + str(self.t[run][i])
                    writeline += '\n'
                    o.write(writeline)
                print(thisStall)

    def drawCorr(self):
        ax.plot(self.execTime, self.avgStall, '.')

if __name__ == '__main__':
    main()

