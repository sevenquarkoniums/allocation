#!/usr/bin/env python
import os
import pandas as pd
import multiprocessing
import pickle

def main():
    al = analysis(logExist=0)
    #al.runtime()
    al.process()

class analysis(ldms):
    def __init__(self, logExist):
        ldms.__init__(self, logExist=logExist)

    def runtime(self):
        fs = getfiles('../miniMD/results')
        for f in fs:
            with open(f, 'r') as o:
                for line in o:
                    if line.startswith('real'):
                        minute = float(line.split()[1].split('m')[0])
                        sec = float(line.split()[1].split('m')[1][:-1])
                        t = minute * 60 + sec
                        print(t)

    def process(self):
        self.setQuery('multi')
        qD = ldms.fetchData(self.queryTime, self.queryEnd, 'rtr')
        rmDup = ldms.removeDuplicate(qD)
        diff = ldms.getDiff(rmDup)
        regu = ldms.regularize(diff)

def getfiles(path):
    # get all the files with full path.
    fileList = []
    for root, directories, files in os.walk(path):
        for filename in files:
            filepath = os.path.join(root, filename)
            fileList.append(filepath)
    return fileList

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
        for p in getAggregatorPaths():
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
            return bstFileNear(left, mid, time_t, f)
        else:
            return bstFileNear(mid, right, time_t, f)

    def findRtrStartFilesV2(self.start_time, file_start_dict):
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
        print('Fetching data..')
        startFiles = self.findRtrStartFilesV2(datetimeStart, self.rtr_log_starts if source=='rtr' else self.nic_log_starts)
        endFiles = self.findRtrEndFilesV2(datetimeEnd, self.rtr_log_starts if source=='rtr' else self.nic_log_starts)
        files = list( set().union(startFiles, endFiles) )
        unixTime = datetimeStart.value // 10**9
        unixEnd = datetimeEnd.value // 10**9
        pargs = [(f, unixTime, unixEnd) for f in files]
        fetched = self.pool.starmap(lh.subsetLinesToDF, pargs)
        queryData = pd.concat(fetched, ignore_index=True)
        if len(queryData) == 0:
            print('No data for this time %s.' % str(datetimeStart))
            return None
        return queryData

    def removeDuplicate(self, queryData):
        '''
        Only select the producer with a smaller node id.
        A few routers are only collected by one producer.
        '''
        print('Removing duplicates..')
        producers = queryData[['component_id','aries_rtr_id']].copy()
        print('All producer number: %d' % len(producers['component_id'].unique()))
        selectedProducer = list( producers.groupby(['aries_rtr_id']).min()['component_id'] )
        print('Selected producer number: %d' % len(selectedProducer))
        rmDup = queryData[queryData['component_id'].isin(selectedProducer)].copy()
        return rmDup

    def getDiff(self, data):
        print('Calculating the difference..')
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
        print('Regularizing the values..')
        noDivCol = ['HTime','Time_usec','ProducerName','component_id','aries_rtr_id']
        metrics = [x for x in data.columns if x not in noDivCol]
        metrics.remove('#Time')# This #Time column is already the differentiated one.
        noDiv = data[noDivCol]
        div = data[metrics].div(data['#Time'], axis=0)
        regu = pd.concat([noDiv, div], axis=1)
        regu['#Time'] = regu['HTime']
        #regu['#Time'] = regu['HTime'].apply(lambda x: x - pd.Timedelta(microseconds=x.microsecond))
        return regu

if __name__ == '__main__':
    main()
