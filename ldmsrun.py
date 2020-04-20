#!/usr/bin/python2.7
"""
 LDMS sampler and store run script
 -- should work in cluster environment and connect
    store to all appropriate samplers (on nodes
    listed in PBS_NODEFILE or SLURM_NODELIST TODO )

   Usage: <script> [sampler|store] [sosperjob jobId#]
       no arg will run both sampler and store (for local testing)
       one arg specifies just one to run (for cluster-based use)
       for store, if 1 sos container per job is desired, use
            "sosperstore jobId#" as 2nd and third args

 - normally, on a compute node this script will be run to just
   start a sampler daemon, and on the head node it will be run
   to start the store daemon

 TODO still lots of control options that should variables
"""
import os
import sys
import subprocess
import time
import re

# Which samplers to load? (need to doc which are supported here)
# set this list to just the samplers you want loaded
# TODO: make this an environment variable
# full list: appinfo, meminfo, hweventpapi, procstat, vmstat,
#            (TODO metric_set_nic, cray_aries_r)
#samplerList = ['appinfo', 'meminfo', 'jobinfo', 'procstat', 'vmstat', 'hweventpapi', 'aries_nic_mmr', 'cray_aries_r']
#samplerList = ['appinfo', 'meminfo', 'jobinfo', 'procstat', 'vmstat']
samplerList = ['aries_nic_mmr', 'cray_aries_r']
#samplerList = ['appinfo', 'meminfo', 'hweventpapi']
#samplerList = ['meminfo']
# wait for sampler daemon to finish (for voltrino)
waitForExit = True

# clear out umask for allowing created and shared containers
os.umask(0)

# """script debug flag"""
debug = True
# if old version of LDMS that allows unix socket control
oldLdmsVersion = False
# ldms debug flag, set to [""] for default
ldmsDebug = ["-v","DEBUG"]
# ldms home default, will get set later
ldmsHome = "/home/jcook/tools"
# current working directory
wdir = os.getcwd()
# hostname default, will get set later
hostName = 'localhost'
# component id (will be set in setup)
componentId = 0
# port for ldmsd data connections
dataPort = 10001
# sampling interval for ldms sampler (usec)
sampleInterval = 1000000
# sample offset for samplers (usec)
sampleOffset = 50
# TODO appname, jobId, username are sent to
# appinfo plugin, but not used
appname = "app"
jobId = 1234
username = "user"
# port for store-ldmsd data? (notused?)
storePort = 10002
# storage producer+updater interval
storeInterval = 1000000
# storage updater offset
storeOffset = 400
# store path and container default, will be set later
storePath = "/home/jcook/ovis"
container = "jecdata"
# True for one SOS container per job
sosPerJob = False

# hardware counter schema select
#papiSchema = "spapiSKY"
#papiSchema = "spapiBASIC"
papiSchema = "spapiHASW"

# Two store modes available sos and csv
storeMode = "sos"

#----------------------------------------------------------
def logDebug(str):
   """ Log a debug message, if debug is set """
   global debug, hostName
   if debug:
      print "LR-{0}: {1}".format(hostName,str)

#----------------------------------------------------------
def setup():
   """ Set up everything for LDMS runs
   - checks/sets OVIS_HOME
   - sets store path and container (from LDMS_STOREPATH, LDMS_CONTAINER)
   - sets LDMSD_PLUGIN_PATH, PATH, ZAP_LIBPATH, LD_LIBRARY_PATH
   - gets hostname from /bin/hostname (or HOSTNAME, but prior is
     needed in clusters, and extracts component id from hostname digits
     (or uses localhost and component id of 0)
   """
   global debug, componentId, hostName, ldmsHome, storePath, container
   global ldmsdCmd, ldmsctlCmd, sosPerJob, storeMode
   # find/set ldms/ovis install home
   if 'OVIS_HOME' in os.environ:
      ldmsHome = os.getenv('OVIS_HOME')
   else:
      # use default and inject into environment
      os.environ['OVIS_HOME'] = ldmsHome
   logDebug("OVIS_HOME is {0}".format(ldmsHome))
   # storage container path and name
   if 'LDMS_STOREPATH' in os.environ:
      storePath = os.environ["LDMS_STOREPATH"]
#   elif 'LDMS_SOS_RAWDB' in os.environ:
#      storePath = os.getenv('LDMS_SOS_RAWDB')
   else:
      storePath = os.environ['PWD']
   # store mode sos or csv
   if os.environ["LDMS_STOREMODE"] == 'csv':
       storeMode = "store_csv"
       print "Env var is passed store mode is:", storeMode
   else:
       storeMode = "store_sos"
       print "Env var is passed store mode is:", storeMode
   # ABOVE PART MODIFIED BY EMRE
   # if 'LDMS_STOREMODE' in os.environ:
   #    if os.environ["LDMS_CONTAINER"] == 'csv':
   #        storeMode = "store_csv"
   #        print "Env var is passed store mode is:", storeMode
   #    else:
   #        storeMode = "store_sos"
   #        print "Env var is passed store mode is:", storeMode
   # else:
   #    storeMode = "store_sos"
   #    print "Env var is not passed store mode is:", storeMode

   if 'LDMS_CONTAINER' in os.environ:
      container = os.environ["LDMS_CONTAINER"]
   else:
      container = "sosdb"
   if sosPerJob:
      # make job id the container name
      container = str(jobId)
   logDebug("Storage container is {0}/{1}".format(storePath,container))
   # set up binary executables and environment variables
   ldmsdCmd = ldmsHome+"/sbin/ldmsd"
   ldmsctlCmd = ldmsHome+"/sbin/ldmsctl"
   os.environ['LDMSD_PLUGIN_LIBPATH'] = ldmsHome+"/lib/ovis-ldms"
   os.environ['ZAP_LIBPATH'] = ldmsHome+"/lib/ovis-ldms"
   if 'PATH' in os.environ:
      os.environ['PATH'] = ldmsHome+"/bin:"+ldmsHome+"/sbin:"+os.environ["PATH"]
   else:
      os.environ['PATH'] = ldmsHome+"/bin:"+ldmsHome+"/sbin"
   if 'LD_LIBRARY_PATH' in os.environ:
      os.environ['LD_LIBRARY_PATH'] = ldmsHome+"/lib:"+os.environ["LD_LIBRARY_PATH"]
   else:
      os.environ['LD_LIBRARY_PATH'] = ldmsHome+"/lib"
   if 'PYTHONPATH' in os.environ:
      os.environ['PYTHONPATH'] = ldmsHome+"/lib/python2.7/site-packages:"+\
                                 os.environ["PYTHONPATH"]
   else:
      os.environ['PYTHONPATH'] = ldmsHome+"/lib/python2.7/site-packages"
   # get proper hostname, and create component id
   hn = None
   if not 'HOSTNAME' in os.environ:
      logDebug("VERY WEIRD: no HOSTNAME in env!")
   else:
      hn = os.environ["HOSTNAME"]
   logDebug("Hostname is {0}".format(hn))
   cp = subprocess.Popen(["/bin/hostname"],\
            stdout=subprocess.PIPE,stderr=subprocess.PIPE)
   r = cp.communicate()
   logDebug("/bin/hostName says {0}".format(r))
   bhn = r[0].strip()
   if bhn != hn:
      # /bin/hostName is different from HOSTNAME -- on bigdat
      # Slurm DOES NOT set HOSTNAME properly, it is the name of
      # the node where the job script runs, no matter what!
      logDebug("/bin/hostName ({0}) is different from $HOSTNAME ({1})!!! Using former".format(bhn,hn))
      hn = bhn
   if hn is None:
      componentId = 0
   else:
      hostName = hn
      componentId = getNodeNumberFromName(hn)
   logDebug("Component ID is {0}".format(componentId))

#----------------------------------------------------------
def getNodeList(hostfile=None):
   """ Get list of nodes from batch job host file or env var
   - checks/uses LDMS_NODEFILE, LDMS_NODELIST, SLURM_NODELIST,
     and PBS_NODEFILE; reads file if file var, parses list if
     list var
   - if none then returns ["localhost"] for non-cluster test runs
   - if node is listed multiple times (consecutively!) it will
     be recorded only once
   - if file line includes options after node name, they will be ignored
   :param hostfile: is a specific host file name to use, but
                    this is generally left out
   """
   if hostfile is None and 'LDMS_NODEFILE' in os.environ:
      hostfile = os.getenv('LDMS_NODEFILE')
   if hostfile is None and 'LDMS_NODELIST' in os.environ:
      return getNodeListFromSlurmList(os.getenv('LDMS_NODELIST'))
   if hostfile is None and 'SLURM_NODELIST' in os.environ:
      return getNodeListFromSlurmList(os.getenv('SLURM_NODELIST'))
   if hostfile is None and 'PBS_NODEFILE' in os.environ:
      hostfile = os.getenv('PBS_NODEFILE')
   logDebug("Looking for compute nodes in file {0}".format(hostfile))
   if type(hostfile) != type('string') or not os.path.exists(hostfile):
      return ['localhost']
   hf = open(hostfile,"r")
   if (hf is None):
      return ['localhost']
   nlist = []
   lastnodename = ''
   for line in hf:
      #print line
      nodename = line.split()[0]
      #print nodename
      if nodename != lastnodename:
         nlist.append(nodename)
         lastnodename = nodename
   hf.close()
   logDebug("Nodes from file: {0}".format(nlist))
   return nlist

#----------------------------------------------------------
def getNodeListFromSlurmList(nlstring):
   """ Create node list from SLURM_NODELIST environment
   variable; can look like "nodid00[21-22,33,45-56]"
      :param nlstring: is the nodelist string
      :returns: list of full unique nodenames
   """
   nlist = []
   command = 'scontrol show hostnames={0}'.format(nlstring)
   process = subprocess.Popen(command.split(),stdout=subprocess.PIPE)
   while True:
      line = process.stdout.readline()
      if not line:
         break
      nlist.append(line.rstrip())
   return nlist

#----------------------------------------------------------
# Omar: This is the old nodelist parsing, please check the above new version
def getNodeListFromSlurmListParse(nlstring):
   """ Create node list from SLURM_NODELIST-like environment
   variable; can look like "nodid00[21-22,33,45-56]"
      :param nlstring: is the nodelist string
      :returns: list of full unique nodenames
   """
   logDebug("Extracting nodes from {0}".format(nlstring))
   nlps = nlstring.split('[')
   if len(nlps) == 1:
      # is just one node name, so return it as list of 1
      return nlps
   # otherwise, need to parse indices
   nlist = []
   basename = nlps[0]
   nparts = nlps[1].split(',')
   for part in nparts:
      v = re.match("(\d+)-(\d+)",part)
      print(v)
      if v is None:
         # then just a single number, not a range
         # could have trailing ], so remove
         nn = basename + part.split(']')[0]
         nlist.append(nn)
         continue
      # else we have a range, so use it
      beg = int(v.group(1))
      end = int(v.group(2))
      for n in range(beg,end+1):
         nlist.append(basename+str(n))
   logDebug("Nodes from string: {0}".format(nlist))
   return nlist

#----------------------------------------------------------
def getNodeNumberFromName(nodename):
   """ Get node number from a nodename that is assumed to
   contain a node number in it. Very simplistic, it just
   grabs the first digits it sees.
      :param nodename: is the nodename string
      :returns: a string of digits extracted from nodename
   """
   v = re.match("\D+(\d+)",nodename)
   if v is None:
      return 0
   return int(v.group(1))

#----------------------------------------------------------
def preCleanup(starting):
   """ Clean up old files from earlier LDMS runs
   - TODO: set var to find 'rm' rather than use '/bin/rm'
   """
   global wdir, componentId
   if starting=="sampler" or starting=="both":
      # remove old shared memory stuff, start clean
      subprocess.call("/bin/rm -f /dev/shm/*ldms*", shell=True)
      # remove sampler log and sock files
      subprocess.call("/bin/rm -rf {0}/samplerd-{1}.log".format(wdir,componentId), shell=True)
      subprocess.call("/bin/rm -rf /tmp/ldms-sample-sock", shell=True)
   if starting=="store" or starting=="both":
      # remove storage log and sock files
      subprocess.call("/bin/rm -rf {0}/storaged-{1}.log".format(wdir,componentId), shell=True)
      subprocess.call("/bin/rm -rf /tmp/ldms-store-sock", shell=True)

#----------------------------------------------------------
def runSampler():
   """ Setup and run a sampler ldmsd
     - assumes many things
   """
   global componentId, ldmsdCmd, ldmsctlCmd, wdir, dataPort, oldLdmsVersion
   global sampleInterval, sampleOffset, samplerList, waitForExit, ldmsHome
   global appname, jobId, username, ldmsDebug, hostName, papiSchema

   offset = 0
   inputstr = ""
   if "jobinfo" in samplerList:
      inputstr += "load name=jobinfo\n\
config name=jobinfo instance={0}/jobinfo producer={3} component_id={0}\n\
start name=jobinfo interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName)
      offset += sampleOffset
   if "meminfo" in samplerList:
      inputstr += "load name=meminfo\n\
config name=meminfo instance={0}/meminfo producer={3} job_set={0}/jobinfo component_id={0}\n\
start name=meminfo interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName)
      offset += sampleOffset
   if "procstat" in samplerList:
      inputstr += "load name=procstat\n\
config name=procstat instance={0}/procstat producer={3} job_set={0}/jobinfo component_id={0} maxcpu=64\n\
start name=procstat interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName)
      offset += sampleOffset
   if "vmstat" in samplerList:
      inputstr += "load name=vmstat\n\
config name=vmstat instance={0}/vmstat producer={3} job_set={0}/jobinfo component_id={0}\n\
start name=vmstat interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName)
      offset += sampleOffset
   if "aries_nic_mmr" in samplerList:
      inputstr += "load name=aries_nic_mmr\n\
config name=aries_nic_mmr instance={0}/aries_nic_mmr producer={3} job_set={0}/jobinfo component_id={0} file={4}/{5}\n\
start name=aries_nic_mmr interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName,ldmsHome,"etc/ldms/aries_mmr_set_configs/metric_set_nic")
      offset += sampleOffset
   if "cray_aries_r" in samplerList:
      # if want luster, add llite=lscratch to config
      inputstr += "load name=cray_aries_r_sampler\n\
config name=cray_aries_r_sampler instance={0}/cray_aries_r producer={3} job_set={0}/jobinfo component_id={0} schema=cray_aries_r off_hsn=0\n\
start name=cray_aries_r_sampler interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName)
      offset += sampleOffset
   if "appinfo" in samplerList:
      inputstr += "load name=appinfo\n\
config name=appinfo producer={6} instance={0}/appinfo job_set={0}/jobinfo component_id={0} metrics=hbeat1:U64,hbeat2:U64,hbeat3:U64,hbeat4:U64,hbeat5:U64 appname={1} jobid={2} username={3}\n\
start name=appinfo interval={4} offset={5}\n".format(componentId,appname,jobId,username,sampleInterval,offset,hostName)
      offset += sampleOffset
   if "hweventpapi" in samplerList:
      inputstr += "load name=hweventpapi\n\
config name=hweventpapi instance={0}/{4} producer={3} job_set={0}/jobinfo component_id={0} metafile=/tmp/hwcntrs\n\
start name=hweventpapi interval={1} offset={2}\n".format(componentId,sampleInterval,offset,hostName,papiSchema)
      offset += sampleOffset

   with open("/tmp/conf.ldms.sampler", "w") as text_file:
       text_file.write("{0}".format(inputstr))

   logDebug("Input to sampler:\n{0}".format(inputstr))

   if oldLdmsVersion:
      arglist = ["{0}".format(ldmsdCmd),"-x","sock:{0}".format(dataPort),\
                 "-p","unix:/tmp/ldms-sample-sock","-l",\
                 "{0}/samplerd-{1}.log".format(wdir,componentId),"-c","/tmp/conf.ldms.sampler"]
   else:
      arglist = ["{0}".format(ldmsdCmd),"-x","sock:{0}".format(dataPort),\
                 "-l","{0}/samplerd-{1}.log".format(wdir,componentId),"-c","/tmp/conf.ldms.sampler"]
   if waitForExit:
      # want to wait, so don't daemonize ldmsd
      arglist.append('-F')
   if len(ldmsDebug) > 0:
      arglist.extend(ldmsDebug)
   samplerProc = subprocess.Popen(arglist)
   print "component {0} sampler pid: {1}".format(componentId,samplerProc.pid)
   time.sleep(1)

   if waitForExit:
      # don't end script but wait (should do in main?)
      #time.sleep(30)
      samplerProc.wait()

#-No intermediate aggregator
#rm -rf $WDIR/aggregatord.log
#rm -rf /tmp/ldms-store-sock
#ldmsd -x sock:10003 -p unix:/tmp/ldms-agg-sock -l $WDIR/aggregatord.log
#def runaggregator():

#----------------------------------------------------------
def runStorage():
   """ Setup and run a storage ldmsd
   """
   global componentId, ldmsdCmd, ldmsctlCmd, wdir, dataPort, storePort, hostName
   global storeInterval, storeOffset, container, ldmsDebug, papiSchema
   global samplerList, oldLdmsVersion, storeMode

   # generate producer lines for every node that job is running on
   prodlines = ""
   for node in getNodeList():
      nid = getNodeNumberFromName(node)
      pl = "prdcr_add name=LHP{0} host={1} xprt=sock port={2} type=active interval={3}\nprdcr_start name=LHP{0}\n".format(nid,node,dataPort,storeInterval)
      prodlines += pl
   # send all stdin to ldmsctlCmd
   inputstr = "{0}\
updtr_add name=LHU interval={1} offset={2}\n\
updtr_prdcr_add name=LHU regex=LHP*\n\
updtr_start name=LHU\n".format(prodlines,storeInterval,storeOffset)

   if storeMode is 'store_sos':
      inputstr += "load name=store_sos\n\
config name=store_sos path={0}\n".format(storePath)
   else:
      inputstr += "load name=store_csv\n\
config name=store_csv path={0}\n".format(storePath)

   if "meminfo" in samplerList:
      inputstr += "strgp_add name=LHSGM plugin={1} schema=meminfo container={0}\nstrgp_start name=LHSGM\n".format(container,storeMode)
   if "jobinfo" in samplerList:
      inputstr += "strgp_add name=LHSGJ plugin={1} schema=jobinfo container={0}\nstrgp_start name=LHSGJ\n".format(container,storeMode)
   if "procstat" in samplerList:
      inputstr += "strgp_add name=LHSGR plugin={1} schema=procstat container={0}\nstrgp_start name=LHSGR\n".format(container,storeMode)
   if "vmstat" in samplerList:
      inputstr += "strgp_add name=LHSGV plugin={1} schema=vmstat container={0}\nstrgp_start name=LHSGV\n".format(container,storeMode)
   if "aries_nic_mmr" in samplerList:
      inputstr += "strgp_add name=LHSGN plugin={1} schema=aries_nic_mmr container={0}\nstrgp_start name=LHSGN\n".format(container,storeMode)
   if "cray_aries_r" in samplerList:
      inputstr += "strgp_add name=LHSGC plugin={1} schema=cray_aries_r container={0}\nstrgp_start name=LHSGC\n".format(container,storeMode)
   if "appinfo" in samplerList:
      inputstr += "strgp_add name=LHSGA plugin={1} schema=appinfo container={0}\nstrgp_start name=LHSGA\n".format(container,storeMode)
   if "hweventpapi" in samplerList:
      inputstr += "strgp_add name=LHSGP plugin={2} schema={1} container={0}\nstrgp_start name=LHSGP\n".format(container,papiSchema,storeMode)

   with open("/tmp/conf.ldms.store", "w") as text_file:
       text_file.write("{0}".format(inputstr))

   logDebug("Input to store:\n{0}".format(inputstr))

   if oldLdmsVersion:
      arglist = ["{0}".format(ldmsdCmd),"-x","sock:{0}".format(storePort),\
                      "-p","unix:/tmp/ldms-store-sock",\
                      "-l","{0}/storaged-{1}.log".format(wdir,componentId),"-c","/tmp/conf.ldms.store"]
   else:
      arglist = ["{0}".format(ldmsdCmd),"-x","sock:{0}".format(storePort),"-m","512M",\
                      "-l","{0}/storaged-{1}.log".format(wdir,componentId),"-c","/tmp/conf.ldms.store"]

   if len(ldmsDebug) > 0:
      arglist.extend(ldmsDebug)
   storagePid = subprocess.Popen(arglist).pid
   print "component {0} storage pid: {1}".format(componentId,storagePid)
   time.sleep(1)

#----------------------------------------------------------
def main():
   """ Main program """
   global starting, ldmsdCmd, ldmsHome, sosPerJob, jobId
   # Are we running sampler, store, or both???
   if len(sys.argv)==1:
      starting = "both"
   elif len(sys.argv)==2 and sys.argv[1] in ['sampler','store']:
      starting = sys.argv[1]
   elif len(sys.argv)==4 and sys.argv[1] == 'store' and sys.argv[2] == 'sosperjob':
      starting = sys.argv[1]
      sosPerJob = True
      jobId = int(sys.argv[3])
   else:
      print "Usage: ",sys.argv[0],"[sampler|store] [sosperjob jobid#]"
      print "  no arg will run both sampler and store"
      print "  arg specifies just one to run"
      print "  for store, if 1 sos container per job is desired, use"
      print "     'sosperstore jobId#' as 2nd and third args"
      exit(1)
   # Now go run what we need to
   setup()
   # Good ldms path setup?
   if not os.path.exists(ldmsdCmd):
      print "No such file ({0}): bad OVIS_HOME? Or not installed at {1}".\
            format(ldmsdCmd,ldmsHome)
      exit(2)
   # ok, setup is good I guess
   preCleanup(starting)
   if starting=="sampler" or starting=="both":
      if 'SLURM_LOCALID' in os.environ:
         logDebug("SLURM_LOCALID is {0}".format(os.environ['SLURM_LOCALID']))
         if os.environ['SLURM_LOCALID'] != '0':
            print 'Not the first local process on {0}...skipping sampler'.format(componentId)
         else:
            runSampler()
      else:
         runSampler()
   if starting=="store" or starting=="both":
      runStorage()
   # all done
   exit(0)

#----------------------------------------------------------
"""
Main program
"""
# protection from execution by PyDoc and other module-based method
if __name__ == '__main__':
   main()

