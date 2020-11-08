#!/bin/bash
#sacct --name=withOSU -S 2020-01-28T13:00:00 -E 2020-02-16T03:00:00 -s R -P -X -o jobid,jobname,nnodes,start,end,NodeList # -P
sacct -S 2020-05-04T13:00:00 -E 2020-05-06T03:00:00 -P -X -o jobid,jobname,nnodes,start,end # -P
