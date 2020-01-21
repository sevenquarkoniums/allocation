#!/bin/bash
sacct --name=withOSU -S 2019-11-02T21:00:00 -E 2020-01-20T03:00:00 -s R -P -X -o jobid,jobname,nnodes,start,end,NodeList # -P
