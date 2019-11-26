#!/bin/bash
sacct --name=miniMD -S 2019-11-02T21:00:00 -E 2019-11-25T00:00:00 -s R -P -X -o jobid,jobname,nnodes,start,end,NodeList # -P
