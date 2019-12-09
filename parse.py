#!/usr/bin/env python
with open('metrics.txt', 'w') as o:
    with open('temp.csv', 'r') as f:
        line = f.readline()
        for m in line.split(','):
            o.write(m+'\n')
