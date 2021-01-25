# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:25:05 2020

@author: seven
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from statistics import median,mean,stdev
import numpy as np
import scipy
from scipy import stats
import datetime
now = datetime.datetime.now()

colors = ['limegreen','orange','darkgreen','chocolate']
#colors = ['limegreen','deepskyblue','darkgreen','royalblue']

#routers = ['c7-0c1s1a0', 'c7-0c0s10a0', 'c5-0c1s14a0', 'c5-0c0s12a0', 'c7-0c1s6a0', 
#           'c7-0c0s9a0', 'c4-0c2s6a0', 'c7-0c1s9a0', 'c7-0c1s4a0', 'c7-0c1s10a0', 
#           'c4-0c2s13a0', 'c5-0c1s3a0', 'c7-0c1s8a0', 'c5-0c1s1a0', 'c7-0c1s5a0', 
#           'c7-0c0s15a0', 'c7-0c1s7a0', 'c5-0c0s13a0', 'c7-0c0s11a0', 'c7-0c1s3a0']
#routers = routers[15:20]
#dfs = []
#for day in range(5, 25):
#    dfs.append(pd.read_csv('D:/BUforD/data/reduced_30min_20r_May%d.csv' % day))
#coriAries = pd.concat(dfs, ignore_index=True)    
#coriAries['HTime'] = pd.to_datetime(coriAries['HTime'])

def main():
    # fig. 4, original index.
#    timeNormalizedReshape(eps=1)
    
    # fig. 5
#    for app in ['lammps']:#, 'hpcg', 'miniMD', 'nekbone', 'hacc', 'graph500', 'miniamr', 'milc', 'qmcpack']:
#        drawBox(app, normalize=1, eps=0)
    
    # fig. 6, 7
#    for metric in ['ratio','stall']:
#        for app in ['lammps', 'hacc', 'qmcpack', 'miniMD', 'miniamr', 'milc', 'nekbone', 'graph500']:
#            drawFix(metric, app, drawPoint=0, standardError=1, nodemetric=0, eps=0)
    
    # fig. 8
#    for metric in ['ratio','stall']:
#        for app in ['lammps', 'qmcpack', 'miniMD', 'miniamr', 'milc', 'hacc']:
#            drawFix(metric, app, drawPoint=0, standardError=1, nodemetric=1, eps=0)

    # fig. 9
#    for app in ['lammps', 'hacc', 'qmcpack', 'miniMD', 'miniamr']:
#        flitTime(app=app, drawPoint=1, standardError=0, nodemetric=1, eps=1, useColor=0)

    # slides.
#    for app in ['miniMD']:#, 'hpcg', 'miniMD', 'nekbone', 'hacc', 'graph500', 'miniamr', 'milc', 'qmcpack']:
#        drawBox(app, normalize=0, eps=0)
#    for metric in ['ratio']:
#        for app in ['miniMD']:
#            drawFix(metric, app, drawPoint=0, standardError=1, nodemetric=0, eps=0, addlgd=1)



#    for app in ['lammps', 'hacc', 'qmcpack', 'miniMD', 'miniamr', 'milc', 'nekbone', 'graph500']:
#        drawFix2('stall', app, drawPoint=0, standardError=1, nodemetric=0, eps=0)

#    for app in ['hacc','milc','lammps','miniMD']:
#        drawNeDD(app,show='mean',sensitive=1)
#    for app in ['qmcpack','hpcg']:
#        drawNeDD(app,show='mean',sensitive=0)
    
    drawNeDDTwo('qmcpack', 'miniMD')

#    drawCADDeach('lammps')

#    df = pd.read_csv('D:/BUforD/data/reduced_30min_20r_May%d.csv' % 16)
#    drawData(df, day=16)
#    drawPort(coriAries)
#    drawLong(coriAries)
    
#    compareLDMS()
#    drawMultiNode()
#    drawSingleNode()

#    for app in ['hacc', 'qmcpack', 'miniMD', 'miniamr']:
#        flitStall(app=app, drawPoint=1, standardError=0, nodemetric=1, eps=1, useColor=0)
#    drawTwo()
#    drawTwoTimeStall()
#    timeNormalized()
#    drawFixSpan()
#    drawAvgStd()
    print('finished in %d seconds.' % (datetime.datetime.now()-now).seconds)

def drawData(df, day):
    rowavg = 5
    granular = 's'
    rnames = 'Routers:\n'
#    routers = list(set(df['aries_rtr_id']))[:5]
    for router in routers:
        rnames += router + '\n'
    for metric in ['PortAvgStall']:#,'PortAvgFlit']:
        figname = 'data_30min_20r_May%d_14.5-15_%s.png' % (day, metric)
        fs = 20
        figsizeX, figsizeY = 10,8
        plt.rc('xtick', labelsize=fs)
        plt.rc('ytick', labelsize=fs)
        fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
        for r in routers:
            sel = df[df['aries_rtr_id']==r]
            avg = sel.groupby(np.arange(len(sel))//rowavg).mean()
            data = avg[metric]
            ax.plot(range(0, len(data)*rowavg, rowavg), data, '-')
        ax.yaxis.grid(linestyle='--', color='black')
        ax.xaxis.grid(linestyle='--', color='black')
        if granular == '2h':
            ax.set_xlabel('Time (h)', fontsize=fs)
            ax.set_xticks([7200*x for x in range(12)])
            ax.set_xticklabels(['%d' % (2*x) for x in range(12)], fontsize=fs)
        elif granular == '1h':
            ax.set_xlabel('Time (h)', fontsize=fs)
            ax.set_xticks([3600*x for x in range(24)])
            ax.set_xticklabels(['%d' % (1*x) for x in range(24)], fontsize=fs)
        elif granular == 's':
            ax.set_xlabel('Time (30 minutes)', fontsize=fs)
            ax.set_xticklabels([], fontsize=fs)
        ax.set_xlim(14.5*3600, 15*3600)
#        ax.set_ylim(0, 1.2*10**8)
        if metric == 'PortAvgStall':
            ax.set_ylabel('Avg Stall per Second', fontsize=fs)
        elif metric == 'PortAvgFlit':
            ax.set_ylabel('Avg Flit per Second', fontsize=fs)
        ax.text(0.1, 0.9, 'Each line is a router', ha='left', va='center', transform=ax.transAxes, fontsize=fs)
        ax.text(0.1, 0.85, 'Every point is an average of %d seconds' % rowavg, ha='left', va='center', transform=ax.transAxes, fontsize=fs)
        ax.text(0.95, 0.5, rnames, ha='right', va='center', transform=ax.transAxes, fontsize=fs)
        fig.tight_layout()
        fig.savefig('C:/Programming/monitoring/CoriAries/%s' % figname)
        plt.close()

def drawMultiNode():
    df = pd.read_csv('ldms_1.csv')
    nodes = list(set(df['component_id']))
    fs = 20
    figsizeX, figsizeY = 10,8
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    exist = []
    for node in nodes:
        if node in exist:
            continue
        figname = 'GPC128_traffic/ldms_node%d.png' % (node)
        fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
        sel = df[df['component_id']==node]
        t = [x for x in sel['#Time']]
        ax.plot(t, sel['traffic_sum'], '-')
#        ax.plot(t, sel['stalled_sum'], '-')
        ax.yaxis.grid(linestyle='--', color='black')
        ax.xaxis.grid(linestyle='--', color='black')
        ax.set_xlabel('Time', fontsize=fs)
#        ax.set_ylim(0, 2*10**10)
        ax.set_ylabel('Network metric', fontsize=fs)
        ax.text(0.9, 0.9, 'node %d' % node, ha='right', va='center', transform=ax.transAxes, fontsize=fs)
        fig.tight_layout()
        fig.savefig('C:/Programming/monitoring/miniMD/%s' % figname)
        plt.close()
        n4 = (node//4)*4
        exist += [n4, n4+1, n4+2, n4+3]

def drawSingleNode():
    df = pd.read_csv('ldms_node4060.csv')
    fs = 20
    figsizeX, figsizeY = 10,8
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    nodraw = ['#Time','component_id']
    cols = [s for s in df.columns if s not in nodraw]
    for metric in cols:
        figname = 'node4060/metric_%s.png' % (metric)
        fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
        ax.plot(df['#Time'], df[metric], '-')
        ax.yaxis.grid(linestyle='--', color='black')
        ax.xaxis.grid(linestyle='--', color='black')
        ax.set_xlabel('Time', fontsize=fs)
        ax.set_ylabel(metric, fontsize=fs)
#        ax.text(0.9, 0.9, 'node %d' % node, ha='right', va='center', transform=ax.transAxes, fontsize=fs)
        fig.tight_layout()
        fig.savefig('C:/Programming/monitoring/miniMD/%s' % figname)
        plt.close()

def drawLong(df):
    rnames = 'Routers:\n'
    for router in routers:
        rnames += router + '\n'
    for metric in ['PortAvgStall']:#,'PortAvgFlit']:
        figname = 'data_30min_20r_May5-24_%s.png' % (metric)
        fs = 20
        figsizeX, figsizeY = 24,5
        plt.rc('xtick', labelsize=fs)
        plt.rc('ytick', labelsize=fs)
        fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
        for r in routers:
            print(r)
            xs, ys = [], []
            thisRouter = df[df['aries_rtr_id']==r]
            for day in range(5, 25):
                for hour in range(24):
                    xs.append(1+hour+day*24)
                    start = pd.to_datetime('2020-05-%02d %02d:00:00' % (day, hour))
                    end = pd.to_datetime('2020-05-%02d %02d:59:59' % (day, hour))
                    thisHour = thisRouter[(thisRouter['HTime']>=start) & (thisRouter['HTime']<=end)]
                    avg = thisHour[metric].mean()
                    ys.append(avg)
            ax.plot(xs, ys, '-')
        ax.yaxis.grid(linestyle='--', color='black')
        ax.xaxis.grid(linestyle='--', color='black')
        ax.set_xlabel('Date (Year 2020)', fontsize=fs)                        
        ax.set_xticks([day*24 for day in range(5, 25)])
        ax.set_xticklabels(['05-%02d' % day for day in range(5, 25)], fontsize=fs)
#        ax.set_xlim(17*3600, 17.1*3600)
#        ax.set_ylim(0, 1.2*10**8)
        if metric == 'PortAvgStall':
            ax.set_ylabel('Avg Stall per Second', fontsize=fs)
        elif metric == 'PortAvgFlit':
            ax.set_ylabel('Avg Flit per Second', fontsize=fs)
        ax.text(0.1, 0.9, 'Each line is a router', ha='left', va='center', transform=ax.transAxes, fontsize=fs)
        ax.text(0.1, 0.85, 'Every point is an average of 1 hour', ha='left', va='center', transform=ax.transAxes, fontsize=fs)
        ax.text(0.95, 0.5, rnames, ha='right', va='center', transform=ax.transAxes, fontsize=fs)
        fig.tight_layout()
        fig.savefig('C:/Programming/monitoring/CoriAries/%s' % figname)
        plt.close()

def drawPort(df):
    rowavg = 360
    routers = list(set(df['aries_rtr_id']))
    granular = '2h'
    fs = 20
    figsizeX, figsizeY = 10,8
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    for router in routers:
        fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
        sel = df[df['aries_rtr_id']==router]
        avg = sel.groupby(np.arange(len(sel))//rowavg).mean()
        metrics = []
        for r in range(5):
            for c in range(8):
                #metrics.append('VCsum_%d_%d' % (r, c))
                metrics.append('AR_RTR_%d_%d_INQ_PRF_ROWBUS_STALL_CNT' % (r,c))
        for metric in metrics[:10]:
            data = avg[metric]
            ax.plot(range(0, len(data)*rowavg, rowavg), data, '-')
        ax.yaxis.grid(linestyle='--', color='black')
        ax.xaxis.grid(linestyle='--', color='black')
        if granular == '2h':
            ax.set_xlabel('Time (h)', fontsize=fs)
            ax.set_xticks([7200*x for x in range(12)])
            ax.set_xticklabels(['%d' % (2*x) for x in range(12)], fontsize=fs)
        elif granular == '1h':
            ax.set_xlabel('Time (h)', fontsize=fs)
            ax.set_xticks([3600*x for x in range(24)])
            ax.set_xticklabels(['%d' % (1*x) for x in range(24)], fontsize=fs)
        elif granular == 's':
            ax.set_xlabel('Time (s)', fontsize=fs)
        ax.set_ylabel('Stall per second (on individual port)', fontsize=fs)
        ax.text(0.1, 0.9, 'Each line is a port from router %s' % router, ha='left', va='center', transform=ax.transAxes, fontsize=fs)
        ax.text(0.1, 0.85, 'Every point is an average of %d seconds' % rowavg, ha='left', va='center', transform=ax.transAxes, fontsize=fs)
        fig.tight_layout()
        fig.savefig('C:/Programming/monitoring/CoriAries/data_30min_20r_May5_%s_24h.png' % router)
        plt.close()
        
def compareLDMS():
    ours = pd.read_csv('ldms_1604887273_nid11796.csv')
    cori = pd.read_csv('corildms_1604887273_nid11796.csv')
    fs = 20
    figsizeX, figsizeY = 10,8
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    for i in range(48):
        fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
        ax.plot(ours['#Time'], ours['stalled_%03d (ns)' % i], '-')
        ax.yaxis.grid(linestyle='--', color='black')
        ax.xaxis.grid(linestyle='--', color='black')
        ax.set_xlabel('Time (s)', fontsize=fs)
        ax.set_ylabel('Stall Count', fontsize=fs)
        ax.text(0.8, 0.5, 'Port %d' % i, ha='center', va='center', transform=ax.transAxes, fontsize=fs)
        fig.tight_layout()
        fig.savefig('C:/Programming/monitoring/compareLDMS/ours/port%03d.png' % i)
        plt.close()

        if i < 40:
            fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
            a, b = i // 8, i % 8
            metric = 'AR_RTR_%d_%d_INQ_PRF_ROWBUS_STALL_CNT' % (a, b)
            ax.plot(cori['#Time'], cori[metric], '-')
            ax.yaxis.grid(linestyle='--', color='black')
            ax.xaxis.grid(linestyle='--', color='black')
            ax.set_xlabel('Time (s)', fontsize=fs)
            ax.set_ylabel('Stall Count', fontsize=fs)
            ax.text(0.8, 0.5, 'Port %d_%d' % (a, b), ha='center', va='center', transform=ax.transAxes, fontsize=fs)
            fig.tight_layout()
            fig.savefig('C:/Programming/monitoring/compareLDMS/cori/port_%d_%d.png' % (a, b))
            plt.close()

def timeNormalizedReshape(eps):
    apps = ['graph500', 'hacc', 'hpcg', 'lammps', 'milc', 'miniamr', 'miniMD', 'qmcpack']
    appsNames = ['Graph500', 'HACC', 'HPCG', 'LAMMPS', 'MILC', 'miniAMR', 'miniMD', 'QMCPACK']
    boxColors = []
    for i in range(len(apps)):
        boxColors += colors
    fs = 20
    figsizeX, figsizeY = 16,6
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    ax.yaxis.grid(linestyle='--', color='black')
    results, positions, med = [], [], {}
    for idx, app in enumerate(apps):
        if app == 'miniMD':
            df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv')
        elif app == 'lammps':
            df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s3.csv' % app)
        else:
            df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s.csv' % app)
        for jdx, col in enumerate(['green','yellow','greenGPC','yellowGPC']):
            validResults = [x for x in df[col] if x!= 0]
            if col == 'green':
                med[app] = median(validResults)
                print('%s\t%f' % (app, med[app]))
            normed = [x/med[app] for x in validResults]
            results.append(normed)
            positions.append(idx*4+jdx)
    b = ax.boxplot(results, whis=100, patch_artist=True, positions=positions)
    ymin, ymax = ax.get_ylim()
    for patch, color in zip(b['boxes'], boxColors):
        patch.set_facecolor(color)
    for m in b['medians']:
        m.set(color='k', linewidth=1.5,)
    ax.set_xlabel('Applications in different experiment settings', fontsize=fs)
    ax.set_ylabel('Normalized Execution Time\n(separately for each application)', fontsize=fs)
    for x in range(len(apps)-1):
        splitPosition = x*4+3.5
        ax.plot([splitPosition, splitPosition], [ymin, ymax], 'k-', linewidth=1)
#    ax.set_ylim(ymin, ymax)
    ax.set_ylim(0.4, 3.1)
#    ax.set_ylim(1, 1.15)
#    ax.text(14.3, 1.25, '(exceed\n  range)', fontsize=12)
    ax.set_xticks([x*4+1.5 for x in range(len(apps))])
    ax.set_xticklabels(appsNames, fontsize=fs)
    patches = []
    for color in colors:
        patches.append(mpatches.Patch(color=color, label=''))
    labels = ['I:Continuous', 'II:Spaced','III:Continuous+Congestor','IV:Spaced+Congestor']
    lgd = ax.legend(handles=patches, labels=labels, loc='upper left', fontsize=fs, ncol=2)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/normalizedReshape.eps'
    else:
        name = 'C:/Programming/monitoring/miniMD/normalizedReshape.png'
    fig.savefig(name)

def timeNormalized():
    apps = ['graph500', 'hacc', 'hpcg', 'milc', 'miniamr', 'miniMD', 'qmcpack']
    colors = ['k','r','y','g','b','m','c']
    boxColors = ['k']*4+['r']*4+['y']*4+['g']*4+['b']*4+['m']*4+['c']*4#+['pink']*4
    fs = 20
    figsizeX, figsizeY = 21,6
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    ax.yaxis.grid(linestyle='--', color='black')
    results, positions, med = [], [], {}
    for idx, app in enumerate(apps):
        if app == 'miniMD':
            df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv')
        else:
            df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s.csv' % app)
        for jdx, col in enumerate(['green','yellow','greenGPC','yellowGPC']):
            validResults = [x for x in df[col] if x!= 0]
            if col == 'green':
                med[app] = median(validResults)
            normed = [x/med[app] for x in validResults]
            results.append(normed)
            positions.append(idx+jdx*(len(apps)+1))
    b = ax.boxplot(results, whis=100, patch_artist=True, positions=positions)
    ymin, ymax = ax.get_ylim()
    for patch, color in zip(b['boxes'], boxColors):
        patch.set_facecolor(color)
    ax.set_xlabel('Experiment settings', fontsize=fs)
    ax.set_ylabel('Normalized Execution Time', fontsize=fs)
    ax.set_xticks([(len(apps)+1)*x + len(apps)/2-0.5 for x in range(4)])
    for x in range(3):
        splitPosition = (x+1)*(len(apps)+1)-1
        ax.plot([splitPosition, splitPosition], [ymin, ymax], 'k-', linewidth=1)
    ax.set_ylim(ymin, ymax)
    labels = ['Continuous', 'Spaced','Continuous\n+GPCNET','Spaced\n+GPCNET']
    ax.set_xticklabels(labels, fontsize=fs)
    patches = []
    for color in colors:
        patches.append(mpatches.Patch(color=color, label=''))
    lgd = ax.legend(handles=patches, labels=apps, loc='upper left', fontsize=fs, ncol=2)
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/normalized.png'
    fig.savefig(name)

def drawBox(app, normalize, eps):
    if app == 'miniMD':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv')
    elif app == 'lammps':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s3.csv' % app)
    else:
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s.csv' % app)
    fs = 20
    figsizeX, figsizeY = 4,6.5
#    figsizeX, figsizeY = 4,5
    labels = ['I:Continuous', 'II:Spaced','III:Continuous\n+Congestor','IV:Spaced\n+Congestor']
    legloc = 'upper right'
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
#    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('LAMMPS\nExperiment settings', fontsize=fs)
#    ax.set_xlabel('miniMD\nExperiment settings', fontsize=fs)
    if app == 'miniMD':
        ax.set_ylabel('miniMD Execution Time (s)', fontsize=fs)
    elif app == 'nekbone':
        ax.set_ylabel('Nekbone Performance (MFLOPS)', fontsize=fs)
    elif app == 'hacc':
        ax.set_ylabel('HACC Execution Time (s)', fontsize=fs)
    elif app == 'graph500':
        ax.set_ylabel('BFS+SSSP Execution Time (s)', fontsize=fs)
    elif app == 'miniamr':
        ax.set_ylabel('miniAMR Execution Time (s)', fontsize=fs)
    elif app == 'lammps':
        if normalize:
            ax.set_ylabel('LAMMPS Execution Time (normalized)', fontsize=fs)
        else:
            ax.set_ylabel('LAMMPS Execution Time (s)', fontsize=fs)
    elif app == 'milc':
        ax.set_ylabel('MILC Execution Time (s)', fontsize=fs)
    elif app == 'qmcpack':
        ax.set_ylabel('QMCPACK Execution Time (s)', fontsize=fs)
    results = []
    if normalize:
        med = median([x for x in df['green']])
    for col in ['green','yellow','greenGPC','yellowGPC']:
        if normalize:
            results.append([x/med for x in df[col] if x!= 0])
        else:
            results.append([x for x in df[col] if x!= 0])
    b = ax.boxplot(results, whis=100, patch_artist=True, labels=labels)
    ax.set_xticklabels([], fontsize=fs)
    if normalize and app == 'lammps':
        ax.set_yticks([0,1,2,4,6,8,10,12])
        ax.set_ylim(0, 13.5)
    for patch, color in zip(b['boxes'], colors):
        patch.set_facecolor(color)
    for m in b['medians']:
        m.set(color='k', linewidth=1.5,)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/fixGPC5_%s.eps' % app
    else:
        name = 'C:/Programming/monitoring/miniMD/fixGPC5_%s.png' % app
    fig.savefig(name)

def drawNeDD(app, show, sensitive):
    if app in ['lammps','miniamr']:
        suffix = '201_new'
    elif app == 'miniMD':
        suffix = '201_80k'
    elif app in ['hpcg','qmcpack','milc','graph500']:
        suffix = '201'
    elif app == 'hacc':
        suffix = '201_comb'
    if app == 'hacc':
        df1 = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDprocess_hacc_201.csv')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDprocess_hacc_201_new.csv')
        df = pd.concat([df1,df2], ignore_index=True)
    else:
        df = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDprocess_%s_%s.csv' % (app, suffix))
    fs, legendfs = 25,20
    figsizeX, figsizeY = 7,9
    colors = ['deepskyblue','grey','pink','orange','limegreen','white']
    if sensitive:
        labels = ['High-Traffic-Router','Random','Low-Stall-Router','Fewer-Router',
                  '[NeDD] Low-Traffic-Router','No-Congestor']
    else:
        labels = ['[NeDD] High-Traffic-Router','Random','Low-Stall-Router','Fewer-Router',
                  'Low-Traffic-Router','No-Congestor']
    cols = ['antinedd','random','lowerRouterStall','fewerSwitch','nedd','fewerNoCong']
    legloc = 'upper right'
    patches = []
    for i, label in enumerate(labels):
        patches.append(mpatches.Patch(facecolor=colors[i], edgecolor='k', label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=legendfs, ncol=1)
#    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('Job Allocation', fontsize=fs)
    ax.set_xticklabels([], fontsize=fs)
    if app == 'milc':
        ax.set_ylabel('MILC Execution Time (s)', fontsize=fs)
    elif app == 'lammps':
        ax.set_ylabel('LAMMPS Execution time (s)', fontsize=fs)
    elif app == 'miniMD':
        ax.set_ylabel('MiniMD Execution time (s)', fontsize=fs)
        ax.set_ylim([50,450])
    elif app == 'hpcg':
        ax.set_ylabel('HPCG Execution time (s)', fontsize=fs)
        ax.set_ylim([30,90])
    elif app == 'miniamr':
        ax.set_ylabel('MiniAMR Execution time (s)', fontsize=fs)
        ax.set_ylim([40,50])
    elif app == 'graph500':
        ax.set_ylabel('Graph500 Execution time (s)', fontsize=fs)
        ax.set_ylim([100,350])
    elif app == 'qmcpack':
        ax.set_ylabel('QMCPACK Execution time (s)', fontsize=fs)
        ax.set_ylim([65,90])
    elif app == 'milc':
        ax.set_ylabel('MILC Execution time (s)', fontsize=fs)
    elif app == 'hacc':
        ax.set_ylabel('HACC Execution time (s)', fontsize=fs)
#    for idx, case in enumerate(['green','yellow']):
#        y, yerr = df[case].mean(), df[case].std()
#        yerr = yerr#/math.sqrt(len(df[case]))
#        ax.bar(idx, y, color=colors[idx], width=0.5, yerr=yerr, ecolor='black', capsize=10)
    values = [[x for x in df[col].values if x != -1] for col in cols] # remove -1.
    if show == 'mean':
        medianprops = dict(linestyle='--', linewidth=1, color='black')
        meanpointprops = dict(marker='D', markeredgecolor='black', markerfacecolor='firebrick')
        b = ax.boxplot(values, whis=100, patch_artist=True, positions=range(1, len(cols)+1), widths=0.6,
                       showmeans=True, meanprops=meanpointprops, medianprops=medianprops)
        fewerSwitch = mean([x for x in df['fewerSwitch']])
        nedd = mean([x for x in df['nedd']])
        random = mean([x for x in df['random']])
        lrs = mean([x for x in df['lowerRouterStall']])
    elif show == 'median':
        b = ax.boxplot(values, whis=100, patch_artist=True, positions=range(1, len(cols)+1), widths=0.6)
        fewerSwitch = median([x for x in df['fewerSwitch']])
        nedd = median([x for x in df['nedd']])
        random = median([x for x in df['random']])
        lrs = median([x for x in df['lowerRouterStall']])
    print('Improve over Fewer-Router: %f%%' % (100*(1-nedd/fewerSwitch)))
    print('Improve over Random: %f%%' % (100*(1-nedd/random)))
    print('Improve over Lower-Router-Stall: %f%%' % (100*(1-nedd/lrs)))
    for patch, color in zip(b['boxes'], colors):
        patch.set_facecolor(color)
#        if color == 'limegreen':
#            patch.set_edgecolor('r')
    for m in b['medians']:
        m.set(color='k', linewidth=1.5,)
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/NeDD_%s_%s.png' % (app, suffix)
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
#    plt.close()

def drawNeDDTwo(app1, app2):
#    df1 = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDTwoprocess_%s.csv' % app1)
#    df2 = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDTwoprocess_%s.csv' % app2)
    df1 = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDTwoproc_%s_%s_1.csv' % (app1, app2))
    df2 = pd.read_csv('C:/Programming/monitoring/miniMD/NeDDTwoproc_%s_%s_2.csv' % (app1, app2))
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,8
    colors = ['deepskyblue','grey','pink','orange','limegreen']
    labels = ['High-Traffic-Router','Random','Low-Stall-Router','Fewer-Router','[NeDD]']
    cols = ['antinedd','random','lowerRouterStall','fewerSwitch','nedd']
    if app1 == 'qmcpack':
        legloc = 'right'
    else:
        legloc = 'upper right'
    patches = []
    for i, label in enumerate(labels):
        patches.append(mpatches.Patch(facecolor=colors[i], edgecolor='k', label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=legendfs, ncol=1)
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    if app1 == 'qmcpack':
        ax.set_xlabel('Execution Time of QMCPACK (s)', fontsize=fs)
    if app1 == 'miniMD':
        ax.set_xlabel('Execution Time of MiniMD (s)', fontsize=fs)
    if app2 == 'qmcpack':
        ax.set_ylabel('Execution Time of QMCPACK (s)', fontsize=fs)
    if app2 == 'milc':
        ax.set_ylabel('Execution Time of MILC (s)', fontsize=fs)
    if app2 == 'miniMD':
        ax.set_ylabel('Execution Time of MiniMD (s)', fontsize=fs)
    values1 = [[x for x in df1[col].values if x != -1] for col in cols] # remove -1.
    values2 = [[x for x in df2[col].values if x != -1] for col in cols] # remove -1.
    mean1 = [mean(values1[i]) for i in range(len(cols))]
    mean2 = [mean(values2[i]) for i in range(len(cols))]
#    mm1 = [(max(values1[i])+min(values1[i]))/2 for i in range(len(cols))]
#    err1 = [(max(values1[i])-min(values1[i]))/2 for i in range(len(cols))]
#    mm2 = [(max(values2[i])+min(values2[i]))/2 for i in range(len(cols))]
#    err2 = [(max(values2[i])-min(values2[i]))/2 for i in range(len(cols))]
#    ax.errorbar(mean1, mm2, yerr=err2, xerr=None, ls='')
#    ax.errorbar(mm1, mean2, yerr=None, xerr=err1, ls='')
#    err1 = [stdev(values1[i]) for i in range(len(cols))]
#    err2 = [stdev(values2[i]) for i in range(len(cols))]
    err1 = [stdev(values1[i])/math.sqrt(len(values1[i])) for i in range(len(cols))]
    err2 = [stdev(values2[i])/math.sqrt(len(values2[i])) for i in range(len(cols))]
    ax.errorbar(mean1, mean2, yerr=err2, xerr=err1, ls='', color='k')
    if app2 == 'qmcpack':
        ax.set_ylim(71, 76)
    if app2 == 'milc':
        ax.set_ylim(90, 220)
    ax.scatter(mean1, mean2, facecolor=colors, edgecolor='k', marker='d', s=500)
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/NeDDTwo_%s_%s' % (app1, app2)
    fig.savefig('%s.png' % name, bbox_extra_artists=(lgd,), bbox_inches='tight')
#    fig.savefig('%s.eps' % name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)

def drawCADDeach(app):
    df = pd.read_csv('C:/Programming/monitoring/miniMD/CADDprocess_%s_2policy.csv' % app)
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,8
    colors = ['limegreen','orange']
#    labels = ['Low-stall Allocation', 'High-stall Allocation']
    labels = ['Congestion-aware Allocation', 'Random Allocation']
    legloc = 'upper'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=legendfs, ncol=1)
#    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('Experiment runs', fontsize=fs)
    ax.set_ylabel('Execution time (s)', fontsize=fs)
    ax.set_xticklabels([], fontsize=fs)
    if app == 'milc':
        ax.set_ylabel('MILC Execution Time (s)', fontsize=fs)
#    for idx, case in enumerate(['green','yellow']):
#        y, yerr = df[case].mean(), df[case].std()
#        yerr = yerr#/math.sqrt(len(df[case]))
#        ax.bar(idx, y, color=colors[idx], width=0.5, yerr=yerr, ecolor='black', capsize=10)
    ax.bar(range(1, 3*len(df['green']), 3), df['green'], color=colors[0])
    ax.bar(range(2, 3*len(df['yellow'])+1, 3), df['yellow'], color=colors[1])
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/CADDeach_%s_2policy.png' % (app)
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
#    plt.close()

def drawFixSpan():
    df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv', index_col='run')
    df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC5_routerSpan.csv', index_col='run')
    df2[df2 <= 0] = None
    fs = 20
    figsizeX, figsizeY = 8,8
    colors = ['forestgreen','orange','darkgreen','chocolate']
    labels = ['Continuous', 'Spaced','Continuous\n+GPCNET','Spaced\n+GPCNET']
    legloc = 'upper right'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=fs, ncol=2)
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('Router Span', fontsize=fs)
    ax.set_ylabel('miniMD Execution Time (s)', fontsize=fs)
    for idx, case in enumerate(['green','yellow','greenGPC','yellowGPC']):
#        x, xerr = df2[case].mean(), df2[case].std()
#        y, yerr = df[case].mean(), df[case].std()
#        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=2, capsize=5)
        for run in range(10):
            value = df2.loc[run][case]
            if value is not None:
                ax.plot(value, df.loc[run][case], '.', markersize=15, color=colors[idx])
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/fixGPC5_routerSpan.png'
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')

def flitStall(app, drawPoint, standardError, nodemetric, eps, useColor):
    df = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_nodeflit.csv' % app, index_col='run')
    df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_nodestall.csv' % (app), index_col='run')
    df[df <= 0] = None
    df2[df2 <= 0] = None
    if app == 'hacc':
        df[df <= 10000] = None
        df2[df2 <= 10000] = None
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,7.5
    if useColor:
        colors = ['forestgreen','orange','darkgreen','chocolate']
        labels = ['Continuous', 'Spaced','Continuous\n+Congestor','Spaced\n+Congestor']
    else:
        colors = ['grey']*4
        labels = ['Experiment run']
    legloc = 'upper center'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=legendfs, ncol=1)
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    xlabel = '%s Flits Per Second' % ('Ptile' if nodemetric else 'Ntile')
    ylabel = '%s Stalls Per Second' % ('Ptile' if nodemetric else 'Ntile')
    ax.set_xlabel(xlabel, fontsize=fs)
    ax.set_ylabel(ylabel, fontsize=fs)
    xs, ys, allxs, allys = [], [], [], []
    for idx, case in enumerate(['green','yellow','greenGPC','yellowGPC']):
        x, xerr = df[case].mean(), df[case].std()
        y, yerr = df2[case].mean(), df2[case].std()
        xs.append(x)
        ys.append(y)
        if standardError:
            xerr = xerr/math.sqrt(len(df[case]))
            yerr = yerr/math.sqrt(len(df2[case]))
        if not drawPoint:
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=3, capsize=5)
        for run in range(10):
            value = df.loc[run][case]
            value2 = df2.loc[run][case]
            if not math.isnan(value):
                if app == 'qmcpack' and value2 > 100000:# remove outlier.
                    continue
                allxs.append(value)
                allys.append(value2)
                if drawPoint:
                    ax.plot(value, value2, '.', markersize=15, color=colors[idx])
    xs, ys = np.array(xs), np.array(ys)
    allxs, allys = np.array(allxs), np.array(allys)
    print(allxs)
    a, b = np.polyfit(allxs, allys, 1)
#    slope, intercept, r2, p_value, std_err = scipy.stats.linregress(xs, ys)
    slope, intercept, r2, p_value, std_err = scipy.stats.linregress(allxs, allys)
    Min = np.amin(allxs)
    Max = np.amax(allxs)
    ax.plot([Min, Max], [a*Min+b, a*Max+b], 'k--')
    if drawPoint:
        ax.text(0.8, 0.2, 'r^2=%.2f' % r2, ha='center', va='center', transform=ax.transAxes, fontsize=fs)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/fixGPC5_%s_%sFlitStall.eps' % (app, 'node' if nodemetric else '')
    else:
        name = 'C:/Programming/monitoring/miniMD/fixGPC5_%s_%sFlitStall.png' % (app, 'node' if nodemetric else '')
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
    plt.close()

def flitTime(app, drawPoint, standardError, nodemetric, eps, useColor):
    df = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_nodeflit.csv' % app, index_col='run')
    if app == 'miniMD':
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv', index_col='run')
    elif app == 'lammps':
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s3.csv' % app, index_col='run')
    else:
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s.csv' % app, index_col='run')
    df[df <= 0] = None
    df2[df2 <= 0] = None
    if app == 'hacc':# remove outlier
        df[df <= 10000] = None
    if app == 'miniMD':
        df[df <= 10**7] = None
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,7.5
    if useColor:
        labels = ['I:Continuous', 'II:Spaced','III:Continuous\n+Congestor','IV:Spaced\n+Congestor']
    else:
        colors = ['saddlebrown']*4
        labels = ['Experiment run']
    legloc = 'upper center'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=legendfs, ncol=1)
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    xlabel = '%s Flits Per Second' % ('Ptile' if nodemetric else 'Ntile')
    ax.set_xlabel(xlabel, fontsize=fs)
    if app == 'miniMD':
        ax.set_ylabel('miniMD Execution Time (s)', fontsize=fs)
    elif app == 'nekbone':
        ax.set_ylabel('Nekbone Performance (MFLOPS)', fontsize=fs)
    elif app == 'hacc':
        ax.set_ylabel('HACC Execution Time (s)', fontsize=fs)
    elif app == 'graph500':
        ax.set_ylabel('BFS+SSSP Execution Time (s)', fontsize=fs)
    elif app == 'miniamr':
        ax.set_ylabel('miniAMR Execution Time (s)', fontsize=fs)
    elif app == 'lammps':
        ax.set_ylabel('LAMMPS Execution Time (s)', fontsize=fs)
    elif app == 'milc':
        ax.set_ylabel('MILC Execution Time (s)', fontsize=fs)
    elif app == 'qmcpack':
        ax.set_ylabel('QMCPACK Execution Time (s)', fontsize=fs)
    xs, ys, allxs, allys = [], [], [], []
    for idx, case in enumerate(['green','yellow','greenGPC','yellowGPC']):
        x, xerr = df[case].mean(), df[case].std()
        y, yerr = df2[case].mean(), df2[case].std()
        xs.append(x)
        ys.append(y)
        if standardError:
            xerr = xerr/math.sqrt(len(df[case]))
            yerr = yerr/math.sqrt(len(df2[case]))
        if not drawPoint:
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=3, capsize=5)
        for run in range(10):
            value = df.loc[run][case]
            value2 = df2.loc[run][case]
            if not math.isnan(value) and not math.isnan(value2):
                allxs.append(value)
                allys.append(value2)
                if drawPoint:
                    ax.plot(value, value2, '.', markersize=15, color=colors[idx])
    xs, ys = np.array(xs), np.array(ys)
    allxs, allys = np.array(allxs), np.array(allys)
    a, b = np.polyfit(allxs, allys, 1)
#    slope, intercept, r2, p_value, std_err = scipy.stats.linregress(xs, ys)
    slope, intercept, r2, p_value, std_err = scipy.stats.linregress(allxs, allys)
    Min = np.amin(allxs)
    Max = np.amax(allxs)
    ax.plot([Min, Max], [a*Min+b, a*Max+b], 'k--')
#    ax.set_xlim(0, 10**4)
    if app == 'qmcpack':
        ax.set_xticks([50000, 100000, 150000])
        ax.set_ylim(63, 85)
    if drawPoint:
        ax.text(0.8, 0.75, 'r^2=%.2f' % r2, ha='center', va='center', transform=ax.transAxes, fontsize=fs)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/fixGPC5_%s_%sFlitTime.eps' % (app, 'node' if nodemetric else '')
    else:
        name = 'C:/Programming/monitoring/miniMD/fixGPC5_%s_%sFlitTime.png' % (app, 'node' if nodemetric else '')
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
    plt.close()

def drawFix(metric, app, drawPoint, standardError, nodemetric, eps, addlgd=1):
    if app == 'miniMD':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv', index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/FixGPC5_%s.csv' % metric, index_col='run')
    elif app == 'lammps':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s3.csv' % app, index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_%s.csv' % (app, metric), index_col='run')
    else:
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s.csv' % app, index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_%s.csv' % (app, metric), index_col='run')
    if nodemetric:
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_node%s.csv' % (app, metric), index_col='run')
    df[df <= 0] = None
    df2[df2 <= 0] = None
    if app == 'qmcpack' and nodemetric:# remove outlier.
        df2[df2 > 100000] = None
    if app == 'hacc' and not nodemetric and metric == 'ratio':
        df2[df2 > 20] = None
#    if app == 'hacc' and not nodemetric and metric == 'stall':
#        df2[df2 > 5*10**7] = None
    fs, legendfs = 28,25
    figsizeX, figsizeY = 8,7.5
    labels = ['I:Continuous', 'II:Spaced','III:Continuous\n+Congestor','IV:Spaced\n+Congestor']
    if nodemetric:
        legloc = 'upper right'
    else:
        legloc = 'lower right'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    if addlgd:
        if nodemetric:
            lgd = fig.legend(handles=patches, labels=labels, 
                             loc=legloc, fontsize=legendfs, ncol=1)
        else:
            lgd = fig.legend(bbox_to_anchor=(1,0.12), handles=patches, labels=labels, 
                             loc=legloc, fontsize=legendfs, ncol=1)
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    if metric == 'ratio':
        xlabel = '%s Stall/Flit Ratio' % ('Ptile' if nodemetric else 'Ntile')
        if not addlgd:
            xlabel = 'Network port Stall/Flit Ratio'
    elif metric == 'stall':
        xlabel = '%s Stalls Per Second' % ('Ptile' if nodemetric else 'Ntile')
    ax.set_xlabel(xlabel, fontsize=fs)
    if app == 'miniMD':
        ax.set_ylabel('miniMD Execution Time (s)', fontsize=fs)
    elif app == 'nekbone':
        ax.set_ylabel('Nekbone Performance (MFLOPS)', fontsize=fs)
    elif app == 'hacc':
        ax.set_ylabel('HACC Execution Time (s)', fontsize=fs)
    elif app == 'graph500':
        ax.set_ylabel('BFS+SSSP Execution Time (s)', fontsize=fs)
    elif app == 'miniamr':
        ax.set_ylabel('miniAMR Execution Time (s)', fontsize=fs)
    elif app == 'lammps':
        ax.set_ylabel('LAMMPS Execution Time (s)', fontsize=fs)
    elif app == 'milc':
        ax.set_ylabel('MILC Execution Time (s)', fontsize=fs)
    elif app == 'qmcpack':
        ax.set_ylabel('QMCPACK Execution Time (s)', fontsize=fs)
    xs, ys, allxs, allys = [], [], [], []
    for idx, case in enumerate(['green','yellow','greenGPC','yellowGPC']):
        x, xerr = df2[case].mean(), df2[case].std()
        y, yerr = df[case].mean(), df[case].std()
        xs.append(x)
        ys.append(y)
        if standardError:
            xerr = xerr/math.sqrt(len(df2[case]))
            yerr = yerr/math.sqrt(len(df[case]))
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=3, capsize=5)
        for run in range(10):
            value = df2.loc[run][case]
            value2 = df.loc[run][case]
            if not math.isnan(value):
                if app == 'qmcpack' and value > 100000 and nodemetric:# remove outlier.
                    continue
#                if app == 'hacc' and value > 20 and not nodemetric:# remove outlier.
#                    continue
                allxs.append(value)
                allys.append(value2)
                if drawPoint:
                    ax.plot(value, value2, '.', markersize=15, color=colors[idx])
    xs, ys = np.array(xs), np.array(ys)
    a, b = np.polyfit(xs, ys, 1)
    slope, intercept, r2, p_value, std_err = scipy.stats.linregress(xs, ys)
#    slope, intercept, r2, p_value, std_err = scipy.stats.linregress(allxs, allys)
    Min = np.amin(xs)
    Max = np.amax(xs)
    if nodemetric or metric == 'ratio' or app != 'hacc':
        ax.plot([Min, Max], [a*Min+b, a*Max+b], 'k--')
#    ax.text(0.2, 0.7, 'r^2=%.2f' % r2, ha='center', va='center', transform=ax.transAxes, fontsize=fs)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/fixGPC5_%s_%s%s.eps' % (app, 'node' if nodemetric else '', metric)
    else:
        name = 'C:/Programming/monitoring/miniMD/fixGPC5_%s_%s%s.png' % (app, 'node' if nodemetric else '', metric)
    if addlgd:
        fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    else:
        fig.savefig(name)
    print('save to %s' % name)
#    plt.close()

def drawFix2(metric, app, drawPoint, standardError, nodemetric, eps):
    if app == 'miniMD':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv', index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/FixGPC5_%s.csv' % metric, index_col='run')
    elif app == 'lammps':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s3.csv' % app, index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_%s.csv' % (app, metric), index_col='run')
    else:
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFix_%s.csv' % app, index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/fixGPC_autotime_%s_%s.csv' % (app, metric), index_col='run')
    df[df <= 0] = None
    df2[df2 <= 0] = None
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,7.5
    labels = ['Without Congestor','With Congestor']
    legloc = 'center right'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=legendfs, ncol=1)
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    xlabel = '%s Stalls Per Second' % ('Ptile' if nodemetric else 'Ntile')
    ax.set_xlabel(xlabel, fontsize=fs)
    if app == 'miniMD':
        ax.set_ylabel('miniMD Execution Time (s)', fontsize=fs)
    elif app == 'nekbone':
        ax.set_ylabel('Nekbone Performance (MFLOPS)', fontsize=fs)
    elif app == 'hacc':
        ax.set_ylabel('HACC Execution Time (s)', fontsize=fs)
    elif app == 'graph500':
        ax.set_ylabel('BFS+SSSP Execution Time (s)', fontsize=fs)
    elif app == 'miniamr':
        ax.set_ylabel('miniAMR Execution Time (s)', fontsize=fs)
    elif app == 'lammps':
        ax.set_ylabel('LAMMPS Execution Time (s)', fontsize=fs)
    elif app == 'milc':
        ax.set_ylabel('MILC Execution Time (s)', fontsize=fs)
    elif app == 'qmcpack':
        ax.set_ylabel('QMCPACK Execution Time (s)', fontsize=fs)
    xs, ys, allxs, allys = [], [], [], []
    for idx, case in enumerate(['yellow','yellowGPC']):
        x, xerr = df2[case].mean(), df2[case].std()
        y, yerr = df[case].mean(), df[case].std()
        xs.append(x)
        ys.append(y)
        if standardError:
            xerr = xerr/math.sqrt(len(df2[case]))
            yerr = yerr/math.sqrt(len(df[case]))
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=3, capsize=5)
        for run in range(10):
            value = df2.loc[run][case]
            value2 = df.loc[run][case]
            if not math.isnan(value):
                allxs.append(value)
                allys.append(value2)
                if drawPoint:
                    ax.plot(value, value2, '.', markersize=15, color=colors[idx])
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/CompStall_%s_%s%s.png' % (app, 'node' if nodemetric else '', metric)
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
    plt.close()

def drawTwo():
    df = pd.read_csv('C:\Programming\monitoring\miniMD\jobparse.csv')
    metric = 'routerShare'
    fs = 20
    figsizeX, figsizeY = 8,8
    colors = ['forestgreen','orange','deepskyblue']
    labels = ['alloc_green', 'alloc_yellow']
    legloc = 'upper right'
    
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=fs, ncol=1)
    
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel(metric, fontsize=fs)
    ax.set_ylabel('Execution Time (s)', fontsize=fs)
    for i in [1,2]:
        sqrt = math.sqrt(len(df))
        for idx, row in df.iterrows():
            ax.plot(row['%s%d' % (metric, i)], row['execTime%d' % i], '.', markersize=15, color=colors[i-1])
        x, xerr = df['%s%d' % (metric, i)].mean(), df['%s%d' % (metric, i)].std()
        y, yerr = df['execTime%d' % i].mean(), df['execTime%d' % i].std()
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[i-1], ecolor=colors[i-1], elinewidth=2, capsize=5)
    
    fig.tight_layout()
    name = 'C:\Programming\monitoring\miniMD\Alloc_c32_ins5_%s.png' % metric
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def drawTwoTimeStall():
    df = pd.read_csv('C:\Programming\monitoring\miniMD\data_osu5.csv')
    df2 = pd.read_csv('C:\Programming\monitoring\miniMD\data_osu.csv')
    mode = 'all'# 'avg', 'all'
    fs = 20
    figsizeX, figsizeY = 8,8
    colors = ['forestgreen','orange','deepskyblue']
    labels = ['alloc_good', 'alloc_bad']#, 'no_OSU']
    legloc = 'upper right'
    
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=fs, ncol=1)
    
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('Avg Stall Count per second', fontsize=fs)
    ax.set_ylabel('Execution Time (s)', fontsize=fs)
    ax.set_xlim(0, 0.5*1e8)
#    ax.set_ylim(0, 200)
    for i in [1,2]:
        sqrt = math.sqrt(len(df))
        if mode == 'all':
            for idx, row in df.iterrows():
                ax.plot(row['avgStall%d' % i], row['execTime%d' % i], '.', markersize=15, color=colors[i-1])
        x, xerr = df['avgStall%d' % i].mean(), df['avgStall%d' % i].std()
        y, yerr = df['execTime%d' % i].mean(), df['execTime%d' % i].std()
#        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[i-1], ecolor=colors[i-1], elinewidth=2, capsize=5)
    
#    select = df2[df2['mode']=='no_OSU']
#    sqrt = math.sqrt(len(select))
#    if mode == 'all':
#        for idx, row in select.iterrows():
#            ax.plot(row['avgStall'], row['execTime'], '.', markersize=15, color=colors[2])
#    x, xerr = select['avgStall'].mean(), select['avgStall'].std()
#    y, yerr = select['execTime'].mean(), select['execTime'].std()
#    ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[2], ecolor=colors[2], elinewidth=2, capsize=5)

    fig.tight_layout()
    name = 'C:\Programming\monitoring\miniMD\Alloc_c32_ins5_%s.png' % mode
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def drawTimeStall(df):
    mode = 'all'# 'avg', 'all'
    metric = 'avgStall'# 'avgStall','congestion'
    fs = 20
    figsizeX, figsizeY = 9, 7
    colors = ['forestgreen','orange','deepskyblue']
    labels = ['OSU_32c', 'no_OSU']
    legloc = 'upper right'
    
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=fs, ncol=1)
    
    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('Avg Stall Count per second', fontsize=fs)
    ax.set_ylabel('Execution Time (s)', fontsize=fs)
    for i, label in enumerate(labels):
        select = df[df['mode']==label]
        sqrt = math.sqrt(len(select))
        if mode == 'all':
            for idx, row in select.iterrows():
                ax.plot(row[metric], row['execTime'], '.', markersize=15, color=colors[i])
        elif mode == 'avg':
            x, xerr = select[metric].mean(), select[metric].std()/sqrt
            y, yerr = select['execTime'].mean(), select['execTime'].std()/sqrt
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[i], ecolor=colors[i], elinewidth=2, capsize=5)

    fig.tight_layout()
    name = 'C:\Programming\monitoring\miniMD\OSU_stall.png'
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

def drawAvgStd():
    df = pd.read_csv('C:\Programming\monitoring\miniMD\OSU.csv', sep='\t')
    df.drop(axis=1, columns='Unnamed: 3', inplace=True)
    fs = 20
    figsizeX, figsizeY = 7, 8
    colors = ['orange','forestgreen','deepskyblue']
    labels = list(df.columns)
    legloc = 'top left'
    
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))

    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
    lgd = fig.legend(handles=patches, labels=labels, loc=legloc, fontsize=fs, ncol=1)
    
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_ylabel('Execution Time (s)', fontsize=fs)
#    ax.set_xlabel('Workload 1', fontsize=fs)
#    df.replace('random','RDN',inplace=True)
#    df.rename(columns={'jobSize':'Job sizes'}, inplace=True)
#    ax.set_ylim(0, ylim)
    ax.axes.set_xticks(list(range(len(labels))))
    ax.axes.xaxis.set_ticklabels(labels)
#    ax.axes.yaxis.set_ticklabels([])
    for i, label in enumerate(labels):
        y = df[label].mean()
        yerr = df[label].std()
        ax.bar(i, y, yerr=yerr, color=colors[i], error_kw=dict(ecolor='gray', lw=5, capsize=30, capthick=4))
        ax.text(i+0.05, y+5, '%.1f' % y, fontsize=fs)

    fig.tight_layout()
    name = 'C:\Programming\monitoring\miniMD\OSU.png'
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.close()

if __name__ == '__main__':
    main()
