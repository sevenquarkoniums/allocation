# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 13:25:05 2020

@author: seven
"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
from statistics import median
import numpy as np
import scipy
from scipy import stats

def main():
    # fig. 4
    timeNormalizedReshape(eps=1)
    
    # fig. 5
#    for app in ['lammps']:#, 'hpcg', 'miniMD', 'nekbone', 'hacc', 'graph500', 'miniamr', 'milc', 'qmcpack']:
#        drawBox(app, normalize=1, eps=1)
    
    # fig. 6, 7
#    for metric in ['ratio','stall']:
#        for app in ['hacc', 'qmcpack', 'miniMD', 'miniamr', 'lammps', 'milc', 'nekbone', 'graph500']:
#            drawFix(metric, app, drawPoint=0, standardError=1, nodemetric=0, eps=1)
    
    # fig. 8
#    for metric in ['ratio','stall']:
#        for app in ['qmcpack', 'miniMD', 'miniamr', 'lammps', 'milc', 'hacc']:
#            drawFix(metric, app, drawPoint=0, standardError=1, nodemetric=1, eps=1)

    # fig. 9
#    for app in ['hacc', 'qmcpack', 'miniMD', 'miniamr']:
#        flitTime(app=app, drawPoint=1, standardError=0, nodemetric=1, eps=1, useColor=0)






#    for app in ['hacc', 'qmcpack', 'miniMD', 'miniamr']:
#        flitStall(app=app, drawPoint=1, standardError=0, nodemetric=1, eps=1, useColor=0)
#    drawTwo()
#    drawTwoTimeStall()
#    timeNormalized()
#    drawCADD('milc')
#    drawFixSpan()
#    drawAvgStd()

def timeNormalizedReshape(eps):
    apps = ['graph500', 'hacc', 'hpcg', 'lammps', 'milc', 'miniamr', 'miniMD', 'qmcpack']
    appsNames = ['Graph500', 'HACC', 'HPCG', 'LAMMPS', 'MILC', 'miniAMR', 'miniMD', 'QMCPACK']
    colors = ['forestgreen','orange','darkgreen','chocolate']
    boxColors = []
    for i in range(len(apps)):
        boxColors += ['forestgreen','orange','darkgreen','chocolate']
    fs = 20
    figsizeX, figsizeY = 20,6
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
            normed = [x/med[app] for x in validResults]
            results.append(normed)
            positions.append(idx*4+jdx)
    b = ax.boxplot(results, whis=100, patch_artist=True, positions=positions)
    ymin, ymax = ax.get_ylim()
    for patch, color in zip(b['boxes'], boxColors):
        patch.set_facecolor(color)
    ax.set_xlabel('Applications in different experiment settings', fontsize=fs)
    ax.set_ylabel('Normalized Execution Time\n(separately for each application)', fontsize=fs)
    for x in range(len(apps)-1):
        splitPosition = x*4+3.5
        ax.plot([splitPosition, splitPosition], [ymin, ymax], 'k-', linewidth=1)
#    ax.set_ylim(ymin, ymax)
    ax.set_ylim(0.4, 3.1)
    ax.text(14.3, 1.25, '(exceed\n  range)', fontsize=12)
    ax.set_xticks([x*4+1.5 for x in range(len(apps))])
    ax.set_xticklabels(appsNames, fontsize=fs)
    patches = []
    for color in colors:
        patches.append(mpatches.Patch(color=color, label=''))
    labels = ['Continuous', 'Spaced','Continuous+Congestor','Spaced+Congestor']
    lgd = ax.legend(handles=patches, labels=labels, loc='upper left', fontsize=fs, ncol=4)
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
    figsizeX, figsizeY = 7,7
    colors = ['forestgreen','orange','darkgreen','chocolate']
    labels = ['Continuous', 'Spaced','Continuous\n+GPCNET','Spaced\n+GPCNET']
    legloc = 'upper right'
    plt.rc('xtick', labelsize=fs)
    plt.rc('ytick', labelsize=fs)
    fig, ax = plt.subplots(1, 1, figsize=(figsizeX,figsizeY))
#    ax.xaxis.grid(linestyle='--', color='black')
    ax.yaxis.grid(linestyle='--', color='black')
    ax.set_xlabel('Experiment settings', fontsize=fs)
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
    if normalize and app == 'lammps':
        ax.set_yticks([0,1,2,4,6,8,10,12])
        ax.set_ylim(0, 13.5)
    for patch, color in zip(b['boxes'], colors):
        patch.set_facecolor(color)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/fixGPC5_%s.eps' % app
    else:
        name = 'C:/Programming/monitoring/miniMD/fixGPC5_%s.png' % app
    fig.savefig(name)

def drawCADD(app):
    df = pd.read_csv('C:/Programming/monitoring/miniMD/CADDprocess_%s.csv' % app)
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,8
    colors = ['forestgreen','orange','darkgreen','chocolate']
    labels = ['Low-stall Allocation', 'High-stall Allocation']
    legloc = 'right'
    patches = []
    for i in range(len(labels)):
        patches.append(mpatches.Patch(color=colors[i], label=''))
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
    for idx, case in enumerate(['green','yellow']):
        y, yerr = df[case].mean(), df[case].std()
        yerr = yerr#/math.sqrt(len(df[case]))
        ax.bar(idx, y, color=colors[idx], width=0.5, yerr=yerr, ecolor='black', capsize=10)
    fig.tight_layout()
    name = 'C:/Programming/monitoring/miniMD/CADD_%s.png' % (app)
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
    plt.close()

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
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=2, capsize=5)
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
        colors = ['forestgreen','orange','darkgreen','chocolate']
        labels = ['Continuous', 'Spaced','Continuous\n+Congestor','Spaced\n+Congestor']
    else:
        colors = ['dodgerblue']*4
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
            ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=2, capsize=5)
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
    if drawPoint:
        ax.text(0.8, 0.8, 'r^2=%.2f' % r2, ha='center', va='center', transform=ax.transAxes, fontsize=fs)
    fig.tight_layout()
    if eps:
        name = 'C:/Programming/monitoring/miniMD/eps/fixGPC5_%s_%sFlitTime.eps' % (app, 'node' if nodemetric else '')
    else:
        name = 'C:/Programming/monitoring/miniMD/fixGPC5_%s_%sFlitTime.png' % (app, 'node' if nodemetric else '')
    fig.savefig(name, bbox_extra_artists=(lgd,), bbox_inches='tight')
    print('save to %s' % name)
    plt.close()

def drawFix(metric, app, drawPoint, standardError, nodemetric, eps):
    if app == 'miniMD':
        df = pd.read_csv('C:/Programming/monitoring/miniMD/resultFixGPC5.csv', index_col='run')
        df2 = pd.read_csv('C:/Programming/monitoring/miniMD/FixGPC5_%s.csv' % metric, index_col='run')
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
    fs, legendfs = 25,20
    figsizeX, figsizeY = 8,7.5
    colors = ['forestgreen','orange','darkgreen','chocolate']
    labels = ['Continuous', 'Spaced','Continuous\n+Congestor','Spaced\n+Congestor']
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
    if metric == 'ratio':
        xlabel = '%s Stall/Flit Ratio' % ('Ptile' if nodemetric else 'Ntile')
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
        ax.errorbar(x, y, yerr=yerr, xerr=xerr, fmt='x', markersize=15, color=colors[idx], ecolor=colors[idx], elinewidth=2, capsize=5)
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
