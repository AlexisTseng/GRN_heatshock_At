#!/usr/bin/env python3
# script: HSPR_AZ_tidy.py
# author: Rituparna Goswami, Enrico Sandro Colizzi, Alexis Jiayi Zeng
script_usage="""
usage
    HSPR_AZ_plot.py -ids <importDataSuffix> [options]

version
    HSPR_AZ_v3.py 0.0.2 (alpha)

dependencies
    Python v3.9.7, Scipy v1.11.2, NumPy v1.22.2, viennarna v2.5.1, Matplotlib v3.5.1, pandas v2.1.0

description
    Re-organised the original code in functions. Introduced parser & flags for easy change of iteration numbers and time step. (HSPR_AZ_tidy.py)

    Stored parameters in a dictionary. Changed naming convention of output files (HSPR_AZ_v2.py)

    Combined plotting function. Introduced progress reporter. Output format either csv or pcl. Optional figure saving (HSPR_AZ_v3.py)

    Introduced options to directly import simulation data for plotting by updating opt. Changed saveGilData() from list*3 to list*2 (HSPR_AZ_v5.py) 

    Save data at customisable time step - default = 1. Changed Hill Coeff to 2. Histogram plotting cleaned (HSPR_AZ_v6.py, Nov 22nd 2023) 

################################################################

--importDataSuffix,-ids
    Suffix in data/parameter file, such as "2023-12-04_numIter2_Time500.00226623247835_HSstart10000_HSduration5000.csv". The file has "Exp3_Para_" or "Exp3_SimuData_" prefix

--saveFig,-sfg
    Whether to save the figures and plots generated. Default = True (default: 1)

--showFig,-shf
    Whether to show  the figures generated (default: 1)

--varAnalysis,-van
    whether to analysis variability by plotting histograms etc. (default: 0)

################################################################################

reference
    A.J.Zeng
    xxxxxxxxx
"""
#h

import time 
import re
import argparse as ap
import pickle
from pydoc import describe
from datetime import datetime
import traceback

import random
import math
import matplotlib.pyplot as plt #for my sanity
import numpy as np
import os
import csv
from os.path import join
import pandas as pd
import sys



def main(opt):
    print("Step1: Specify output directory")
    data_dir, plot_dir = dir_gen()

    print("Step2: Extracting Parameter Dictionary")
    param_dict, numberofiteration, end_time, hss, hsd, date = param_extract(data_dir, opt)
    #print(param_dict)

    print("Step3: Import Simulating Results")
    data_df, grouped_data, Rows, Columns = import_tidy_simuData(data_dir, numberofiteration, opt)
    #print(data_df)
    #exit()

    #plot_resTime_vs_preHSHSPR(data_df, hss, param_dict)
    #exit()

    print("Step 4: Generate Plot Name")
    name_suffix, diff_dict = genPlotName_nondefault(param_dict, numberofiteration, end_time, hss, hsd, date, opt)
    #print(diff_dict)
    #print(name_suffix)


    print("Step4: Plot Temporal Trajectories")
    ## Plot trajectories of all species for all iterations
    plot_allvsTime_separate(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, opt, hss, hsd, diff_dict)

    plot_allvsZoomInTime_separate(data_df, hss, hsd, plot_dir, numberofiteration,name_suffix, opt)

    plot_FMPMMPvsTime(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, hss, hsd, opt)

    #plot_FMPMMP_zoom(data_df, hss, hsd, plot_dir, numberofiteration,name_suffix, opt)

    #plot_A1BvsTime_separate(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, opt)
    ## Plot trajectory of total HSPR for all iterations
    #plot_totalHSPRvsTime_subplots(grouped_data, data_df, plot_dir, numberofiteration, name_suffix, hss, hsd, opt)
    ## Plot overlayed trajectory of A1 concentrations for all trajectory
    #plot_A1vsTime_asOne(grouped_data, plot_dir, numberofiteration, name_suffix, opt)

    print("Step 5: Variability Analysis")
    if bool(opt.van) == True:
        df_list = df_Processing_HS(data_df, plot_dir,hss,hsd, end_time, opt)
        bootstrap_HSPR_hist_overlap(df_list, plot_dir, name_suffix, opt)
        bootstrap_HSPR_hist_subplot(df_list, plot_dir, name_suffix, opt)
        totalHSPR_df_outlist = df_HSPR_stats(df_list, opt)
        plot_HSPR_hist(totalHSPR_df_outlist, plot_dir, name_suffix, opt)
        plot_CVsq_mean(totalHSPR_df_outlist, plot_dir, name_suffix, opt)



##########################################################################
## 1. Generate Output Directories
##########################################################################
def dir_gen():
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)

    data_dir = os.path.join(partiii_dir,"Ritu_simulation_data")
    if not os.path.isdir(data_dir): os.makedirs(data_dir, 0o777)
    plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)
    return data_dir, plot_dir


#######################################################################
## 2. Parameter specification
#######################################################################
def param_extract(data_dir, opt):
    para_csv_name = f"{data_dir}/Exp3_Para_{opt.ids}"
    param_dict = {}
    with open(para_csv_name, 'r') as param_file:
        csv_reader = csv.reader(param_file)
        headers = next(csv_reader)
        data = next(csv_reader)
        param_dict = dict(zip(headers, data))

    pattern = re.compile(r"(\d+-\d+-\d+)_numIter(\d+)_Time([\d.]+)_HSstart(\d+)_HSduration(\d+)\.(pcl|csv)")
    match = pattern.match(opt.ids)
    if match:
        date = str(match.group(1))
        numberofiteration = int(match.group(2))
        end_time = float(match.group(3))
        hss = int(match.group(4))
        hsd = int(match.group(5))
    else:
        print("Filename does not match the expected pattern.")
    #parameter = type("parameter", (object,), param_dict)
    #para = parameter()
    #print(f"para.h1:{para.h1}")
    #print(f"para.init_C_HSFA1_HSPR:{para.init_C_HSFA1_HSPR}")
    #print(f"para.numberofiteration:{para.numberofiteration}")
    return param_dict, numberofiteration, end_time, hss, hsd, date



#######################################################################
## 3. Import Simulation Data
#######################################################################
def import_tidy_simuData(data_dir, numberofiteration, opt):
    data_df = pd.read_csv(f"{data_dir}/Exp3_SimuData_{opt.ids}")
    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    #print(data_df)
    #print(data_df.shape)
    ### number of rows and columns for all iterations
    Rows = int(math.sqrt(numberofiteration))
    Columns = int(math.ceil(numberofiteration/ Rows))
    grouped_data = data_df.groupby('Iteration_Identifier')
    return data_df, grouped_data, Rows, Columns


#######################################################################
## 4. Generate Plot Name Suffix
#######################################################################

def genPlotName_nondefault_2(param_dict, numberofiteration, end_time, hss, hsd, date, opt):
    default_param_dict = {
        ## initial concentrations
        'init_HSFA1': 1,
        'init_HSPR': 2,
        'init_C_HSFA1_HSPR': 50,
        'init_MMP': 0,
        'init_FMP': 50,
        'init_C_HSPR_MMP': 50,
        'init_HSFA2': 1,
        'init_HSFB': 1,
        'Time': 0.0,
        ## Maximum expression level in Hill equation
        'a1': 10.0,
        'a2': 100.0,
        'a3': 5.0,
        'a4': 5.0,
        'a5': 5.0,
        'a6': 0.2, # refolding rate from MMP-HSPR
        'a7': 10,
        'a8': 5.0,
        ## Ka in Hill equation
        'h1': 1.0,
        'h2': 1.0,
        'h3': 1.0,
        'h4': 1.0,
        'h5': 1.0,
        'h6': 1.0,
        ## association rates
        'c1': 10.0,
        'c3': 0.5, #between MMP and HSPR
        ## decay rates
        'd1': 0.1, # decay path 1 of A1-HSPR
        'd3': 0.01, # dissociation rate of MMP-HSPR
        'd4_heat': 0.05,
        'd4_norm': 0.01,
        'Decay1': 0.01,
        'Decay2': 0.01, # decay of free HSPR
        'Decay3': 0.01,
        'Decay4': 0.01,
        'Decay6': 0.01,
        'Decay7': 0.01, # decay path 2 of A1-HSPR
        'Decay8': 0.01, # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage': 0.01,
        'hillcoeff': 1,
        'numberofiteration': numberofiteration,
        'hstart':hss,
        'hduration':hsd,
        'end_time':end_time
    }
    diff_dict = {}
    for (dk,dv) in default_param_dict.items():
        if str(param_dict[dk]) != str(dv):
            print(f"default: {dk}, {dv}")
            print(f"actual: {dk}, {param_dict[dk]}")
            diff_dict[dk] = param_dict[dk]

    name_suffix = f"{date}_numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}"
    for key, val in diff_dict.items():
        name_suffix += f"_{key}-{val}"
    return name_suffix

def genPlotName_nondefault(param_dict, numberofiteration, end_time, hss, hsd, date, opt):
    default_param_dict = {
        ## initial concentrations
        'init_HSFA1': 1,
        'init_HSPR': 2,
        'init_C_HSFA1_HSPR': 50,
        'init_MMP': 0,
        'init_FMP': 50,
        'init_C_HSPR_MMP': 50,
        'init_HSFA2': 1,
        'init_HSFB': 1,
        'Time': 0.0,
        ## Maximum expression level in Hill equation
        'a1': 10.0,
        'a2': 100.0,
        'a3': 5.0,
        'a4': 5.0,
        'a5': 5.0,
        'a6': 0.2, # refolding rate from MMP-HSPR
        'a7': 10,
        'a8': 5.0,
        ## Ka in Hill equation
        'h1': 1.0,
        'h2': 1.0,
        'h3': 1.0,
        'h4': 1.0,
        'h5': 1.0,
        'h6': 1.0,
        ## association rates
        'c1': 10.0,
        'c3': 0.5, #between MMP and HSPR
        ## decay rates
        'd1': 0.1, # decay path 1 of A1-HSPR
        'd3': 0.01, # dissociation rate of MMP-HSPR
        'd4_heat': 0.05,
        'd4_norm': 0.01,
        'Decay1': 0.01,
        'Decay2': 0.01, # decay of free HSPR
        'Decay3': 0.01,
        'Decay4': 0.01,
        'Decay6': 0.01,
        'Decay7': 0.01, # decay path 2 of A1-HSPR
        'Decay8': 0.01, # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage': 0.001,
        'hillcoeff': 2,
        'numberofiteration': numberofiteration,
        'hstart':hss,
        'hduration':hsd,
        'end_time':end_time
    }
    diff_dict = {}
    for (dk,dv) in default_param_dict.items():
        if str(param_dict[dk]) != str(dv):
            print(f"default: {dk}, {dv}")
            print(f"actual: {dk}, {param_dict[dk]}")
            diff_dict[dk] = param_dict[dk]

    name_suffix = f"{date}_numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}\n"
    for key, val in diff_dict.items():
        name_suffix += f"_{key}-{val}"
    return name_suffix, diff_dict



#######################################################################
## 5. Plotting Trajectories
#######################################################################

def plot_trajectory(ax, data_df, x_col, y_col_list, hss, hsd, Iteration_Identifier):
    for y_col in y_col_list:
        ax.plot(data_df[x_col], data_df[y_col], label=y_col, linewidth=1)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Protein copy number')
    ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.5)
    ax.set_title(f"{Iteration_Identifier}")
    ax.legend(loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def saveFig(plot_dir, name_suffix, opt, prefix):
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/{prefix}_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/{prefix}_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f" save figure {opt.sfg == True}")

def plot_allvsTime_separate(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, opt, hss, hsd, diff_dict):

    print(" Plot trajectories of all species for all iterations")
    conc_col = data_df.drop(columns = ["time", "Iteration_Identifier"])

    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(ax, data_df, 'time', conc_col, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax, group_data, 'time', conc_col, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    #fig.suptitle('Plot of all concentrations vs time for all iterations separately', fontsize=16, y = 1)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(plot_dir, name_suffix, opt, prefix ='allConcTraj')
    if bool(opt.shf) == True: plt.show()
    plt.close()

def plot_allvsZoomInTime_separate(data_df, hss, hsd, plot_dir, numberofiteration,name_suffix, opt):

    print(" Zoomed In Trajectory Around HeatShock")
    cut_data_df = data_df[(data_df['time'] >= hss -50) & (data_df['time'] <= hss + hsd + 100)]
    conc_col = cut_data_df.drop(columns = ["time", "Iteration_Identifier"])
    
    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(ax, cut_data_df, 'time', conc_col, hss, hsd, "iteration 0")
    else:
        grouped_data = cut_data_df.groupby('Iteration_Identifier')
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax, group_data, 'time', conc_col, hss, hsd, Iteration_Identifier=Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Zoomed In Trajectories, Around HeatShock')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(plot_dir, name_suffix, opt, prefix = "Zoom_allConcTraj")

    if bool(opt.shf) == True: plt.show()
    plt.close()

def plot_FMPMMPvsTime(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, hss, hsd, opt):

    print(" Plot trajectories of Proteins and Regulators")
    reg_conc_col = data_df.drop(columns = ["time", "Iteration_Identifier",'FMP','MMP'])
    
    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
        plot_trajectory(ax[0], data_df, 'time', ['FMP','MMP'], hss, hsd, "iteration 0")
        plot_trajectory(ax[1], data_df, 'time', reg_conc_col, hss, hsd, "iteration 0")

    else:
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], group_data, 'time', ['FMP','MMP'], hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], group_data, 'time', reg_conc_col, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(plot_dir, name_suffix, opt, prefix ='ProReg')

    if bool(opt.shf) == True: plt.show()
    plt.close()





def plot_FMPMMP_zoom(data_df, hss, hsd, plot_dir, numberofiteration,name_suffix, opt):
    print(" Zoomed In Protein & Regulator Trajectory Around HeatShock")
    cut_data_df = data_df[(data_df['time'] >= hss -50) & (data_df['time'] <= hss + hsd + 100)]
    reg_conc_col = data_df.drop(columns = ["time", "Iteration_Identifier",'FMP','MMP'])
    
    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
        plot_trajectory(ax[0], cut_data_df, 'time', ['FMP','MMP'], hss, hsd, "iteration 0")
        plot_trajectory(ax[1], cut_data_df, 'time', reg_conc_col, hss, hsd, "iteration 0")
    else:
        grouped_data = cut_data_df.groupby('Iteration_Identifier')
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], group_data, 'time', ['FMP','MMP'], hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], group_data, 'time', reg_conc_col, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    #fig.suptitle('Zoomed In Trajectories, Around HeatShock')
    plt.tight_layout()

    saveFig(plot_dir, name_suffix, opt, prefix ='ProRegZoom')
    if bool(opt.shf) == True: plt.show()
    plt.close()



def plot_A1BvsTime_separate(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, opt):

    print(" Plot trajectories of all species for all iterations")
    conc_col = ['HSFA1','HSFB','C_HSFA1_HSPR']

    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        for species in conc_col:
            ax.plot(data_df['time'], data_df[f'{species}'], label ='{}'.format(species), linewidth = 1) 
        ax.set_xlabel('Time')
        ax.set_ylabel('Concentration')
        ax.legend(loc="upper right")
        ax.set_title(f"iteration 0")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
        # Adjust the figure size to accommodate the legend
        plt.subplots_adjust(right=0.8)  # Increase the right margin

    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):# Now 'ax' is a 1D array, and you can iterate over it
            for species in conc_col:
                ax.plot(group_data['time'], group_data[f'{species}'], label ='{}'.format(species), linewidth = 1) 
            ax.set_xlabel('Time')
            ax.set_ylabel('Concentration')
            ax.legend(loc="upper right")
            ax.set_title(f"{Iteration_Identifier}")
            # Move the legend outside the plot
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
            # Adjust the figure size to accommodate the legend
            plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Plot of all concentrations vs time for all iterations separately')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/A1-BConcTraj_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/A1-BConcTraj_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f" save figure {opt.sfg == True}")

    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_totalHSPRvsTime_subplots(grouped_data, data_df, plot_dir, numberofiteration, name_suffix, hss, hsd, opt):
    print("Plot trajectory of total HSPR for all iterations")
    if numberofiteration == 1:
        # If only one subplot, create a single subplot without flattening
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(data_df['time'], data_df['totalHSPR']) 
        ax.set_xlabel('Time (hour)')
        ax.set_ylabel('totalHSPR')
        ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.5)
        ax.legend(loc="upper right")
        ax.set_title(f"iteration 0")
    else:
        # If more than one subplot, create a subplot grid
        fig, ax = plt.subplots(nrows=numberofiteration, figsize=(15, 10))
        ax = ax.flatten()

        # Iterate through grouped data and plot on each subplot
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):
            ax.plot(group_data['time'], group_data['totalHSPR'], label='{}'.format(Iteration_Identifier))
            ax.set_xlabel('Time (hour)')
            ax.set_ylabel('totalHSPR')
            ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.5)
            ax.set_title(f"{Iteration_Identifier}")
            ax.legend(loc="upper right")

        # Set the title for the entire plot
    fig.suptitle('Plot of time vs total HSPR for all Iterations separately')
    plt.tight_layout()
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/totalHSPRtrajec_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/totalHSPRtrajec_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

    if bool(opt.shf) == True: plt.show()
    plt.close()



def plot_A1vsTime_asOne(grouped_data, plot_dir, numberofiteration, name_suffix, opt):
    fig, ax1 = plt.subplots(figsize=(15,10))  # Set the figure size 
    for Iteration_Identifier, group_data in grouped_data:
        ax1.plot(group_data['time'], group_data['HSFA1'], label='{}'.format(Iteration_Identifier))
        ax1.set_xlabel('time')
        ax1.legend()
        ax1.set_ylabel('HSFA1')
        ax1.set_title('Plot of HSFA1 vs time for all Iterations')
    plt.tight_layout()
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/A1TrajMerged_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/A1TrajMerged_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
    if bool(opt.shf) == True: plt.show()
    plt.close()





#######################################################################
## 5. Variability Analysis
#######################################################################

def df_Processing_HS(data_df, plot_dir,hss,hsd, end_time, opt):
    ss1_start = 1000
    ss1_end = int(hss)
    ssHS_start = int(hss) + 100
    ssHS_end = int(hss) + int(hsd)
    ss3_start = ssHS_end + 500
    ss3_end = end_time
    print(f"hss:{hss}, hsd: {hsd}")
    print(f"ss1: {ss1_start} - {ss1_end} \nssHS: {ssHS_start} - {ssHS_end} \nss3:{ss3_start} - {ss3_end} ")

    ss1_df = data_df[(data_df['time'] >= ss1_start) & (data_df['time'] <= ss1_end)]
    ssHS_df = data_df[(data_df['time'] >= ssHS_start) & (data_df['time'] <= ssHS_end)]
    ss3_df = data_df[(data_df['time'] >= ss3_start) & (data_df['time'] <= ss3_end)]

    #print("ss1_df")
    #print(ss1_df)
    #print(ss1_df.shape)
    #print("ssHS_df")
    #print(ssHS_df)
    #print(ssHS_df.shape)
    #print("ss3_df")
    #print(ss3_df)
    #print(ss3_df.shape)

    df_list = [ss1_df, ssHS_df, ss3_df]
    return df_list


def df_Processing_HS_1(data_df, plot_dir,hss,hsd, end_time, opt):
    ss1_start = 1000
    ss1_end = int(hss)
    ssHS_start = int(hss) + 100
    ssHS_end = int(hss) + int(hsd)
    ss3_start = ssHS_end + 500
    ss3_end = end_time
    print(f"hss:{hss}, hsd: {hsd}")
    print(f"ss1: {ss1_start} - {ss1_end} \nssHS: {ssHS_start} - {ssHS_end} \nss3:{ss3_start} - {ss3_end} ")

    ss1_df = data_df[(data_df['time'] >= ss1_start) & (data_df['time'] <= ss1_end)].groupby('Iteration_Identifier')['totalHSPR']
    ssHS_df = data_df[(data_df['time'] >= ssHS_start) & (data_df['time'] <= ssHS_end)].groupby('Iteration_Identifier')['totalHSPR']
    ss3_df = data_df[(data_df['time'] >= ss3_start) & (data_df['time'] <= ss3_end)].groupby('Iteration_Identifier')['totalHSPR']

    #print("ss1_df")
    #print(ss1_df)
    #print(ss1_df.shape)
    #print("ssHS_df")
    #print(ssHS_df)
    #print(ssHS_df.shape)
    #print("ss3_df")
    #print(ss3_df)
    #print(ss3_df.shape)

    df_list = [ss1_df, ssHS_df, ss3_df]
    totalHSPR_df_outlist = []
    for grouped_df in df_list:
        result_df = grouped_df.agg(['mean','std'])
        result_df['cv'] = result_df['std'] / result_df['mean']
        result_df.columns = ['mean_totalHSPR', 'std_totalHSPR', 'cv_totalHSPR']
        result_df.reset_index(inplace=True)
        totalHSPR_df_outlist.append(result_df)
    
    return totalHSPR_df_outlist

def df_HSPR_stats(df_list, opt):
    totalHSPR_df_outlist = []
    for df in df_list:
        df = df.groupby('Iteration_Identifier')['totalHSPR']
        result_df = df.agg(['mean','std'])
        result_df['cv'] = result_df['std'] / result_df['mean']
        result_df.columns = ['mean_totalHSPR', 'std_totalHSPR', 'cv_totalHSPR']
        result_df.reset_index(inplace=True)
        totalHSPR_df_outlist.append(result_df)
    return totalHSPR_df_outlist

def plot_CVsq_mean(totalHSPR_df_outlist, plot_dir, name_suffix, opt):
    print("plot HSPR CV vs Mean")
    fig, axes = plt.subplots(nrows=1, ncols=3, sharey=True, figsize=(15,5))
    for i, (df,ax) in enumerate(zip(totalHSPR_df_outlist,axes)):
        ax.scatter(df['mean_totalHSPR'], df['cv_totalHSPR'])
        ax.set_xlabel('Mean Total HSPR')
        ax.set_ylabel('CV')
        if i == 0:
            ax.set_title("Before HS")
        elif i ==1: 
            ax.set_title("During HS")
        elif i ==2:
            ax.set_title("After HS")
        else: print("i exception in function plot_CVsq_mean")
    fig.suptitle('Variability of Total HSPR')
    plt.tight_layout()

    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/CV-Mean_TotalHSPR_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/CV-Mean_TotalHSPR_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f"save figure {opt.sfg == True}")

    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_HSPR_hist(totalHSPR_df_outlist, plot_dir, name_suffix, opt):

    print("Plot total HSPR histogram")
    fig = plt.figure(figsize=(12, 8))
    #plt.hist(ss1_df['mean_totalHSPR'], label="before HS", density=True,alpha=0.50, color="blue")
    #plt.hist(ssHS_df['mean_totalHSPR'], label="during HS", density=True, alpha=0.50, color="red")
    #plt.hist(ss3_df['mean_totalHSPR'], label="after HS", density=True, alpha=0.50, color="orange")
    label_list = ["before HS", "during HS", "after HS"]

    for df, label in zip(totalHSPR_df_outlist, label_list):
        plt.hist(df['mean_totalHSPR'], bins = range(0,200,1), label=label, density=True, alpha=0.50)

    plt.title("Distribution of mean total HSPR")
    plt.xlabel("total HSPR")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Adjust the figure size to accommodate the legend
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    plt.tight_layout()

    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/Hist_TotalHSPR_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{plot_dir}/Hist_TotalHSPR_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f"save figure {opt.sfg == True}")
    if bool(opt.shf) == True: plt.show()
    plt.close()


def bootstrap_HSPR_hist_overlap(df_list, plot_dir, name_suffix, opt):
    HSPR_list = []
    for df in df_list:
        HSPR_conc = random.choices(df['totalHSPR'].tolist(), k=2000)
        HSPR_list.append(HSPR_conc)
    print("Plot total HSPR histogram")
    fig = plt.figure(figsize=(12, 6))
    label_list = ["before HS", "during HS", "after HS"]
    #for list, label in zip(HSPR_list, label_list):
    #    plt.hist(list, bins, label=label, density=True, alpha=0.50)
    plt.hist(HSPR_list, bins = range(0, 250, 1), label = label_list, density=True, alpha=0.50, histtype='stepfilled')

    plt.title("Distribution of HSPR conc, bootstrapped")
    plt.xlabel("HSPR")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Adjust the figure size to accommodate the legend
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    plt.tight_layout()

    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/HistOverlap_bootstrapHSPR_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{plot_dir}/HistOverlap_bootstrapHSPR_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f"save figure {opt.sfg == True}")
    if bool(opt.shf) == True: plt.show()
    plt.close()

def bootstrap_HSPR_hist_subplot(df_list, plot_dir, name_suffix, opt):
    HSPR_list = []
    for df in df_list:
        HSPR_conc = random.choices(df['totalHSPR'].tolist(), k=2000)
        HSPR_list.append(HSPR_conc)
    print("Plot total HSPR histogram")
    label_list = ["before HS", "during HS", "after HS"]
    color_list = ['green','red','blue']
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,sharey=True, figsize=(15,12))
    for i, (HSPR_conc, label, ax, color) in enumerate(zip(HSPR_list, label_list, axes, color_list)):
        ax.hist(HSPR_conc, bins = range(0, 250, 1), label = label, density=True, alpha=0.50, histtype='stepfilled', color = color)
        ax.set_ylabel("Frequency")
        if i == 0:
            ax.set_title("Before HS")
        elif i ==1: 
            ax.set_title("During HS")
        elif i ==2:
            ax.set_title("After HS")
        else: print("i exception in function plot_CVsq_mean")
    fig.suptitle('Distribution of HSPR conc, bootstrapped')
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    plt.xlabel('HSPR level')
    plt.tight_layout()

    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/HistSub_bootstrapHSPR_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{plot_dir}/HistSub_bootstrapHSPR_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f"save figure {opt.sfg == True}")
    if bool(opt.shf) == True: plt.show()
    plt.close()



#############################################################################
## Response time analysis
#############################################################################

def plot_resTime_vs_preHSHSPR(data_df, hss, param_dict):
    preHS_totalHSPR = []
    preHS_HSPR_A1 = []
    preHS_HSPR_MMP = []
    res_time = []
    numberofiteration = int(param_dict["numberofiteration"])
    #for i in range(numberofiteration + 1):
    PreHS_row = data_df[(data_df["Iteration_Identifier"] == 1)].iloc[100]
    print(PreHS_row["time"])







###############################################################################
## Small functions
###############################################################################


def loadData(fname):
    ## load data with pickle
    pcl_file = os.path.join(f"{fname}")
    with open(pcl_file, "r+b") as pcl_in:
        pcl_data = pickle.load(pcl_in)
    return pcl_data

def saveData(pcl_data, fname):
    ## save data with pickle
    pcl_file = os.path.join(f"{fname}")
    with open(pcl_file, "w+b") as pcl_out:
        pickle.dump(pcl_data, pcl_out , protocol=4)

def get_unique_filename(base_filename):
    counter = 1
    new_filename = base_filename

    # Keep incrementing the counter until a unique filename is found
    while os.path.exists(new_filename):
        counter += 1
        filename, extension = os.path.splitext(base_filename)
        new_filename = f"{filename}-run{counter}{extension}"

    return new_filename








##################
## parser
################################################################################

class options(object):
    def __init__(self, **data):
        self.__dict__.update((k,v) for k,v in data.items())
    def plot(self, sep):
        ldat = sep.join([f"{var}" for key,var in vars(self).items()])
        return ldat

if __name__ == "__main__":

    ############################################################################
    ## get time and save call
    sscript = sys.argv[0]
    start_time = time.time()
    current_time = time.strftime('%x %X')
    scall = " ".join(sys.argv[1:])
    with open(f"{sscript}.log", "a") as calllog:
        calllog.write(f"Start : {current_time}\n")
        calllog.write(f"Script: {sscript}\n")
        calllog.write(f"Call  : {scall}\n")
    print(f"Call: {scall}")
    print(f"Status: Started at {current_time}")
    ############################################################################
    ## transform string into int, float, bool if possible
    def trans(s):
        if isinstance(s, str):
            if s.lower() == "true":
                return True
            elif s.lower() == "false":
                return False
            try: return int(s)
            except ValueError:
                try: return float(s)
                except ValueError:
                    if s in ["True", "False"]: return s == "True"
                    else: return s
        else: return s
    ############################################################################
    ## save documentation
    rx_text = re.compile(r"\n^(.+?)\n((?:.+\n)+)",re.MULTILINE)
    rx_oneline = re.compile(r"\n+")
    rx_options = re.compile(r"\((.+?)\:(.+?)\)")
    help_dict, type_dict, text_dict, mand_list = {}, {}, {}, []
    for match in rx_text.finditer(script_usage):
        argument = match.groups()[0].strip()
        text = " ".join(rx_oneline.sub("",match.groups()[1].strip()).split())
        argopts = {"action":"store", "help":None, "default":None, "choices":None}
        for option in rx_options.finditer(text):
            key = option.group(1).strip()
            var = option.group(2).strip()
            if var == "False": argopts["action"] = "store_true"
            if var == "True": argopts["action"] = "store_false"
            if key == "choices": var = [vs.strip() for vs in var.split(",")]
            if key == "default": var = trans(var)
            argopts[key] = var
        if argopts["default"]: add_default = f" (default: {str(argopts['default'])})"
        else: add_default = ""
        argopts["help"] = rx_options.sub("",text).strip()+add_default
        argnames = argument.split(",")
        if len(argnames) > 1:
            if argopts["default"] == None:
                mand_list.append(f"{argnames[1][1:]}")
            type_dict[f"{argnames[1][1:]}"] = argopts["default"]
            argopts["argshort"] = argnames[1]
            help_dict[argnames[0]] = argopts
        else:
            text_dict[argnames[0]] = argopts["help"]
    ############################################################################
    ## get arguments
    if text_dict["dependencies"]:
        desc = f"{text_dict['description']} (dependencies: {text_dict['dependencies']})"
    else:
        desc = text_dict['description']
    p = ap.ArgumentParser(prog=sscript, prefix_chars="-", usage=text_dict["usage"],
                          description=desc, epilog=text_dict["reference"])
    p.add_argument("-v", "--version", action="version", version=text_dict["version"])
    for argname,argopts in help_dict.items():
        argshort = argopts["argshort"]
        if argopts["choices"]:
            p.add_argument(argshort, argname,            dest=f"{argshort[1:]}",\
                           action=argopts["action"],     help=argopts["help"],\
                           default=argopts["default"],   choices=argopts["choices"])
        else:
            p.add_argument(argopts["argshort"], argname, dest=f"{argshort[1:]}",\
                           action=argopts["action"],     help=argopts["help"],\
                           default=argopts["default"])
    p._optionals.title = "arguments"
    opt = vars(p.parse_args())
    ############################################################################
    ## validate arguments
    if None in [opt[mand] for mand in mand_list]:
        print("Error: Mandatory arguments missing!")
        print(f"Usage: {text_dict['usage']} use -h or --help for more information.")
        sys.exit()
    for key,var in opt.items():
        if key not in mand_list:
            arg_req, arg_in = type_dict[key], trans(var)
            if type(arg_req) == type(arg_in):
                opt[key] = arg_in
            else:
                print(f"Error: Argument {key} is not of type {type(arg_req)}!")
                sys.exit()
    ############################################################################
    ## add log create options class
    opt["log"] = True
    copt = options(**opt)
    ############################################################################
    ## call main function
    try:
        #saved = main(opt)
        saved = main(copt)
    except KeyboardInterrupt:
        print("Error: Interrupted by user!")
        sys.exit()
    except SystemExit:
        print("Error: System exit!")
        sys.exit()
    except Exception:
        print("Error: Script exception!")
        traceback.print_exc(file=sys.stderr)
        sys.exit()
    ############################################################################
    ## finish
    started_time = current_time
    elapsed_time = time.time()-start_time
    elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    current_time = time.strftime('%x %X')
    if saved:
        with open(f"{sscript}.log", "a") as calllog,\
             open(os.path.join(saved,f"call.log"), "a") as dirlog:
            calllog.write(f"Save  : {os.path.abspath(saved)}\n")
            calllog.write(f"Finish: {current_time} in {elapsed_time}\n")
            ## dirlog
            dirlog.write(f"Start : {started_time}\n")
            dirlog.write(f"Script: {sscript}\n")
            dirlog.write(f"Call  : {scall}\n")
            dirlog.write(f"Save  : {os.path.abspath(saved)}\n")
            dirlog.write(f"Finish: {current_time} in {elapsed_time}\n")
    print(f"Status: Saved at {saved}")
    print(f"Status: Finished at {current_time} in {elapsed_time}")
    sys.exit(0)

