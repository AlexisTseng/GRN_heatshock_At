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
    Suffix in data/parameter file, such as from SA: '2024-02-24_step50_time900_hss600_hsd50/159.461275414781_SimuData_2024-02-28_replaceA1_numIter20_Time1000.0026486889622_HSstart600_HSduration30' // or from simuData: '2023-11-26_numIter1_Time20000.002736231276_HSstart10000_HSduration5000' // or from variance analysis 'simuData_for_varAna/159.461275414781_replaceA1_numIter5_Time600.0008946411682_HSstart400_HSduration1'..... All without .csv

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
    data_dir, param_rootdir, varPara_dir, varData_dir, partiii_dir = dir_gen()

    print("Step2: Extracting Parameter Dictionary")
    param_dict, plot_dir, model_name, opt = load_Param_fromDataFile(param_rootdir, data_dir, partiii_dir, varPara_dir, varData_dir, opt)
    #end_time, hss, hsd, opt, date, model_name = info_from_param_dict(param_dict, opt)

    print("Step3: Import Simulating Results")
    data_df, grouped_data, numberofiteration, hss, hsd, end_time, opt = import_tidy_simuData(data_dir, partiii_dir, param_rootdir, opt, model_name)

    print("Step 4: Generate Plot Name")
    name_suffix, diff_dict = genPlotName_nondefault(param_dict, numberofiteration, end_time, hss, hsd, opt)
    #print(diff_dict)
    #print(name_suffix)


    print("Step4: Plot Temporal Trajectories")
    #plotReactionRate(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, opt, hss, hsd)
    #plot_FMPMMPvsTime_2(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, hss, hsd, opt)
    #plot_FMPMMPvsTime_2_overlayed(data_df, grouped_data, plot_dir, numberofiteration,name_suffix, hss, hsd, opt)
    #plot_FMPMMP_zoom(data_df, hss, hsd, plot_dir, numberofiteration,name_suffix, opt)



    print("Step 5: Variability Analysis")
    if bool(opt.van) == True:
        df_list = df_Processing_HS(data_df, hss,hsd, end_time, opt)
        maxVal = data_df['totalHSPR'].max()
        bootstrap_HSPR_hist_overlap(df_list, plot_dir, maxVal, opt)
        exit()
        bootstrap_HSPR_hist_subplot(df_list, plot_dir, maxVal, opt)
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
    #plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    #if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)
    param_rootdir = os.path.join(partiii_dir,"Param_Optimisation")
    if not os.path.isdir(param_rootdir): os.makedirs(param_rootdir, 0o777)

    varPara_dir = os.path.join(partiii_dir,"param_for_varAna")
    if not os.path.isdir(varPara_dir): os.makedirs(varPara_dir, 0o777)
    varData_dir = os.path.join(partiii_dir,"simuData_for_varAna")
    if not os.path.isdir(varData_dir): os.makedirs(varData_dir, 0o777)
    return data_dir, param_rootdir, varPara_dir, varData_dir, partiii_dir

#######################################################################
## 2. Parameter specification
#######################################################################
def param_extract(data_dir, opt):
    if os.path.exists(f"{data_dir}/Exp3_Para_{opt.ids}"):
        para_csv_name = f"{data_dir}/Exp3_Para_{opt.ids}"
        model_name = model_from_date(para_csv_name)
    elif os.path.exists(f"{data_dir}/replaceA1_Para_{opt.ids}"):
        para_csv_name = f"{data_dir}/replaceA1_Para_{opt.ids}"
        model_name = "replaceA1"
    elif os.path.exists(f"{data_dir}/woA2_Para_{opt.ids}"):
        para_csv_name = f"{data_dir}/woA2_Para_{opt.ids}"
        model_name = "woA2"
    elif os.path.exists(f"{data_dir}/d1upCons_Para_{opt.ids}"):
        para_csv_name = f"{data_dir}/d1upCons_Para_{opt.ids}"
        model_name = "d1upCons"

    with open(para_csv_name, 'r') as param_file:
        csv_reader = csv.reader(param_file)
        headers = next(csv_reader)
        data = next(csv_reader)
        param_dict = dict(zip(headers, data))

    numberofiteration = int(param_dict['numberofiteration'])
    end_time = float(param_dict['end_time'])
    hss = int(float(param_dict['hstart']))
    hsd = int(float(param_dict['hduration']))

    if 'hstart2' in param_dict: opt.hs2 = int(param_dict['hstart2'])
    else: opt.hs2 = False

    param_dict['model_name'] = model_name
    date = opt.ids[:10]

    return param_dict, numberofiteration, end_time, hss, hsd, opt, date, model_name




def load_Param_fromDataFile(param_rootdir, data_dir, partiii_dir, varPara_dir, varData_dir, opt):
    if bool(re.search(re.compile(r'/'), opt.ids)) == False:
        ## import from SA data
        param_dir, data_filename = os.path.split(f"{param_rootdir}/{opt.ids}.csv")
        plot_dir = param_dir
        S_val = float(re.match(r'^([\d.]+)_', data_filename).group(1))
        print(param_dir)
        try: 
            S, cost_func, param_dict = loadData(f"{param_rootdir}/{S_val}.pcl")
            para_csv_name = f"{param_dir}/{S_val}.pcl"
            opt.S = S
            opt.cost_func = cost_func
            model_name = extract_model_name(para_csv_name)
        except FileNotFoundError:
            if os.path.exists(f"{param_dir}/{S_val}.csv"):
                para_csv_name = f"{param_dir}/{S_val}.csv"
                model_name = extract_model_name(para_csv_name)
                param_dict, opt = param_dict_import_impute(para_csv_name, model_name, opt)
    elif bool(re.search(re.compile(r'Para'), opt.ids)) == True: ## imported param from simuData
        plot_dir = data_dir
        if os.path.exists(f"{data_dir}/Exp3_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/Exp3_Para_{opt.ids}.csv"
            model_name = model_from_date(para_csv_name)
            param_dict, opt = param_dict_import_impute(para_csv_name, model_name, opt)
        elif os.path.exists(f"{data_dir}/replaceA1_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/replaceA1_Para_{opt.ids}.csv"
            model_name = "replaceA1"
            param_dict, opt = param_dict_import_impute(para_csv_name, model_name, opt)
        elif os.path.exists(f"{data_dir}/woA2_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/woA2_Para_{opt.ids}.csv"
            model_name = "woA2"
            param_dict, opt = param_dict_import_impute(para_csv_name, model_name, opt)
        elif os.path.exists(f"{data_dir}/d1upCons_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/d1upCons_Para_{opt.ids}.csv"
            model_name = "d1upCons"
            param_dict, opt = param_dict_import_impute(para_csv_name, model_name, opt)
    else: ## from variance analysis
        print('var')
        plot_dir = varData_dir
        path, filename = os.path.split(f"{partiii_dir}/{opt.ids}.csv") # this is simuData name
        S_val = re.search(r'^(\d+\.\d+)_', filename).group(1)
        para_csv_name = f"{varPara_dir}/{S_val}.csv"
        model_name = "replaceA1"
        param_dict, opt = param_dict_import_impute(para_csv_name, model_name, opt)
    if 'hstart2' in param_dict: opt.hs2 = int(param_dict['hstart2'])
    else: opt.hs2 = False
    return param_dict, plot_dir, model_name, opt

def param_dict_import_impute(para_csv_name, model_name, opt):
    param_dict = {}
    with open(para_csv_name, 'r') as param_file:
        csv_reader = csv.reader(param_file)
        headers = next(csv_reader)
        data = next(csv_reader)
    for key, val in zip(headers, data):
        if key == 'model_name': 
            param_dict[key] = str(val)
        else: param_dict[key] = float(val)
    if not 'hstart2' in param_dict: param_dict['hstart2'] = 0
    if not 'model_name' in param_dict: 
        param_dict['model_name'] = model_name
    opt.para_csv_name = para_csv_name
    return param_dict, opt



def model_from_date(file):
    ctime = datetime.fromtimestamp(os.path.getmtime(file))
    woA2_start = datetime(2023,12,6,23,30)
    replaceA1_start = datetime(2024,2,16,15,8)
    others_start = datetime(2024,2,26)
    only_d1upCons = datetime(2024,2,19,10,25)
    if ctime >= woA2_start and ctime < replaceA1_start: model_name = "woA2"
    elif ctime > replaceA1_start and ctime <= others_start: 
        if ctime == only_d1upCons: model_name = 'd1upCons'
        else: model_name = "replaceA1"
    else: print('model_from_date() receives unexpected param_file creation date')
    return model_name


def extract_model_name(filename):
    ctime = datetime.fromtimestamp(os.path.getmtime(filename))
    if ctime < datetime(2024,2,28): model_name = "replaceA1"
    else:
        pattern = re.compile(r'\(\w+\)_(\w+)_')
        match = pattern.search(filename)
        model_name = match.group(1)
    return model_name



def info_from_param_dict(param_dict, opt):
    #numberofiteration = int(param_dict['numberofiteration'])
    end_time = float(param_dict['end_time'])
    hss = int(float(param_dict['hstart']))
    hsd = int(float(param_dict['hduration']))

    if 'hstart2' in param_dict: opt.hs2 = int(param_dict['hstart2'])
    else: opt.hs2 = False

    model_name = param_dict['model_name']
    date = opt.ids[:10]

    return end_time, hss, hsd, opt, date, model_name


#######################################################################
## 3. Import Simulation Data
#######################################################################
def import_tidy_simuData(data_dir, partiii_dir, param_rootdir, opt, model_name):
    if bool(re.search(r'simuData_for_varAna', opt.ids)) == True:## from variance analysis
        data_df = pd.read_csv(f"{partiii_dir}/{opt.ids}.csv")
        path, filename = os.path.split(f"{partiii_dir}/{opt.ids}.csv")
        opt.name_suffix = filename[:-4]
    elif bool(re.search(re.compile(r'/'), opt.ids)) == True: #input from SA param
        data_df = pd.read_csv(f"{param_rootdir}/{opt.ids}.csv")
        path, filename = os.path.split(f"{param_rootdir}/{opt.ids}.csv")
        opt.name_suffix = filename[:-4]
    else: #input from simuData param
        try: data_df = pd.read_csv(f"{data_dir}/{model_name}_SimuData_{opt.ids}")
        except FileNotFoundError: data_df = pd.read_csv(f"{data_dir}/Exp3_SimuData_{opt.ids}")
        opt.name_suffix = opt.ids
    
    numberofiteration = int(re.search(r'_numIter(\d+)_', opt.name_suffix).group(1))
    hss = int(re.search(r'_HSstart(\d+)_', opt.name_suffix).group(1))
    hsd = int(re.search(r'_HSduration(\d+)', opt.name_suffix).group(1))
    end_time = re.search(r'_Time(\d+\.\d+)_', opt.name_suffix).group(1)


    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    data_df['totalHSFA1'] = data_df['C_HSFA1_HSPR'] + data_df['HSFA1']
    #print(data_df)
    #print(data_df.shape)
    ### number of rows and columns for all iterations
    grouped_data = data_df.groupby('Iteration_Identifier')
    return data_df, grouped_data, numberofiteration, hss, hsd, end_time, opt


#######################################################################
## 4. Generate Plot Name Suffix
#######################################################################


def genPlotName_nondefault(param_dict, numberofiteration, end_time, hss, hsd, opt):
    default_param_dict = {
        ## initial concentrations
        'init_HSFA1': 1,
        'init_HSPR': 2,
        'init_C_HSFA1_HSPR': 50,
        'init_MMP': 0,
        'init_FMP': 50,
        'init_C_HSPR_MMP': 50,
        'init_HSFB': 1,
        'Time': 0.0,
        ## Maximum expression level in Hill equation
        'a1': 10.0,
        'a2': 100.0,
        'a5': 5.0,
        'a6': 2.0, # refolding rate from MMP-HSPR
        'a7': 10,
        #'a8': 5.0,
        ## Ka in Hill equation
        'h1': 1.0,
        'h2': 1.0,
        #'h3': 1.0,
        #'h4': 1.0,
        'h5': 1.0,
        #'h6': 1.0,
        ## association rates
        'c1': 10.0,
        'c2': 5.0,
        'c3': 0.5, #between MMP and HSPR
        'c4': 10.0,
        ## decay rates
        'd1': 0.1, # decay path 1 of A1-HSPR
        'd3': 0.01, # dissociation rate of MMP-HSPR
        'd4_heat': 0.05,
        'd4_norm': 0.01,
        'Decay1': 0.04,
        'Decay2': 0.04, # decay of free HSPR
        'Decay4': 0.04,
        'Decay6': 0.04,
        'Decay7': 0.04, # decay path 2 of A1-HSPR
        'Decay8': 0.04, # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage_A1': 0.001,
        'leakage_B': 0.001,
        'leakage_HSPR': 0.001,
        'hillcoeff': 2,
        'numberofiteration': numberofiteration,
        'hstart':hss,
        'hstart2':0,
        'model_name':'undefined',
        'hduration':hsd,
        'end_time':end_time
    }
    diff_dict = {}
    for (dk,dv) in default_param_dict.items():
        if not 'hstart2' in param_dict: param_dict['hstart2'] = 0
        #if not 'model_name' in param_dict:  --> this line, not needed, cuz model_name always specified
        if str(param_dict[dk]) != str(dv):
            #print(f"default: {dk}, {dv}")
            #print(f"actual: {dk}, {param_dict[dk]}")
            diff_dict[dk] = param_dict[dk]
    #name_suffix_save = f"{date}_numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}"
    #name_suffix = f"{date}_numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}\n"
    name_suffix = f"numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}\n"
    for key, val in diff_dict.items():
        name_suffix += f"_{key}-{val}"
        #name_suffix_save += f"_{key}-{val}"
    return name_suffix, diff_dict

def param_dict_toText(param_dict, opt):
    param_dict_text = ''
    for i, (key, val) in enumerate(param_dict.items(),1):
        if type(val) == float:
            to_add = f"  {key}-{round(val,3)}"
        else:
            to_add = f"  {key}-{val}"
        param_dict_text += to_add
        if i % 8 ==0:
            param_dict_text += f'\n'
    return param_dict_text


#######################################################################
## 5. Plotting Trajectories
#######################################################################

def plot_trajectory(opt, ax, data_df, x_col, y_col_list, hss, hsd, Iteration_Identifier):
    for y_col in y_col_list:
        ax.plot(data_df[x_col], data_df[y_col], label=y_col, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Protein copy number')
    ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.2)
    if bool(opt.hs2) == True: ax.axvspan(opt.hs2, opt.hs2+hsd, facecolor='r', alpha=0.2)
    ax.set_title(f"{Iteration_Identifier}")
    ax.legend(loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def plot_trajectory(ax, fig, data_df, x_col, y_col_list, hss, hsd, plot_title_to_show, param_dict_text, Iteration_Identifier):
    for y_col in y_col_list:
        ax.plot(data_df[x_col], data_df[y_col], label=y_col, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Protein copy number')
    ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.2)
    ax.set_title(f"{Iteration_Identifier}")
    ax.legend(loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

    plt.subplots_adjust(right=0.8) 
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    fig.text(0.5, 0.95, f'{plot_title_to_show}\n{param_dict_text}', ha = 'center', va='center', fontsize = int(plt.rcParams['font.size']), linespacing=2,  wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)


def saveFig(data_dir, name_suffix, opt, prefix):
    if bool(opt.sfg) == True:
        plot_name = f"{data_dir}/{name_suffix}_{prefix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        #plot_name = f"{data_dir}/{prefix}_{name_suffix}.svg"
        #unique_plot_name = get_unique_filename(plot_name)
        #plt.savefig(f"{unique_plot_name}")
        print(f" save figure {opt.sfg == True}")

def plotReactionRate(data_df, grouped_data, data_dir, numberofiteration, name_suffix, opt, hss, hsd):
    rr = ['R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR']
    #rr = ['R_HSFA1_inc','R_HSPR_inc', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_MMP_inc','R_MMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2','R_HSFB_inc']

    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(opt, ax, data_df, 'time', rr, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):
            plot_trajectory(opt, ax, group_data, 'time', rr, hss, hsd, Iteration_Identifier = Iteration_Identifier)

    plt.tight_layout()
    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='ReactionRate')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2(data_df, grouped_data, data_dir, numberofiteration,name_suffix, hss, hsd, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB','totalHSFA1']
    protein = ['FMP','MMP']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(opt, ax[0], data_df, 'time', protein, hss, hsd, "iteration 0")
        plot_trajectory(opt, ax[1], data_df, 'time', reg, hss, hsd, "iteration 0")

    else:
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(opt, ax[i,0], group_data, 'time', protein, hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(opt, ax[i,1], group_data, 'time', reg, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='ProReg2')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2_overlayed(data_df, grouped_data, data_dir, numberofiteration,name_suffix, hss, hsd, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB','totalA1']
    protein = ['FMP','MMP']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(opt, ax[0], data_df, 'time', protein, hss, hsd, "iteration 0")
        plot_trajectory(opt, ax[1], data_df, 'time', reg, hss, hsd, "iteration 0")

    else:
        fig, ax = plt.subplots(ncols = 2, figsize=(20,10))
        plot_trajectory(opt, ax[0], data_df, 'time', protein, hss, hsd, Iteration_Identifier = "all iter")
        plot_trajectory(opt, ax[1], data_df, 'time', reg, hss, hsd, Iteration_Identifier = "all iter")
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='ProReg2overlay')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMP_zoom(data_df, hss, hsd, data_dir, numberofiteration,name_suffix, opt):
    print(" Zoomed In Protein & Regulator Trajectory Around HeatShock")
    cut_data_df = data_df[(data_df['time'] >= hss -50) & (data_df['time'] <= hss + hsd + 100)]
    reg_conc_col = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB','totalHSFA1']
    
    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=3, figsize=(20, 5))
        plot_trajectory(opt, ax[0], cut_data_df, 'time', ['FMP','MMP'], hss, hsd, "iteration 0")
        plot_trajectory(opt, ax[1], cut_data_df, 'time', reg_conc_col, hss, hsd, "iteration 0")
    else:
        grouped_data = cut_data_df.groupby('Iteration_Identifier')
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(opt, ax[i,0], group_data, 'time', ['FMP','MMP'], hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(opt, ax[i,1], group_data, 'time', reg_conc_col, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    #fig.suptitle('Zoomed In Trajectories, Around HeatShock')
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='ProRegZoom')
    if bool(opt.shf) == True: plt.show()
    plt.close()



########################################
#########################################
    

def plot_allvsTime_separate(data_df, grouped_data, data_dir, numberofiteration, name_suffix, opt, hss, hsd, diff_dict):

    print(" Plot trajectories of all species for all iterations")
    conc_col = ['HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB']

    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(opt, ax, data_df, 'time', conc_col, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(opt, ax, group_data, 'time', conc_col, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    #fig.suptitle('Plot of all concentrations vs time for all iterations separately', fontsize=16, y = 1)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='allConcTraj')
    if bool(opt.shf) == True: plt.show()
    plt.close()

def plot_allvsZoomInTime_separate(data_df, hss, hsd, data_dir, numberofiteration,name_suffix, opt):

    print(" Zoomed In Trajectory Around HeatShock")
    cut_data_df = data_df[(data_df['time'] >= hss -50) & (data_df['time'] <= hss + hsd + 100)]
    conc_col = ['HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB']
    
    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(opt, ax, cut_data_df, 'time', conc_col, hss, hsd, "iteration 0")
    else:
        grouped_data = cut_data_df.groupby('Iteration_Identifier')
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(opt, ax, group_data, 'time', conc_col, hss, hsd, Iteration_Identifier=Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Zoomed In Trajectories, Around HeatShock')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix = "Zoom_allConcTraj")

    if bool(opt.shf) == True: plt.show()
    plt.close()

def plot_FMPMMPvsTime_3(data_df, grouped_data, data_dir, numberofiteration,name_suffix, hss, hsd, opt):

    print(" Plot trajectories of Proteins and Regulators")
    HSPR_complex = ['C_HSPR_MMP','totalHSPR']
    reg = ['HSFA1','HSFB','C_HSFA1_HSPR','HSPR']
    protein = ['FMP','MMP']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=3, figsize=(27, 5))
        plot_trajectory(opt, ax[0], data_df, 'time', protein, hss, hsd, "iteration 0")
        plot_trajectory(opt, ax[1], data_df, 'time', HSPR_complex, hss, hsd, "iteration 0")
        plot_trajectory(opt, ax[2], data_df, 'time', reg, hss, hsd, "iteration 0")

    else:
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 3, figsize=(27,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(opt, ax[i,0], group_data, 'time', protein, hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(opt, ax[i,1], group_data, 'time', HSPR_complex, hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(opt, ax[i,2], group_data, 'time', reg, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='ProReg3')

    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2_overlayed_2(data_df, grouped_data, data_dir, numberofiteration,name_suffix, name_suffix_save, hss, hsd, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB', 'totalA1']
    protein = ['FMP','MMP']

    fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
    for (i, group_data) in grouped_data:
        plot_trajectory(opt, ax[0], group_data, 'time', protein, hss, hsd, "Merge all iterations")
        plot_trajectory(opt, ax[1], group_data, 'time', reg, hss, hsd, "Merge all iterations")

    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(data_dir, str(opt.name_suffix), opt, prefix ='ProReg2overlay')
    if bool(opt.shf) == True: plt.show()
    plt.close()





def plot_A1BvsTime_separate(data_df, grouped_data, data_dir, numberofiteration,name_suffix, opt):

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
        plot_name = f"{data_dir}/A1-BConcTraj_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{data_dir}/A1-BConcTraj_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f" save figure {opt.sfg == True}")

    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_totalHSPRvsTime_subplots(grouped_data, data_df, data_dir, numberofiteration, name_suffix, hss, hsd, opt):
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
        plot_name = f"{data_dir}/totalHSPRtrajec_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{data_dir}/totalHSPRtrajec_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

    if bool(opt.shf) == True: plt.show()
    plt.close()



def plot_A1vsTime_asOne(grouped_data, data_dir, numberofiteration, name_suffix, opt):
    fig, ax1 = plt.subplots(figsize=(15,10))  # Set the figure size 
    for Iteration_Identifier, group_data in grouped_data:
        ax1.plot(group_data['time'], group_data['HSFA1'], label='{}'.format(Iteration_Identifier))
        ax1.set_xlabel('time')
        ax1.legend()
        ax1.set_ylabel('HSFA1')
        ax1.set_title('Plot of HSFA1 vs time for all Iterations')
    plt.tight_layout()
    if bool(opt.sfg) == True:
        plot_name = f"{data_dir}/A1TrajMerged_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{data_dir}/A1TrajMerged_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
    if bool(opt.shf) == True: plt.show()
    plt.close()





#######################################################################
## 5. Variability Analysis
#######################################################################

def df_Processing_HS(data_df, hss, hsd, end_time, opt):
    ss1_start = 200
    ss1_end = int(hss)
    ssHS_start = int(hss)
    ssHS_end = int(hss) + int(hsd)
    ss3_start = ssHS_end
    ss3_end = end_time
    print(f"hss:{hss}, hsd: {hsd}")
    print(f"ss1: {ss1_start} - {ss1_end} \nssHS: {ssHS_start} - {ssHS_end} \nss3:{ss3_start} - {ss3_end} ")

    ss1_df = data_df[(data_df['time'] >= ss1_start) & (data_df['time'] <= ss1_end)]
    ssHS_df = data_df[(data_df['time'] >= ssHS_start) & (data_df['time'] <= ssHS_end)]
    ss3_df = data_df[data_df['time'] >= ss3_start]

    print("ss1_df", ss1_df.shape)
    print("ssHS_df", ssHS_df.shape)
    print("ss3_df", ss3_df.shape)

    df_list = [ss1_df, ssHS_df, ss3_df]

    #totalHSPR_df_outlist = []
    #for grouped_df in df_list:
    #    result_df = grouped_df.agg(['mean','std'])
    #    result_df['cv'] = result_df['std'] / result_df['mean']
    #    result_df.columns = ['mean_totalHSPR', 'std_totalHSPR', 'cv_totalHSPR']
    #    result_df.reset_index(inplace=True)
    #    totalHSPR_df_outlist.append(result_df)
    return df_list




def bootHSPR_toList(df_list):
    HSPR_list = []
    for df in df_list:
        print(len(df['totalHSPR'].tolist()))
        HSPR_conc = random.choices(df['totalHSPR'].tolist(), k=5)
        HSPR_list.append(HSPR_conc)
    return HSPR_list



def bootstrap_HSPR_hist_overlap(df_list, plot_dir, maxVal, opt):
    HSPR_list = bootHSPR_toList(df_list)

    print("Plot total HSPR histogram")
    fig = plt.figure(figsize=(12, 6))
    label_list = ["before HS", "during HS", "after HS"]
    plt.hist(HSPR_list, bins = range(0, maxVal, 1), label = label_list, density=True, alpha=0.50, histtype='stepfilled')
    plt.title("Distribution of HSPR conc, bootstrapped")
    plt.xlabel("HSPR")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Adjust the figure size to accommodate the legend
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    plt.tight_layout()

    saveFig(plot_dir, str(opt.name_suffix), opt, prefix ='bootHistOver')
    if bool(opt.shf) == True: plt.show()
    plt.close()




def bootstrap_HSPR_hist_subplot(df_list, plot_dir, maxVal, opt):
    HSPR_list = bootHSPR_toList(df_list)
    print("Plot total HSPR histogram")
    label_list = ["before HS", "during HS", "after HS"]
    color_list = ['green','red','blue']
    fig, axes = plt.subplots(nrows=3, ncols=1, sharex=True,sharey=True, figsize=(15,12))
    for i, (HSPR_conc, label, ax, color) in enumerate(zip(HSPR_list, label_list, axes, color_list)):
        ax.hist(HSPR_conc, bins = range(0, maxVal, 1), label = label, density=True, alpha=0.50, histtype='stepfilled', color = color)
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

    saveFig(plot_dir, str(opt.name_suffix), opt, prefix ='bootHistSep')
    if bool(opt.shf) == True: plt.show()
    plt.close()

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


def plot_CVsq_mean(totalHSPR_df_outlist, data_dir, name_suffix, opt):
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
        plot_name = f"{data_dir}/CV-Mean_TotalHSPR_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{data_dir}/CV-Mean_TotalHSPR_{name_suffix}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f"save figure {opt.sfg == True}")

    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_HSPR_hist(totalHSPR_df_outlist, data_dir, name_suffix, opt):

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
        plot_name = f"{data_dir}/Hist_TotalHSPR_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{data_dir}/Hist_TotalHSPR_{name_suffix}.svg"
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

