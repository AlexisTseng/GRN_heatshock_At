script_usage="""
usage
    HSPR_AZ_SA.py -ops <optSteps> -nit <numberInteration> [options]

version
    HSPR_AZ_SA.py 0.0.2 (alpha)

dependencies
    Python v3.9.7, Scipy v1.11.2, NumPy v1.22.2, viennarna v2.5.1, Matplotlib v3.5.1, pandas v2.1.0

description
    Version 1 for simulated annealing only

    Version 2 for plotting and simulated annealing

################################################################

--optSteps,-ops
    how many steps the simulated annealing algorithm is gonna take (default:10)

--costFunc,-cof
    which cost function to use. e.g. 'fctHSPR', 'fcA1tHSPR-pHSA1HSPR', 'maxFMP' (default: fcA1tHSPR-pHSA1HSPR)

#################################################################

--numberInteration,-nit
    The number of interations for Gillespi simulation (default: 2)

--modelName,-mdn
    which model version or Gillespie Function to use (default: replaceA1)

--timeStep,-tsp
    The time duration for each Gillespi simulation run/iteration (default:1000)

--heatShockStart,-hss
    The time point at which a heat shock is introduced (default:600)

--heatShockDuration,-hsd
    The duration of heat shock introduced (default: 3)

--heatShockStart2,-hs2
    The time point at which a second heat shock is introduced (default: 0)

--withA2,-wa2
    whether HSFA2 is present in the model (default: 0)

--A2positiveAutoReg,-a2p
    Whether HSFA2 positively regulates itself in the model (default: 0)

####################################################################

--misfoldRateNormal,-mfn
    The formation rate of misfolded protein from folded protein under normal temperature (default: 0.01)

--misfoldRateHS,-mfh
    The formation rate of misfolded protein from folded protein under heat shock condition (default: 0.05)

--assoA1_HSPR,-aah
    c1, association rate between A1 and HSPR (default:10.0)

--repMMP_A1H,-rma
    c2, rate at which MMP replaces A1 from A1_HSPR complex, to form MMP_HSPR instead (default: 0.1)

--assoMMP_HSPR,-amh
    c3, The association rate between MMP and HSPR (default: 0.5)

--repA1_MMPH,-ram
    c4, rate at which A1 replaces MMP from MMP_HSPR complex, to form A1_HSPR instead (default: 0.1)

--disoA1_HSPR,-dah
    d1, dissociation rate of A1-HSPR (default: 0.1)

--HSdisoA1_HSPR,-hda
    d1_HS, dissociation rate of A1-HSPR (default: 0.1)

--disoMMP_HSPR,-dmh
    d3, dissociation rate of A1-HSPR (default: 0.01)

--hillCoeff,-hco
    The Hill Coefficient (default: 2)

##################################################################################

--leakage_A1,-lga
    Trancription leakage for HSFA1 (default: 0.001)

--leakage_B,-lgb
    Trancription leakage for HSFB (default: 0.001)

--leakage_HSPR,-lgh
    Trancription leakage for HSPR (default: 0.001)

--leakage_A2,-la2
    Trancription leakage for HSFA2 (default: 0.001)

################################################################################

--hilHalfSaturation,-hhs
    The conc of inducer/repressor at half-max transcriptional rate (default: 1.0)

--KA1actA1,-hAA
    h1, K constant for A1 (default: 1.0)

--KA1actHSPR,-hAH
    h2, K constant for HSPR (default: 1.0)

--KA2actHSPR,-hA2
    h4, K constant for HSFA2 (default: 1.0)

--KA1actB,-hAB
    h5, K constant for HSFB (default: 1.0)

##################################################################################

--initFMP,-ifp
    Initial FMP abundance (default: 5000)

--initMMP,-imp
    Initial MMP abundance (default: 0)

--initA1_HSPR,-iah
    initial C_HSFA1_HSPR abundance (default: 50)

--initA1free,-iaf
    initial free A1 (default: 1)

--initB,-ibf
    initial HSFB (default: 1)

--initHSPRfree,-ihf
    initial free HSPR (default: 2)

--initMMP_HSPR,-imh
    initial C_MMP_HSPR abundance (default: 50)

##################################################################################


######################################################

--maxRateB,-a5B
    a5, Max transcription rate of HSFB (default: 10)

--maxRateA1,-a1A
    a1, Max transcription rate of HSFA1 (default: 10)

--maxRateHSPR,-a2H
    a2, Max transcription rate of HSFB (default: 100)

--maxRateA2,-a4A
    a4, max transcription rate of HSFA2 (default: 10)

--foldedProduction,-fpp
    a7, folded protein production rate (default: 300)

-refoldRate,-a6R
    a6, refolding rate from MMP-HSPR (default: 2.0)

--globalDecayRate,-gdr
    the decay rate of species except for MMP (default:0.04)

#############################################################

--A1mut,-a1m
    if 1, no HSFA1 (default: 0)

--Bmut,-bmu
    if 1, no HSFB (default: 0)

--HSPRmut,-hmu
    if 1, no HSPR (default: 0)

######################################################################

--outputFormat,-ofm
    Whether to save Gillespi simulation output as csv or picklefile (default: csv)

--samplingFreq,-spf
    How often is simulation datasaved (default: 0.1)

--thread,-thr
    The number of threads used for multiprocessing (default: 4)

##############################################################

--saveFig,-sfg
    Whether to save the figures and plots generated. Default = True (default: 1)

--showFig,-shf
    Whether to show  the figures generated (default: 0)

--varAnalysis,-van
    whether to analysis variability by plotting histograms etc. (default: 0)

--plotResult,-prs
    Set to 0 if running on HPC (default: 1)

--saveOutput,-sop
    whether to save param explore results (default: 1)

################################################################################

reference
    A.J.Zeng
    xxxxxxxxx
"""


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
    param_dict = param_spec(opt)
    #param_dict = updatePara_unif(param_dict, opt)

    #neutral_walk(param_dict, opt)
    simuAnneal(param_dict, opt)


def neutral_walk(param_dict, opt):
    if bool(opt.sop) == True:
        param_dir, param_rootdir, plot_dir, data_dir = dir_gen(opt)
    S_record, param_record = [], []
    for i in range(opt.ops):
        data_df, grouped_data, numberofiteration, listM6, end_time, hss, hsd = one_ops_step(param_dict, opt)
        
        ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df = df_Processing_HS(data_df, hss, hsd, end_time, opt)

        S, cost_func, S_record, param_record = update_S_param(data_df, ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, i, S_record, param_record, opt)

        if bool(opt.sop) == True: 
            data_file, date, data_name, plot_title_to_show = saveData_oneSAstep(listM6, param_dir, numberofiteration, end_time, S, opt)
            saveParam_csv_pcl(param_dict, param_dir, S, cost_func, opt)
            if bool(opt.prs) == True:
                plot_results(param_dir, data_name, data_df, grouped_data, param_dict, S, numberofiteration, hss, hsd, plot_title_to_show, opt)

        param_dict = updatePara_unif(param_dict, opt)

    if bool(opt.sop) == True: 
        save_S_param(S_record, param_record, param_rootdir, param_dir, cost_func, opt)

def simuAnneal(param_dict, opt):
    SA_rootdir, SA_dir = dir_genSA(opt)
    S_record, param_record = [], []
    T = 50
    for i in range(opt.ops):
        
        T = 0.95*T
        data_df, grouped_data, numberofiteration, listM6, end_time, hss, hsd = one_ops_step(param_dict, opt)
        ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df = df_Processing_HS(data_df, hss, hsd, end_time, opt)
        S, delta_S, cost_func = obj_func(data_df, ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, i, S_record, opt)

        if math.isnan(S) or math.isinf(S): 
            print(' reject')

        elif accept_step(T, delta_S) == True or i ==0 :
            S_record, param_record = add_step(i, S, delta_S, param_dict, param_record, S_record, cost_func, listM6, SA_dir, numberofiteration, end_time, hss, hsd, data_df, grouped_data, opt)

        param_dict = updatePara_unif(param_dict, opt)

    if bool(opt.sop) == True: 
        save_S_param(S_record, param_record, SA_rootdir, SA_dir, cost_func, opt)


######################################################
## Main Functions
######################################################

def dir_gen(opt):
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)
    date = datetime.now().date()

    data_dir = os.path.join(partiii_dir,"Ritu_simulation_data")
    if not os.path.isdir(data_dir): os.makedirs(data_dir, 0o777)

    plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)

    param_dir = os.path.join(partiii_dir,"Param_Optimisation", f"{date}_{opt.mdn}_step{opt.ops}_time{opt.tsp}_hss{opt.hss}_hsd{opt.hsd}_withA2-{bool(opt.wa2)}_cosFunc-{opt.cof}")
    param_dir = get_unique_filename(param_dir)
    if not os.path.isdir(param_dir): os.makedirs(param_dir, 0o777)

    param_rootdir = os.path.join(partiii_dir,"Param_Optimisation")
    if not os.path.isdir(param_rootdir): os.makedirs(plot_dir, 0o777)

    return param_dir, param_rootdir, plot_dir, data_dir

def dir_genSA(opt):
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)
    date = datetime.now().date()

    SA_rootdir = os.path.join(partiii_dir,"SimuAnneal")
    if not os.path.isdir(SA_rootdir): os.makedirs(SA_rootdir, 0o777)

    SA_dir = os.path.join(SA_rootdir, f"{date}_{opt.mdn}_withA2-{bool(opt.wa2)}_step{opt.ops}_time{opt.tsp}_hss{opt.hss}_hsd{opt.hsd}_cosFunc-{opt.cof}")
    SA_dir = get_unique_filename(SA_dir)
    if not os.path.isdir(SA_dir): os.makedirs(SA_dir, 0o777)
    return SA_rootdir, SA_dir

def param_spec(opt):
    if bool(opt.a1m) == False:
        init_HSFA1, a1, leakage_A1 = int(opt.iaf), int(opt.a1A), float(opt.lga)
    elif bool(opt.a1m) == True: 
        print("A1 mutant")
        init_HSFA1, a1, leakage_A1 = 0, 0,0

    if bool(opt.bmu) == False: 
        init_HSFB, a5, leakage_B = int(opt.ibf), int(opt.a5B), float(opt.lgb)
    elif bool(opt.bmu) == True: 
        print("HSFB mutant")
        init_HSFB, a5, leakage_B = 0,0, 0

    if bool(opt.hmu) == False:
        init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR = int(opt.imh), int(opt.ihf), int(opt.a2H), float(opt.lgh)
    elif bool(opt.hmu) == True: 
        print("HSPR mutant")
        init_C_HSPR_MMP, init_HSPR, a2, leakage_HSPR = 0,0,0, 0

    if bool(opt.hmu) == True or bool(opt.a1m) == True:
        init_C_HSFA1_HSPR_val = 0
    else: init_C_HSFA1_HSPR_val = int(opt.iah)

    param_dict = {
        ## initial concentrations
        'init_HSFA1': init_HSFA1,
        'init_C_HSFA1_HSPR': init_C_HSFA1_HSPR_val,
        'init_HSPR': init_HSPR,
        'init_MMP': int(opt.imp),
        'init_FMP': int(opt.ifp),
        'init_C_HSPR_MMP': init_C_HSPR_MMP,
        'init_HSFB': init_HSFB,
        'Time': 0.0,
        ## Maximum expression level in Hill equation
        'a1': a1,
        'a2': a2,
        'a5': a5,
        'a6': float(opt.a6R), # refolding rate from MMP-HSPR
        'a7': int(opt.fpp), #folded protein production rate
        #'a8': 50.0,
        ## Ka in Hill equation
        'h1': float(opt.hAA),
        'h2': float(opt.hAH),
        #'h3': int(opt.hBA),
        #'h4': int(opt.hBH),
        'h5': float(opt.hAB),
        #'h6': int(opt.hBB),
        ## association rates
        'c1': float(opt.aah), # binding rate between A1 and HSPR
        'c3': float(opt.amh), # binding rate between MMP and HSPR
        ## decay rates
        'd1': float(opt.dah), # decay path 1 of A1-HSPR
        'd3': float(opt.dmh), # dissociation rate of MMP-HSPR
        'd4_heat': float(opt.mfh),
        'd4_norm': float(opt.mfn),
        'Decay1': float(opt.gdr),
        'Decay2': float(opt.gdr), # decay of free HSPR
        'Decay4': float(opt.gdr),
        'Decay6': float(opt.gdr),
        'Decay7': float(opt.gdr), # decay path 2 of A1-HSPR
        'Decay8': float(opt.gdr), # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage_A1': leakage_A1,
        'leakage_B': leakage_B,
        'leakage_HSPR': leakage_HSPR,
        'numberofiteration': int(opt.nit),
        'hillcoeff': int(opt.hco),
        'hstart':int(opt.hss),
        'hstart2':int(opt.hs2),
        'hduration':int(opt.hsd),
        'model_name': str(opt.mdn)
    }
    if bool(opt.wa2) == True:
        param_dict['leakage_A2'] = float(opt.la2)
        param_dict['a4'] = int(opt.a4A)
        param_dict['h4'] = float(opt.hA2)
        param_dict['Decay3'] = float(opt.gdr)
        param_dict["init_HSFA2"] = 1
    if opt.mdn == 'replaceA1':
        param_dict['c2'] = float(opt.rma) # MMP replace A1 in complex with HSPR
        param_dict['c4'] = float(opt.ram) # A1 replace MMP in complex with HSPR
        param_dict['d1_HS'] = ''
    elif opt.mdn == 'd1upCons':
        param_dict['d1_HS'] = float(opt.hda)
        param_dict['c2'] = ''
        param_dict['c4'] = ''
    
    #print(param_dict)
    return param_dict


def one_ops_step(param_dict, opt):
    numberofiteration = param_dict['numberofiteration']
    hss = param_dict['hstart']
    hsd = param_dict['hduration']
    listM4, listtime2, numberofiteration, end_time, rr_list2, model_name = gillespie(param_dict, opt)
    listM6 = combine_data(listtime2, listM4, rr_list2, opt)
    Stoich, Stoich_df, header = get_stoich(opt)
    data_df, grouped_data = Gillespie_list_to_df(header, listM6, opt)
    return data_df, grouped_data, numberofiteration, listM6, end_time, hss, hsd


def update_S_param(data_df, ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, i, S_record, param_record, opt):
    ## Objective Function Calculation
    if i == 0: S_old = 0
    else: S_old = S_record[-1]
    if opt.cof == 'fcA1tHSPR-pHSA1HSPR':
        S, cost_func = obj_func_fcA1tHSPRpHSA1HSPR(ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, opt)
    elif opt.cof == 'fctHSPR':
        S, cost_func = obj_func_fctHSPR(ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, opt)
    elif opt.cof == 'maxFMP':
        S, cost_func = obj_func_maxFMP(data_df, opt)
    delta_S = S - S_old
    S_record = S_record + [S]
    param_dict_toSave = param_dict.copy()
    param_record.append(param_dict_toSave)
    print(f"\n------> STEP {i}, S:{S}, delta_S: {delta_S}\n param_dict: {param_dict}\n")
    return S, cost_func, S_record, param_record




def saveParam_csv_pcl(param_dict, param_dir, S, cost_func, opt):
    param_name = f"{param_dir}/{S}.csv"
    header = param_dict.keys()
    with open(param_name, 'w', newline='') as csvfile_2:
        # Create a CSV writer object
        writer = csv.DictWriter(csvfile_2, fieldnames=header)
        # Write the header
        writer.writeheader()
        # Write the parameter values
        writer.writerow(param_dict)
    saveData((S, cost_func, param_dict), f"{param_dir}/{S}.pcl")


def save_S_param(S_record, param_record, param_rootdir, param_dir, cost_func, opt):
    date = datetime.now().date()
    param_opt_log = f"{param_dir}/ParamRecord_{date}_{opt.mdn}_step{opt.ops}_time{opt.tsp}_hss{opt.hss}_hsd{opt.hsd}_withA2-{bool(opt.wa2)}.csv"

    with open(param_opt_log, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        header_row = ['cost_function']+ ['S'] + list(param_record[0].keys())
        csv_writer.writerow(header_row)
        for (dict, S) in zip(param_record, S_record):
            row = [cost_func] + [S] +  list(dict.values())
            csv_writer.writerow(row)


def save_S_param_toLog(S_record, param_record, param_rootdir, param_dir, cost_func, opt):
    date = datetime.now().date()
    del param_record[0]['model_name']
    header_row = ['param_dir'] + ['model_name'] + ['cost_function'] + ['S'] + list(param_record[0].keys())
    if bool(opt.wa2) == False:
        param_opt_log = f"{param_rootdir}/Param_optimisation_log.csv"
    else: 
        param_opt_log = f"{param_rootdir}/Param_optimisation_log_withA2.csv"
    if os.path.exists(param_opt_log):
        with open(param_opt_log, mode='a', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            for (dict, S) in zip(param_record, S_record):
                try: del dict['model_name']
                except KeyError: pass
                row = [param_dir] + [opt.mdn] + [cost_func] + [S] +  list(dict.values())
                csv_writer.writerow(row)
    else:
        with open(param_opt_log, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(header_row)
            for (dict, S) in zip(param_record, S_record):
                del dict['model_name']
                row = [f"{date}_{opt.mdn}_step{opt.ops}_time{opt.tsp}_hss{opt.hss}_hsd{opt.hsd}"] + [opt.mdn] + [cost_func] + [S] +  list(dict.values())
                csv_writer.writerow(row)





#####################################################
## Gillespie functions
#####################################################


def gillespie(param_dict, opt):
    model_name = opt.mdn
    listM4, listtime2, rr_list2, numberofiteration =[], [], [], int(opt.nit)
    Stoich, Stoich_df, header = get_stoich(opt)
    a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d3, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n = unpack_param_dict(param_dict)
    if (opt.wa2) == True:
        Decay3, a4, h4, leakage_A2 = param_dict['Decay3'], param_dict['a4'], param_dict['h4'], param_dict['leakage_A2']
    if opt.mdn == 'replaceA1':
        c2, c4 = float(param_dict['c2']), float(param_dict['c4'])

    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
        listM = get_listM(opt, param_dict)
        listM2, Time, listtime, rr_list, counter = init_iter()

        while Time < int(opt.tsp): 
            if bool(opt.wa2) == False: 
                HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM
            else: 
                HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB = listM

            d1, d4 = get_d1_d4(param_dict, Time, opt)
            listR = cal_basic_rate(a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d1, d3, d4, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n, HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB)
            if (opt.wa2) == True:
                R_HSFA2_dec=Decay3*HSFA2
                R_HSFA2_inc=leakage_A2+a4*HSFA1**n/(h4**n+HSFA1**n+HSFB**n)
                listR = np.append(listR, [R_HSFA2_inc, R_HSFA2_dec])
            if opt.mdn == 'replaceA1':
                MMP_replace_A1HSPR = c2*C_HSFA1_HSPR*MMP
                A1_replace_MMPHSPR = c4*C_HSPR_MMP*HSFA1
                listR = np.append(listR, [MMP_replace_A1HSPR, A1_replace_MMPHSPR])

            listM, Time, counter, listtime, listM2, rr_list = update_save_tsp(listR, Stoich, listM, Time, counter, listtime, listM2, rr_list, opt)

            rate_dict = dict(zip(Stoich_df.index.to_list(), listR))
            #print(rate_dict)
            max_rate = max(rate_dict, key=rate_dict.get)
            if counter % 5000 ==0 and counter != 0:
                #print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%, TotR: {sum(listR)}, max_rate: {max_rate} = {rate_dict[max_rate]}", end='\r')

        listM4, listtime2, rr_list2, end_time, param_dict = save_iter(listM2, listtime, rr_list, listM4, listtime2, rr_list2, Time, param_dict)
    return listM4, listtime2, numberofiteration, end_time, rr_list2, model_name


def unpack_param_dict(param_dict):
    a1 = float(param_dict['a1'])
    a2 = float(param_dict['a2'])
    a5 = float(param_dict['a5'])
    a6 = float(param_dict['a6'])
    a7 = float(param_dict['a7'])
    h1 = float(param_dict['h1'])
    h2 = float(param_dict['h2'])
    h5 = float(param_dict['h5'])
    c1 = float(param_dict['c1'])
    c3 = float(param_dict['c3'])
    #d1 = float(param_dict['d1'])
    d3 = float(param_dict['d3'])
    Decay1 = float(param_dict['Decay1'])
    Decay2 = float(param_dict['Decay2'])
    Decay4 = float(param_dict['Decay4'])
    Decay6 = float(param_dict['Decay6'])
    Decay7 = float(param_dict['Decay7'])
    Decay8 = float(param_dict['Decay8'])
    Decay5 = float(param_dict['Decay5'])
    leakage_A1 = float(param_dict['leakage_A1'])
    leakage_B = float(param_dict['leakage_B'])
    leakage_HSPR = float(param_dict['leakage_HSPR'])
    n = int(param_dict['hillcoeff'])
    return a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d3, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n

def get_stoich(opt):
    if bool(opt.wa2) == False:
        if opt.mdn == 'woA2' or opt.mdn == 'd1upCons':
            Stoich = [[1,0,0,0,0,0,0], #R_HSFA1_inc
              [-1,0,0,0,0,0,0], #R_HSFA1_dec
              [0,1,0,0,0,0,0], #R_HSPR_inc
              [0,-1,0,0,0,0,0], #R_HSPR_dec
              [-1,-1,1,0,0,0,0], #R_C_HSFA1_HSPR_inc
              [1,1,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec1
              [0,0,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec2
              [0,0,0,1,-1,0,0], #R_MMP_inc
              [0,0,0,-1,0,0,0], #R_MMP_dec
              [0,0,0,0,1,0,0], #R_FMP_inc
              [0,0,0,0,-1,0,0], #R_FMP_dec
              [0,-1,0,-1,0,1,0], #R_C_HSPR_MMP_inc
              [0,1,0,1,0,-1,0], #R_C_HSPR_MMP_dec1 = dissociation of the complex to form free HSPR and MMP
              [0,1,0,0,1,-1,0], #R_C_HSPR_MMP_dec2 = refolding step, dissociation of the complex to form free HSPR and FMP 
              [0,0,0,0,0,-1,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
              [0,0,0,0,0,0,1], #R_HSFB_inc
              [0,0,0,0,0,0,-1] #R_HSFB_dec
              ]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec'])
        elif opt.mdn == 'replaceA1':
            Stoich = [[1,0,0,0,0,0,0], #R_HSFA1_inc
              [-1,0,0,0,0,0,0], #R_HSFA1_dec
              [0,1,0,0,0,0,0], #R_HSPR_inc
              [0,-1,0,0,0,0,0], #R_HSPR_dec
              [-1,-1,1,0,0,0,0], #R_C_HSFA1_HSPR_inc
              [1,1,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec1
              [0,0,-1,0,0,0,0], #R_C_HSFA1_HSPR_dec2
              [0,0,0,1,-1,0,0], #R_MMP_inc
              [0,0,0,-1,0,0,0], #R_MMP_dec
              [0,0,0,0,1,0,0], #R_FMP_inc
              [0,0,0,0,-1,0,0], #R_FMP_dec
              [0,-1,0,-1,0,1,0], #R_C_HSPR_MMP_inc
              [0,1,0,1,0,-1,0], #R_C_HSPR_MMP_dec1 = dissociation of the complex to form free HSPR and MMP
              [0,1,0,0,1,-1,0], #R_C_HSPR_MMP_dec2 = refolding step, dissociation of the complex to form free HSPR and FMP 
              [0,0,0,0,0,-1,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
              [0,0,0,0,0,0,1], #R_HSFB_inc
              [0,0,0,0,0,0,-1], #R_HSFB_dec
              [1,0,-1,-1,0,1,0], #MMP_replace_A1HSPR
              [-1,0,1,1,0,-1,0] #A1_replace_MMPHSPR
              ]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
        else: print('opt.mdn empty or unexpected')
    elif bool(opt.wa2) == True:
        if opt.mdn == 'woA2' or opt.mdn == 'd1upCons':
            Stoich = [[1,0,0,0,0,0,0,0], 
                      [-1,0,0,0,0,0,0,0], 
                      [0,1,0,0,0,0,0,0], 
                      [0,-1,0,0,0,0,0,0], 
                      [-1,-1,1,0,0,0,0,0], 
                      [1,1,-1,0,0,0,0,0], 
                      [0,0,-1,0,0,0,0,0], 
                      [0,0,0,1,-1,0,0,0], 
                      [0,0,0,-1,0,0,0,0], 
                      [0,0,0,0,1,0,0,0], 
                      [0,0,0,0,-1,0,0,0], 
                      [0,-1,0,-1,0,1,0,0], 
                      [0,1,0,1,0,-1,0,0], 
                      [0,1,0,0,1,-1,0,0], 
                      [0,0,0,0,0,-1,0,0], 
                      [0,0,0,0,0,0,0,1], 
                      [0,0,0,0,0,0,0,-1],
                      [0,0,0,0,0,0,1,0], 
                      [0,0,0,0,0,0,-1,0],]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFA2', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3',  'R_HSFB_inc', 'R_HSFB_dec', 'R_HSFA2_inc', 'R_HSFA2_dec'])
        elif opt.mdn == 'replaceA1':
            Stoich = [[1,0,0,0,0,0,0,0], 
                      [-1,0,0,0,0,0,0,0], 
                      [0,1,0,0,0,0,0,0], 
                      [0,-1,0,0,0,0,0,0], 
                      [-1,-1,1,0,0,0,0,0], 
                      [1,1,-1,0,0,0,0,0], 
                      [0,0,-1,0,0,0,0,0], 
                      [0,0,0,1,-1,0,0,0], 
                      [0,0,0,-1,0,0,0,0], 
                      [0,0,0,0,1,0,0,0], 
                      [0,0,0,0,-1,0,0,0], 
                      [0,-1,0,-1,0,1,0,0], 
                      [0,1,0,1,0,-1,0,0], 
                      [0,1,0,0,1,-1,0,0], 
                      [0,0,0,0,0,-1,0,0], 
                      [0,0,0,0,0,0,0,1], 
                      [0,0,0,0,0,0,0,-1],
                      [1,0,-1,-1,0,1,0,0], #MMP_replace_A1HSPR
                      [-1,0,1,1,0,-1,0,0], #A1_replace_MMPHSPR
                      [0,0,0,0,0,0,1,0], 
                      [0,0,0,0,0,0,-1,0], 
                      ]
            Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFA2', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3',  'R_HSFB_inc', 'R_HSFB_dec','R_HSFA2_inc', 'R_HSFA2_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
        else: print('opt.mdn empty or unexpected')
    header = ['Iteration_Identifier', 'time'] + Stoich_df.columns.to_list() + Stoich_df.index.to_list()
    return Stoich, Stoich_df, header

def get_listM(opt, param_dict):
    if bool(opt.wa2) == False:
        listM = np.array([int(param_dict["init_HSFA1"]),
                      int(param_dict["init_HSPR"]),
                      int(param_dict["init_C_HSFA1_HSPR"]),
                      int(param_dict["init_MMP"]),
                      int(param_dict["init_FMP"]),
                      int(param_dict["init_C_HSPR_MMP"]),
                      int(param_dict["init_HSFB"])])
    else:
        listM = np.array([param_dict["init_HSFA1"],
                          param_dict["init_HSPR"],
                          param_dict["init_C_HSFA1_HSPR"],
                          param_dict["init_MMP"],
                          param_dict["init_FMP"],
                          param_dict["init_C_HSPR_MMP"],
                          param_dict["init_HSFA2"],
                          param_dict["init_HSFB"]])
    return listM

def init_iter():
    listM2 =[]
    Time=0
    listtime =[]
    rr_list = []
    counter = 0
    return listM2, Time, listtime, rr_list, counter

def get_d1_d4(param_dict, Time, opt):
    if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): 
        d4 = param_dict['d4_heat']
        if opt.mdn == 'd1upCons': 
            d1 = param_dict['d1_HS']
        else:
            d1 = param_dict['d1']
    elif bool(opt.hs2) == True:
        if Time >= int(opt.hs2) and Time <= int(opt.hs2) + int(opt.hsd): 
            d4 = param_dict['d4_heat']
            if opt.mdn == 'd1upCons': 
                d1 = param_dict['d1_HS']
            else:
                d1 = param_dict['d1']
        else: 
            d4 = param_dict['d4_norm']
            d1 = param_dict['d1']
    else: 
        d4 = param_dict['d4_norm']
        d1 = param_dict['d1']
    return d1, d4


def cal_basic_rate(a1, a2, a5, a6, a7, h1, h2, h5, c1, c3, d1, d3, d4, Decay1, Decay2, Decay4, Decay6, Decay7, Decay8, Decay5, leakage_A1, leakage_B, leakage_HSPR, n, HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB):
    R_HSFA1_inc=leakage_A1+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n) 
    R_HSFA1_dec= Decay1*HSFA1
    R_HSPR_inc= leakage_HSPR+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n)
    R_HSPR_dec= Decay2*HSPR
    R_C_HSFA1_HSPR_inc=c1*HSFA1*HSPR
    R_C_HSFA1_HSPR_dec1=d1*C_HSFA1_HSPR
    R_C_HSFA1_HSPR_dec2=Decay7*C_HSFA1_HSPR
    R_MMP_inc= d4*FMP
    R_MMP_dec= Decay5*MMP
    R_FMP_inc=a7
    R_FMP_dec= Decay6*FMP
    R_C_HSPR_MMP_inc=c3*HSPR*MMP
    R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP
    R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP
    R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP
    R_HSFB_inc=leakage_B+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
    R_HSFB_dec=Decay4*HSFB

    listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec])
    return listR


def update_save_tsp(listR, Stoich, listM, Time, counter, listtime, listM2, rr_list, opt):
    Tau = -math.log(random.random())/sum(listR) 
    #print(f'Stoich length = {len(Stoich)}, {Stoich}')
    #print(f'listR length = {len(listR)}, {listR}')

    Outcome = random.choices(Stoich, weights = listR, k=1)
    listM = listM+Outcome[0] 
    last_time = Time
    Time += Tau 
    counter += 1

    if "{:.1f}".format(Time) == "{:.1f}".format(last_time + opt.spf) or Time == 0:
        listtime.append("{:.1f}".format(Time)) #this is to add stuff to the list
        listM2.append(listM)
        rr_list.append(listR)
    return listM, Time, counter, listtime, listM2, rr_list

def save_iter(listM2, listtime, rr_list, listM4, listtime2, rr_list2, Time, param_dict):
    listM4.append(listM2)
    listtime2.append(listtime)
    rr_list2.append(rr_list)
    end_time = Time
    param_dict['end_time'] = end_time
    return listM4, listtime2, rr_list2, end_time, param_dict


#####################################################
## Process Gillespie Output
#####################################################

def combine_data(listtime2, listM4, rr_list2, opt):
    listM6 = []
    listM7 = []
    for Iteration_Identifier, (time_list, iter_conc_list, rate_list) in enumerate(zip(listtime2, listM4, rr_list2)):
        for time_step, conc_list, listR in zip(time_list[:-1], iter_conc_list[:-1],rate_list):
            listM7 = [f"Iteration {Iteration_Identifier}"]+ [time_step] + conc_list.tolist() + listR.tolist()
            listM6.append(listM7)
    return listM6


def Gillespie_list_to_df(header, listM6, opt):
    data_df = pd.DataFrame(listM6, columns = header)
    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    data_df['totalHSFA1'] = data_df['HSFA1'] + data_df['C_HSFA1_HSPR']
    conc_list = ['HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','totalHSPR', 'totalHSFA1']
    for column in header:
        if column == 'Iteration_Identifier': data_df[column] = data_df[column].astype(str)
        elif column in conc_list: data_df[column] = data_df[column].astype(int)
        else: data_df[column] = data_df[column].astype(float)
    grouped_data = data_df.groupby('Iteration_Identifier')
    return data_df, grouped_data



def saveData_oneSAstep(listM6, param_dir, numberofiteration, end_time, S, opt):

    date = datetime.now().date()
    data_file = f"{param_dir}/{S}_SimuData_{date}_{opt.mdn}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_withA2-{bool(opt.wa2)}.csv"
    data_file = get_unique_filename(data_file)

    data_name = f"{date}_{opt.mdn}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_withA2-{bool(opt.wa2)}_cosFunc-{opt.cof}"

    plot_title_to_show = f"{opt.mdn}_cosFunc-{opt.cof}_{S}_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_withA2-{bool(opt.wa2)}"

    Stoich, Stoich_df, header = get_stoich(opt)
    with open(data_file, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(header)
        csv_writer.writerows(listM6) 
    return data_file, date, data_name, plot_title_to_show



#####################################################
## Plotting
#####################################################

def plot_results(param_dir, data_name, data_df, grouped_data, param_dict, S, numberofiteration, hss, hsd, plot_title_to_show, opt):
    Stoich, Stoich_df, header = get_stoich(opt)
    conc = Stoich_df.columns.to_list()
    rates = Stoich_df.index.to_list()

    param_dict_text = param_dict_toText(param_dict, opt)

    plotReactionRate(rates, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt)
    plot_FMPMMPvsTime_2(conc, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt)
    #plot_FMPMMPvsTime_2_overlayed(conc, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt)
    plot_FMPMMP_zoom(conc, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt)




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


def saveFig(plot_dir, name_suffix, opt, prefix):
    if bool(opt.sfg) == True:
        #plot_name = f"{plot_dir}/{prefix}_{name_suffix}.pdf"
        plot_name = f"{plot_dir}/{prefix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        #print(f" save figure {opt.sfg == True}")


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


def plotReactionRate(rates, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt):
    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        #plot_trajectory(ax, data_df, 'time', rr, hss, hsd, "iteration 0")
        plot_trajectory(ax, fig, data_df,'time', rates, hss, hsd, plot_title_to_show, param_dict_text, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, data_df), ax in zip(grouped_data, ax):
            #plot_trajectory(ax, group_data, 'time', rr, hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax, fig, data_df,'time', rates, hss, hsd, plot_title_to_show, param_dict_text, Iteration_Identifier = Iteration_Identifier)
    saveFig(param_dir, data_name, opt, prefix =f'{S}_ReactionRate')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2(conc, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    #reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB', 'totalHSFA1']
    protein = ['FMP','MMP']
    reg = list(set(conc) - set(protein)) + ['totalHSFA1', 'totalHSPR']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0], fig, data_df, 'time', protein, hss, hsd,plot_title_to_show, param_dict_text, "iteration 0")
        plot_trajectory(ax[1], fig, data_df, 'time', reg, hss, hsd, plot_title_to_show, param_dict_text,"iteration 0")
        
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], fig, group_data, 'time', protein, hss, hsd,plot_title_to_show, param_dict_text, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], fig, group_data, 'time', reg, hss, hsd, plot_title_to_show, param_dict_text,Iteration_Identifier = Iteration_Identifier)
    saveFig(param_dir, data_name, opt, prefix =f'{S}_ProReg2')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2_overlayed(conc, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    #reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB', 'totalHSFA1']
    protein = ['FMP','MMP']
    reg = list(set(conc) - set(['FMP','MMP'])) + ['totalHSFA1', 'totalHSPR']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0],fig, data_df, 'time', protein, hss, hsd,plot_title_to_show, param_dict_text, "iteration 0")
        plot_trajectory(ax[1],fig, data_df, 'time', reg, hss, hsd,plot_title_to_show, param_dict_text, "iteration 0")

    else:
        fig, ax = plt.subplots(ncols = 2, figsize=(20,10))
        plot_trajectory(ax[0],fig, data_df, 'time', protein, hss, hsd, plot_title_to_show, param_dict_text,Iteration_Identifier = "all iter")
        plot_trajectory(ax[1],fig, data_df, 'time', reg, hss, hsd, plot_title_to_show, param_dict_text,Iteration_Identifier = "all iter")

    saveFig(param_dir, data_name, opt, prefix =f'{S}_ProReg2overlay')
    if bool(opt.shf) == True: plt.show()
    plt.close()



def plot_FMPMMP_zoom(conc, data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, S, plot_title_to_show, param_dict_text, opt):
    print(" Zoomed In Protein & Regulator Trajectory Around HeatShock")
    cut_data_df = data_df[(data_df['time'] >= hss -50) & (data_df['time'] <= hss + hsd + 100)]
    reg = list(set(conc) - set(['FMP','MMP']))+ ['totalHSFA1', 'totalHSPR']
    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0], fig, cut_data_df, 'time', ['FMP','MMP'], hss, hsd, plot_title_to_show, param_dict_text,"iteration 0")
        plot_trajectory(ax[1], fig, cut_data_df, 'time', reg, hss, hsd,plot_title_to_show, param_dict_text, "iteration 0")
    else:
        grouped_data = cut_data_df.groupby('Iteration_Identifier')
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], fig, group_data, 'time', ['FMP','MMP'], hss, hsd, plot_title_to_show, param_dict_text, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], fig, group_data, 'time', reg, hss, hsd, plot_title_to_show, param_dict_text, Iteration_Identifier = Iteration_Identifier)

    saveFig(param_dir, data_name, opt, prefix =f'{S}_ProRegZoom')
    if bool(opt.shf) == True: plt.show()
    plt.close()

######################################################
##  Stimulated Annealing
######################################################


def obj_func(data_df, ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, i, S_record, opt):
    ## Objective Function Calculation
    if len(S_record) == 0: S_old = 0
    else: S_old = S_record[-1]
    if opt.cof == 'fcA1tHSPR-pHSA1HSPR':
        S, cost_func = obj_func_fcA1tHSPRpHSA1HSPR(ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, opt)
    elif opt.cof == 'fctHSPR':
        S, cost_func = obj_func_fctHSPR(ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, opt)
    elif opt.cof == 'maxFMP':
        S, cost_func = obj_func_maxFMP(data_df, opt)
    elif opt.cof == 'maxFMP_lowleak':
        S, cost_func = obj_func_maxFMP_lowleakage(ssbHS_df, ssHS_df)
    elif opt.cof == 'maxFMP_lowleak_a1HFC':
        S, cost_func = obj_func_maxFMP_lowleakage_FC(ssbHS_df, ssHS_df, ssHS_lh_df)
    delta_S = S - S_old
    print(f"\n------> STEP {i}, S:{S}, delta_S: {delta_S}")
    return S, delta_S, cost_func

def accept_step(T, delta_S):
    if delta_S > 0: accept = 1
    else: #delta_S <= 0
        prob = math.exp(delta_S/T)
        print(f" prob = {prob}")
        accept = random.choices([0,1], weights=[(1-prob),prob], k=1)[0]
    return bool(accept)

def add_step(i, S, delta_S, param_dict, param_record, S_record, cost_func, listM6, param_dir, numberofiteration, end_time, hss, hsd, data_df, grouped_data, opt):
    print(f" add step, param_dict: {param_dict}\n")
    S_record = S_record + [S]
    param_dict_toSave = param_dict.copy()
    param_record.append(param_dict_toSave)

    data_file, date, data_name, plot_title_to_show = saveData_oneSAstep(listM6, param_dir, numberofiteration, end_time, S, opt)
    saveParam_csv_pcl(param_dict, param_dir, S, cost_func, opt)
    if bool(opt.prs) == True:
        plot_results(param_dir, data_name, data_df, grouped_data, param_dict, S, numberofiteration, hss, hsd, plot_title_to_show, opt)
    return S_record, param_record



def df_Processing_HS(data_df, hss,hsd, end_time, opt):
    if opt.tsp < 100: ss1_start = 0
    else: ss1_start = 100
    ss1_end = int(hss)
    ssHS_start = int(hss)
    ssHS_lh_start = (int(hss) + int(hsd))/2
    ssHS_end = int(hss) + int(hsd)
    ss3_start = ssHS_end
    ss3_end = end_time
    #print(f"hss:{hss}, hsd: {hsd}")
    #print(f"ss1: {ss1_start} - {ss1_end} \nssHS: {ssHS_start} - {ssHS_end} \nss3:{ss3_start} - {ss3_end} ")

    ssbHS_df = data_df[(data_df['time'] >= ss1_start) & (data_df['time'] <= ss1_end)]
    ssHS_df = data_df[(data_df['time'] >= ssHS_start) & (data_df['time'] <= ssHS_end)]
    ssHS_lh_df = data_df[(data_df['time'] >= ssHS_lh_start) & (data_df['time'] <= ssHS_end)]
    sspHS_df = data_df[(data_df['time'] >= ss3_start) & (data_df['time'] <= ss3_end)]

    return ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df


def obj_func_fcA1tHSPRpHSA1HSPR(ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, opt):
    preHS_A1_ave = ssbHS_df['HSFA1'].mean()
    preHS_B_ave = ssbHS_df['HSFB'].mean()
    preHS_totalHSPR_ave = ssbHS_df['totalHSPR'].mean()
    preHS_freeHSPR_ave = ssbHS_df['HSPR'].mean()

    HSlh_A1_ave = ssHS_lh_df['HSFA1'].mean()
    HSlh_B_ave = ssHS_lh_df['HSFB'].mean()
    HSlh_totalHSPR_ave = ssHS_lh_df['totalHSPR'].mean()

    try:
        logFC_A1 = math.log2(HSlh_A1_ave/preHS_A1_ave)
        #print(f"logFC_A1: {logFC_A1}")
        logFC_totalHSPR = math.log2(HSlh_totalHSPR_ave/preHS_totalHSPR_ave)
        #print(f"logFC_totalHSPR: {logFC_totalHSPR}")
        print(f' HSlh_A1_ave: {HSlh_A1_ave}\n preHS_A1_ave: {preHS_A1_ave}\n HSlh_totalHSPR_ave: {HSlh_totalHSPR_ave}\n preHS_totalHSPR_ave: {preHS_totalHSPR_ave}')
    except ValueError:
        print(f' HSlh_A1_ave: {HSlh_A1_ave}\n preHS_A1_ave: {preHS_A1_ave}\n HSlh_totalHSPR_ave: {HSlh_totalHSPR_ave}\n preHS_totalHSPR_ave: {preHS_totalHSPR_ave}')
        exit()

    small_A1_pHS = preHS_A1_ave - 0.5
    #print(f"small_A1_pHS: {small_A1_pHS}")
    large_freeHSPR_pHS = preHS_freeHSPR_ave-2
    #print(f"large_freeHSPR_pHS: {large_freeHSPR_pHS}")

    S = logFC_A1*logFC_totalHSPR*(10*large_freeHSPR_pHS + 5*small_A1_pHS)
    cost_func = 'logFC_A1*logFC_totalHSPR*(10*large_freeHSPR_pHS + 5*small_A1_pHS)'
    #print(f"S: {S}")

    return S, cost_func

def obj_func_fctHSPR(ssbHS_df, ssHS_df, ssHS_lh_df, sspHS_df, param_dict, opt):
    preHS_totalHSPR_ave = ssbHS_df['totalHSPR'].mean()
    HSlh_totalHSPR_ave = ssHS_lh_df['totalHSPR'].mean()
    if preHS_totalHSPR_ave == 0:
        S = 0
    else:
        S = math.log2(HSlh_totalHSPR_ave/preHS_totalHSPR_ave)
    cost_func = 'math.log2(HSlh_totalHSPR_ave/preHS_totalHSPR_ave)'
    return S, cost_func



def obj_func_maxFMP(data_df, opt):
    meanFMP = data_df['FMP'].mean()
    S = meanFMP/100
    cost_func = 'meanFMP/100'
    return S, cost_func

def obj_func_maxFMP_lowleakage(ssbHS_df, ssHS_df):
    preHS_FMP_mean = ssbHS_df['FMP'].mean()
    HS_FMP_min = ssHS_df['FMP'].min()
    preHS_tA1_ave = ssbHS_df['totalHSFA1'].mean()
    preHS_tHSPR_ave = ssbHS_df['totalHSPR'].mean()
    print(f' preHS_FMP_mean: {preHS_FMP_mean}\n HS_FMP_min: {HS_FMP_min}\n preHS_tA1_ave: {preHS_tA1_ave} \n preHS_tHSPR_ave: {preHS_tHSPR_ave}')
    S = 0.01*preHS_FMP_mean + 0.01*HS_FMP_min - preHS_tA1_ave - preHS_tHSPR_ave
    cost_func = '0.01*preHS_FMP_mean + 0.01*HS_FMP_min - preHS_tA1_ave - preHS_tHSPR_ave'
    return S, cost_func

def obj_func_maxFMP_lowleakage(ssbHS_df, ssHS_df):
    preHS_FMP_mean = ssbHS_df['FMP'].mean()
    HS_FMP_min = ssHS_df['FMP'].min()
    preHS_tA1_ave = ssbHS_df['totalHSFA1'].mean()
    preHS_tHSPR_ave = ssbHS_df['totalHSPR'].mean()
    print(f' preHS_FMP_mean: {preHS_FMP_mean}\n HS_FMP_min: {HS_FMP_min}\n preHS_tA1_ave: {preHS_tA1_ave} \n preHS_tHSPR_ave: {preHS_tHSPR_ave}')
    S = 0.01*preHS_FMP_mean + 0.01*HS_FMP_min - preHS_tA1_ave - preHS_tHSPR_ave
    cost_func = '0.01*preHS_FMP_mean + 0.01*HS_FMP_min - preHS_tA1_ave - preHS_tHSPR_ave'
    return S, cost_func

def obj_func_maxFMP_lowleakage_FC(ssbHS_df, ssHS_df, ssHS_lh_df):
    preHS_FMP_mean = ssbHS_df['FMP'].mean()
    HS_FMP_min = ssHS_df['FMP'].min()
    preHS_tA1_ave = ssbHS_df['totalHSFA1'].mean()
    preHS_tHSPR_ave = ssbHS_df['totalHSPR'].mean()
    preHS_A1_ave = ssbHS_df['HSFA1'].mean()
    preHS_totalHSPR_ave = ssbHS_df['totalHSPR'].mean()

    HSlh_A1_ave = ssHS_lh_df['HSFA1'].mean()
    HSlh_totalHSPR_ave = ssHS_lh_df['totalHSPR'].mean()
    A1FC = HSlh_A1_ave/preHS_A1_ave
    tHSPR_FC = HSlh_totalHSPR_ave/preHS_totalHSPR_ave

    S = 0.001*preHS_FMP_mean + 0.001*HS_FMP_min - preHS_tA1_ave - 0.1*preHS_tHSPR_ave + 10*A1FC + 10*tHSPR_FC
    print(f' preHS_FMP_mean: {preHS_FMP_mean}\n HS_FMP_min: {HS_FMP_min}\n preHS_tA1_ave: {preHS_tA1_ave} \n preHS_tHSPR_ave: {preHS_tHSPR_ave} \nA1FC:{A1FC} \ntHSPR_FC: {tHSPR_FC}')
    S = 0.001*preHS_FMP_mean + 0.01*HS_FMP_min - preHS_tA1_ave - 0.1*preHS_tHSPR_ave + 10*(A1FC-1) + 10*(tHSPR_FC-1)
    cost_func = '0.001*preHS_FMP_mean + 0.01*HS_FMP_min - preHS_tA1_ave - 0.1*preHS_tHSPR_ave + 10*(A1FC-1) + 10*(tHSPR_FC-1)'
    return S, cost_func

def updatePara_unif(param_dict, opt):
    param_dict['a1'] = random.uniform(5,50) # max A1 transcription rate, default = 10
    param_dict['a2'] = random.uniform(50,100) # max HSPR transcription rate, default = 100
    param_dict['a5'] = random.uniform(5,50) # max HSFB transcription rate, default = 5
    #param_dict['a6'] = random.uniform(0.2,20) # refolding rate from MMP-HSPR, default = 0.2
    param_dict['h1'] = random.uniform(1,50)
    param_dict['h2'] = random.uniform(1,50)
    param_dict['h5'] = random.uniform(1,50)
    param_dict['c1'] = random.uniform(5,15)
    param_dict['c2'] = random.uniform(0.5,5) # rate of MMP replacing A1 in HSPR complex
    param_dict['c3'] = random.uniform(5,15)
    param_dict['c4'] = random.uniform(0.01,2) # rate of A1 replacing MMP in HSPR complex
    param_dict['d1'] = random.uniform(0.01, 10.0) #A1_HSPR dissociation
    param_dict['d3'] = random.uniform(0.01, 5.0) #MMP_HSPR dissociation

    param_dict['leakage_A1'] = random.uniform(0.001, 0.01)
    param_dict['leakage_B'] = random.uniform(0.001, 0.01)
    param_dict['leakage_HSPR'] = random.uniform(0.001, 0.01)

    ##default of non-MMP decay rate is 0.04
    param_dict['Decay1'] = random.uniform(0.01,0.4) # A1
    param_dict['Decay2'] = random.uniform(0.01,0.4) # free HSPR
    param_dict['Decay4'] = random.uniform(0.01,0.4) # HSFB
    param_dict['Decay7'] = random.uniform(0.01,0.4) # A1-HSPR 
    param_dict['Decay8'] = random.uniform(0.01,0.4) # MMP_HSPR
    if bool(opt.wa2) == True:
        param_dict['leakage_A2'] = random.uniform(0.001, 0.01)
        param_dict['a4'] = random.uniform(5,50)
        param_dict['h4'] = random.uniform(1,50)
        param_dict['Decay3'] = random.uniform(0.01,0.4)
    return param_dict


def updatePara_one_normal(param_dict, opt):
    param_to_update = ['a1','a2','a5','h1','h2','h5','c1','c2','c3','c4','d1','d3'

    param_dict['leakage_A1'] = random.uniform(0.001, 0.01)
    param_dict['leakage_B'] = random.uniform(0.001, 0.01)
    param_dict['leakage_HSPR'] = random.uniform(0.001, 0.01)

    ##default of non-MMP decay rate is 0.04
    param_dict['Decay1'] = random.uniform(0.01,0.4) # A1
    param_dict['Decay2'] = random.uniform(0.01,0.4) # free HSPR
    param_dict['Decay4'] = random.uniform(0.01,0.4) # HSFB
    param_dict['Decay7'] = random.uniform(0.01,0.4) # A1-HSPR 
    param_dict['Decay8'] = random.uniform(0.01,0.4) # MMP_HSPR
    if bool(opt.wa2) == True:
        param_dict['leakage_A2'] = random.uniform(0.001, 0.01)
        param_dict['a4'] = random.uniform(5,50)
        param_dict['h4'] = random.uniform(1,50)
        param_dict['Decay3'] = random.uniform(0.01,0.4)


def updatePara_int(param_dict, opt):
    #start_t = time.time()
    #print(" update param start")
    param_dict['a1'] = random.randint(1,30) # max A1 transcription rate, default = 10, search range 1-30
    param_dict['a2'] = 10*(random.randint(1,21)) # max HSPR transcription rate, default = 100, search range 10-200
    param_dict['a5'] = random.randint(1,30) # max HSFB transcription rate, default = 5, search range 1-30
    param_dict['a6'] = 10**(random.randint(-2,1)) # refolding rate from MMP-HSPR, default = 0.2, search range = 0.01-1
    #param_dict['a7'] = 10*random.randint(1, 11) # folded protein production rate, default = 10, search range = 10-100
    param_dict['h1'] = param_dict['h2'] = param_dict['h5'] = random.randint(1,10)
    #param_dict['h2'] = random.randint(1,10)
    #param_dict['h5'] = random.randint(1,10)
    param_dict['c1'] = random.randint(5,30) # 5-30
    param_dict['c3'] = random.randint(1,6)
    param_dict['d1'] = 10**(random.randint(-2,1)) #search range = 0.01-1
    param_dict['d3'] = 10**(random.randint(-2,1))#search range = 0.01-1
    #param_dict['d4_norm'] = 10**(random.randint(-2,1)) # search range = 0.01-1
    #param_dict['d4_heat'] = param_dict['d4_norm']*5
    #param_dict['Decay1'] = param_dict['Decay2'] = param_dict['Decay4'] = param_dict['Decay6'] = param_dict['Decay7'] = param_dict['Decay8'] = 10**random.randint(-3, 0)

    #param_dict['Decay5'] = param_dict['Decay1']*(10**random.randint(1,3))#MMP decay rate
    param_dict['hillcoeff']  = random.randint(1,3)
    param_dict['leakage'] = 0.001
    #time_used = time.time()-start_t
    #print(f" update param, time used = {time_used}")

    R_sum = 0
    list = ['a1','a2','a5','a6','a7','h1','h2','h5','c1','c3','d1','d3','Decay1','Decay2','Decay4','Decay5','Decay6','Decay7','Decay8','leakage','hillcoeff']
    for it in list:
        R_sum += param_dict[it]
    print(f"R_sum: {R_sum}")
    return param_dict






#################################################################
## Other Functions
#################################################################





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




def df_split(data_df, param_dict, opt):
    ss1_start = 1000
    ss1_end = int(opt.hss)
    ssHS_start = int(opt.hss) + 500
    ssHS_end = int(opt.hss) + int(opt.hsd)
    ss3_start = ssHS_end + 500
    ss3_end = param_dict['end_time']
    print(f"hss:{opt.hss}, hsd: {opt.hsd}")
    print(f"ss1: {ss1_start} - {ss1_end} \nssHS: {ssHS_start} - {ssHS_end} \nss3:{ss3_start} - {ss3_end} ")

    ss1_df = data_df[(data_df['time'] >= ss1_start) & (data_df['time'] <= ss1_end)]
    ssHS_df = data_df[(data_df['time'] >= ssHS_start) & (data_df['time'] <= ssHS_end)]
    ss3_df = data_df[(data_df['time'] >= ss3_start)]

    #print("ss1_df")
    #print(ss1_df)
    #print(ss1_df.shape)
    #print("ssHS_df")
    #print(ssHS_df)
    #print(ssHS_df.shape)
    #print("ss3_df")
    #print(ss3_df)
    #print(ss3_df.shape)

    return ss1_df, ssHS_df, ss3_df





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

