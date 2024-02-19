script_usage="""
usage
    HSPR_AZ_SA.py -ops <optSteps> -nit <numberInteration> [options]

version
    HSPR_AZ_SA.py 0.0.2 (alpha)

dependencies
    Python v3.9.7, Scipy v1.11.2, NumPy v1.22.2, viennarna v2.5.1, Matplotlib v3.5.1, pandas v2.1.0

description


################################################################

--optSteps,-ops
    how many steps the simulated annealing algorithm is gonna take (default:10)

--numberInteration,-nit
    The number of interations for Gillespi simulation

--timeStep,-tsp
    The time duration for each Gillespi simulation run/iteration (default:1200)

--heatShockStart,-hss
    The time point at which a heat shock is introduced (default:700)

--heatShockDuration,-hsd
    The duration of heat shock introduced (default: 3)

--misfoldRateNormal,-mfn
    The formation rate of misfolded protein from folded protein under normal temperature (default: 0.01)

--misfoldRateHS,-mfh
    The formation rate of misfolded protein from folded protein under heat shock condition (default: 0.05)

--assoA1_HSPR,-aah
    c1, association rate between A1 and HSPR (default:10.0)

--repMMP_A1H,-rma
    c2, rate at which MMP replaces A1 from A1_HSPR complex, to form MMP_HSPR instead (default: 5.0)

--assoMMP_HSPR,-amh
    c3, The association rate between MMP and HSPR (default: 5.0)

--repA1_MMPH,-ram
    c4, rate at which A1 replaces MMP from MMP_HSPR complex, to form A1_HSPR instead (default: 10.0)

--disoA1_HSPR,-dah
    d1, dissociation rate of A1-HSPR (default: 0.1)

--HSdisoA1_HSPR,-hda
    d1_HS, dissociation rate of A1-HSPR (default: 0.1)

--disoMMP_HSPR,-dmh
    d3, dissociation rate of A1-HSPR (default: 0.01)

--hillCoeff,-hco
    The Hill Coefficient (default: 2)

--A2positiveAutoReg,-a2p
    Whether HSFA2 positively regulates itself in the model (default: 0)

--leakage,-lkg
    Trancription leakage (default: 0.001)

--hilHalfSaturation,-hhs
    The conc of inducer/repressor at half-max transcriptional rate (default: 1.0)

--KA1actA1,-hAA
    h1, K constant for activating A1 (default: 1.0)

--KA1actHSPR,-hAH
    h2, K constant for activating HSPR (default: 1.0)

--KA1actB,-hAB
    h5, K constant for activating HSFB (default: 1.0)

--KBrepA1,-hBA
    h3, K constant for HSFB repressing A1 (default: 1.0)

--KBrepB,-hBB
    h6, K constant for HSFB repressing HSFB (default: 1.0)

--KBrepHSPR,-hBH
    h4, K constant for HSFB repressing HSPR (default: 1.0)

--initFMP,-ifp
    Initial FMP abundance (default: 50)

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

--decayA1,-da1
    decay1, decay rate of free A1 (default: 0.01)

--maxRateB,-a5B
    a5, Max transcription rate of HSFB (default: 10)

--maxRateA1,-a1A
    a1, Max transcription rate of HSFA1 (default: 10)

--maxRateHSPR,-a2H
    a2, Max transcription rate of HSFB (default: 100)

--foldedProduction,-fpp
    a7, folded protein production rate (default: 300)

-refoldRate,-a6R
    a6, refolding rate from MMP-HSPR (default: 0.2)

--globalDecayRate,-gdr
    the decay rate of species except for MMP (default:0.01)

--A1mut,-a1m
    if 1, no HSFA1 (default: 0)

--Bmut,-bmu
    if 1, no HSFB (default: 0)

--HSPRmut,-hmu
    if 1, no HSPR (default: 0)

--outputFormat,-ofm
    Whether to save Gillespi simulation output as csv or picklefile (default: csv)

--samplingFreq,-spf
    How often is simulation datasaved (default: 0.1)

--thread,-thr
    The number of threads used for multiprocessing (default: 4)

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
    param_dir, plot_dir, data_dir = dir_gen()
    param_dict = param_spec(opt)
    #print(param_dict)
    param_explore(param_dir, plot_dir, data_dir, param_dict, opt)

    S_record, param_record = simuAnneal(param_dict, opt)

    save_param(S_record, param_record, opt, param_dir)


def param_explore(param_dir, plot_dir, data_dir, param_dict, opt):
    for i in int(opt.ops):
        ## one simulation step
        listM4, listtime2, numberofiteration, end_time, rr_list2 = gillespie_woA2_replaceA1(param_dict, opt)
        listM6 = combine_data(listtime2, listM4, rr_list2, opt)
        data_file, date, data_name = saveGilData_replace(listM6, data_dir, numberofiteration, end_time, opt)
        saveParam(param_dir, data_dir, numberofiteration, end_time, opt)
        data_df, grouped_data, Rows, Columns = import_tidy_simuData(data_dir, numberofiteration, data_file)

        ## Objective Function Calculation
        S = obj_func(data_df, param_dict, opt)

        # Plot Results
        plot_results(param_dir, data_name, data_df, grouped_data, param_dict, S, end_time, date, opt)

        param_dict_new = updatePara(param_dict, opt)
        param_dict = param_dict_new



def plot_results(param_dir, data_name, data_df, grouped_data, param_dict, S, end_time, date, opt):
    numberofiteration = param_dict['numberofiteration']
    hss = param_dict['hstart']
    hsd = param_dict['hduration']

    name_suffix, diff_dict, name_suffix_save = genPlotName_nondefault(param_dict, numberofiteration, end_time, hss, hsd, date, opt)

    plotReactionRate(data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, name_suffix, S, opt)
    plot_FMPMMPvsTime_2(data_df, grouped_data, param_dir, numberofiteration,data_name, hss, hsd, name_suffix, S, opt)
    plot_FMPMMPvsTime_2_overlayed(data_df, param_dir, numberofiteration, data_name, hss, hsd, name_suffix, S, opt)
    plot_FMPMMP_zoom(data_df, hss, hsd, param_dir, numberofiteration, data_name, name_suffix, S, opt)


######################################################
## 1. Specify directory and initial params
######################################################

def dir_gen():
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)

    data_dir = os.path.join(partiii_dir,"Ritu_simulation_data")
    if not os.path.isdir(data_dir): os.makedirs(data_dir, 0o777)
    param_dir = os.path.join(partiii_dir,"Param_Optimisation")
    if not os.path.isdir(param_dir): os.makedirs(param_dir, 0o777)
    plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)
    return param_dir, plot_dir, data_dir


def param_spec(opt):
    if bool(opt.a1m) == False:
        init_HSFA1, a1 = int(opt.iaf), int(opt.a1A)
    elif bool(opt.a1m) == True: 
        print("A1 mutant")
        init_HSFA1, a1 = 0, 0

    if bool(opt.bmu) == False: 
        init_HSFB, a5 = int(opt.ibf), int(opt.a5B)
    elif bool(opt.bmu) == True: 
        print("HSFB mutant")
        init_HSFB,a5 = 0,0

    if bool(opt.hmu) == False:
        init_C_HSPR_MMP, init_HSPR, a2 = int(opt.imh), int(opt.ihf), int(opt.a2H)
    elif bool(opt.hmu) == True: 
        print("HSPR mutant")
        init_C_HSPR_MMP, init_HSPR, a2 = 0,0,0

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
        #'init_HSFA2': 1,
        'init_HSFB': init_HSFB,
        'Time': 0.0,
        ## Maximum expression level in Hill equation
        'a1': a1,
        'a2': a2,
        #'a3': 5.0,
        #'a4': 5.0,
        'a5': a5,
        'a6': float(opt.a6R), # refolding rate from MMP-HSPR
        'a7': int(opt.fpp), #folded protein production rate
        #'a8': 50.0,
        ## Ka in Hill equation
        #'h1': int(opt.hhs),
        #'h2': int(opt.hhs),
        #'h5': int(opt.hhs),
        'h1': float(opt.hAA),
        'h2': float(opt.hAH),
        #'h3': int(opt.hBA),
        #'h4': int(opt.hBH),
        'h5': float(opt.hAB),
        #'h6': int(opt.hBB),
        ## association rates
        'c1': float(opt.aah), #between A1 and HSPR
        'c2': float(opt.rma), # MMP replace A1 in complex with HSPR
        'c3': float(opt.amh), #between MMP and HSPR
        'c4': float(opt.ram), # A1 replace MMP in complex with HSPR
        ## decay rates
        'd1': float(opt.dah), # decay path 1 of A1-HSPR
        'd3': float(opt.dmh), # dissociation rate of MMP-HSPR
        'd4_heat': float(opt.mfh),
        'd4_norm': float(opt.mfn),
        'Decay1': float(opt.gdr),
        'Decay2': float(opt.gdr), # decay of free HSPR
        #'Decay3': float(opt.gdr),
        'Decay4': float(opt.gdr),
        'Decay6': float(opt.gdr),
        'Decay7': float(opt.gdr), # decay path 2 of A1-HSPR
        'Decay8': float(opt.gdr), # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage': float(opt.lkg),
        'numberofiteration': int(opt.nit),
        'hillcoeff': int(opt.hco),
        'hstart':int(opt.hss),
        'hduration':int(opt.hsd)
    }
    #print(param_dict['a1'])  # prints the value of a1
    #print(param_dict['init_HSFA1'])  # prints the value of init_HSFA1
    print(param_dict)
    return param_dict


def gillespie_woA2_replaceA1(param_dict, opt):
    listM4=[]
    listtime2=[]
    rr_list2 = []
    numberofiteration = param_dict["numberofiteration"]
    
    a1 = param_dict['a1']
    a2 = param_dict['a2']
    a5 = param_dict['a5']
    a6 = param_dict['a6']
    a7 = param_dict['a7']
    h1 = param_dict['h1']
    h2 = param_dict['h2']
    h5 = param_dict['h5']
    c1 = param_dict['c1']
    c2 = param_dict['c2']
    c3 = param_dict['c3']
    c4 = param_dict['c4']
    d1 = param_dict['d1']
    d3 = param_dict['d3']
    Decay1 = param_dict['Decay1']
    Decay2 = param_dict['Decay2']
    Decay4 = param_dict['Decay4']
    Decay6 = param_dict['Decay6']
    Decay7 = param_dict['Decay7']
    Decay8 = param_dict['Decay8']
    Decay5 = param_dict['Decay5']
    leakage = param_dict['leakage']
    n = param_dict['hillcoeff']

    if bool(opt.a1m) == False: leakage_A1 = leakage
    elif bool(opt.a1m) == True: 
        leakage_A1 = 0
        print("A1 leakage = 0")
    if bool(opt.bmu) == False: leakage_B = leakage
    elif bool(opt.bmu) == True: 
        leakage_B = 0
        print("B leakage = 0")
    if bool(opt.hmu) == False: leakage_HSPR = leakage
    elif bool(opt.hmu) == True: 
        leakage_HSPR = 0
        print("HSPR leakage = 0")
    
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
    #print(Stoich_df)
    #exit()
    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
        listM = np.array([param_dict["init_HSFA1"],
                      param_dict["init_HSPR"],
                      param_dict["init_C_HSFA1_HSPR"],
                      param_dict["init_MMP"],
                      param_dict["init_FMP"],
                      param_dict["init_C_HSPR_MMP"],
                      param_dict["init_HSFB"]])
        listM2 =[listM]
        Time=0
        listtime =[Time]
        rr_list = []
        counter = 0

        while Time < int(opt.tsp): 
            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
            else: d4 = param_dict['d4_norm']
                

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
            MMP_replace_A1HSPR = c2*C_HSFA1_HSPR*MMP
            A1_replace_MMPHSPR = c4*C_HSPR_MMP*HSFA1


            listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec,MMP_replace_A1HSPR,A1_replace_MMPHSPR])
            TotR = sum(listR) #production of the MRNA 
            Rn = random.random() #getting random numbers
            Tau=-math.log(Rn)/TotR #when the next thing happen
            #Rn2= random.uniform(0,TotR) # for the next random number
            # HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB
            Outcome = random.choices(Stoich, weights = listR, k=1)
            #print(f"listM before: {listM}")
            #print(f"outcome: {Outcome} \n {Outcome[0]}")
            listM = listM+Outcome[0] ### why does it add term-by-term??? -> because listM is a np.array
            #print(f"listM after: {listM}")
            #exit()
            last_time = Time
            Time+=Tau # the new time the time before the step +the time to happen the next step ()
            counter += 1

            if "{:.1f}".format(Time) == "{:.1f}".format(last_time + opt.spf):
                listtime.append("{:.1f}".format(Time)) #this is to add stuff to the list
                listM2.append(listM)
                rr_list.append(listR)
                #print(rr_list[-1][7])
        listM4.append(listM2)
        listtime2.append(listtime)
        rr_list2.append(rr_list)
        end_time = Time
        param_dict['end_time'] = end_time
        #print(rr_list)
    return listM4, listtime2, numberofiteration, end_time, rr_list2

def combine_data(listtime2, listM4, rr_list2, opt):
    listM6 = []
    listM7 = []
    for Iteration_Identifier, (time_list, iter_conc_list, rate_list) in enumerate(zip(listtime2, listM4, rr_list2)):
        for time_step, conc_list, listR in zip(time_list[:-1], iter_conc_list[:-1],rate_list):
            listM7 = [f"Iteration {Iteration_Identifier}"]+ [time_step] + conc_list.tolist() + listR.tolist()
            listM6.append(listM7)
    return listM6


def saveGilData_replace(list, data_dir, numberofiteration, end_time, opt):
    # Name output file
    date = datetime.now().date()
    if opt.ofm == "csv":
        data_name = f'{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}'
        data_file = f"{data_dir}/Exp3_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
        data_file = get_unique_filename(data_file)
        print(data_file)
        # Open the CSV file in write mode with the specified directory and file name
        with open(data_file, 'w') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
            csv_writer.writerows(list) #how different it is to use .writerow and .writerows
    return data_file, date, data_name

def saveParam(param_dict, data_dir, numberofiteration, end_time, opt):
    date = datetime.now().date()
    if opt.ofm == "csv":
        param_name = f"{data_dir}/Exp3_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
        param_outfile = get_unique_filename(param_name)
        header = param_dict.keys()
        with open(param_outfile, 'w', newline='') as csvfile_2:
            # Create a CSV writer object
            writer = csv.DictWriter(csvfile_2, fieldnames=header)
            # Write the header
            writer.writeheader()
            # Write the parameter values
            writer.writerow(param_dict)
    print(f" Parameters Saved as {opt.ofm}")



#####################################################
## Plotting
#####################################################

def import_tidy_simuData(data_dir, numberofiteration, data_file):
    data_df = pd.read_csv(f"{data_file}")
    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    #print(data_df)
    #print(data_df.shape)
    ### number of rows and columns for all iterations
    Rows = int(math.sqrt(numberofiteration))
    Columns = int(math.ceil(numberofiteration/ Rows))
    grouped_data = data_df.groupby('Iteration_Identifier')
    return data_df, grouped_data, Rows, Columns


def genPlotName_nondefault(param_dict, numberofiteration, end_time, hss, hsd, date, opt):
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
        'a6': 0.2, # refolding rate from MMP-HSPR
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
        'Decay1': 0.01,
        'Decay2': 0.01, # decay of free HSPR
        #'Decay3': 0.01,
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
    name_suffix_save = f"{date}_numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}"
    name_suffix = f"{date}_numIter{numberofiteration}_Time{end_time}_HSstart{hss}_HSduration{hsd}\n"
    for key, val in diff_dict.items():
        name_suffix += f"_{key}-{val}"
        name_suffix_save += f"_{key}-{val}"
    return name_suffix, diff_dict, name_suffix_save

##################################################
##################################################


def plot_trajectory(ax, data_df, x_col, y_col_list, hss, hsd, Iteration_Identifier):
    for y_col in y_col_list:
        ax.plot(data_df[x_col], data_df[y_col], label=y_col, linewidth=1, alpha=0.8)
    ax.set_xlabel('Time (hour)')
    ax.set_ylabel('Protein copy number')
    ax.axvspan(hss, hss+hsd, facecolor='r', alpha=0.2)
    ax.set_title(f"{Iteration_Identifier}")
    ax.legend(loc='upper left')
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)

def saveFig(plot_dir, name_suffix, opt, prefix):
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/{prefix}_{name_suffix}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f" save figure {opt.sfg == True}")


def plotReactionRate(data_df, grouped_data, param_dir, numberofiteration, data_name, hss, hsd, name_suffix, S, opt):
    rr = ['R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR']

    if numberofiteration == 1:
        fig, ax = plt.subplots(figsize=(15, 5))
        plot_trajectory(ax, data_df, 'time', rr, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, figsize=(15,5*numberofiteration))
        ax = ax.flatten() # Flatten the 2D array of subplots to a 1D array
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):
            plot_trajectory(ax, group_data, 'time', rr, hss, hsd, Iteration_Identifier = Iteration_Identifier)

    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()
    saveFig(plot_dir, data_name, opt, prefix =f'{S}_ReactionRate')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2(data_df, grouped_data, plot_dir, numberofiteration, data_name, hss, hsd, name_suffix, S, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB']
    protein = ['FMP','MMP']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0], data_df, 'time', protein, hss, hsd, "iteration 0")
        plot_trajectory(ax[1], data_df, 'time', reg, hss, hsd, "iteration 0")
    else:
        fig, ax = plt.subplots(nrows= numberofiteration, ncols = 2, figsize=(20,5*numberofiteration))
        for i, (Iteration_Identifier, group_data) in enumerate(grouped_data):# Now 'ax' is a 1D array, and you can iterate over it
            plot_trajectory(ax[i,0], group_data, 'time', protein, hss, hsd, Iteration_Identifier = Iteration_Identifier)
            plot_trajectory(ax[i,1], group_data, 'time', reg, hss, hsd, Iteration_Identifier = Iteration_Identifier)
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(plot_dir, data_name, opt, prefix =f'{S}_ProReg2')
    if bool(opt.shf) == True: plt.show()
    plt.close()


def plot_FMPMMPvsTime_2_overlayed(data_df, plot_dir, numberofiteration, data_name, hss, hsd, name_suffix, S, opt):

    print(" Plot trajectories of Proteins and Regulators")
    #HSPR_complex = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR']
    reg = ['C_HSPR_MMP','C_HSFA1_HSPR','totalHSPR','HSPR','HSFA1','HSFB']
    protein = ['FMP','MMP']

    if numberofiteration == 1:
        fig, ax = plt.subplots(ncols=2, figsize=(20, 5))
        plot_trajectory(ax[0], data_df, 'time', protein, hss, hsd, "iteration 0")
        plot_trajectory(ax[1], data_df, 'time', reg, hss, hsd, "iteration 0")

    else:
        fig, ax = plt.subplots(ncols = 2, figsize=(20,10))
        plot_trajectory(ax[0], data_df, 'time', protein, hss, hsd, Iteration_Identifier = "all iter")
        plot_trajectory(ax[1], data_df, 'time', reg, hss, hsd, Iteration_Identifier = "all iter")
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    #fig.suptitle('Trajectories of Proteins and Regulators')
    fig.text(0.5, 0.99, name_suffix, ha = 'center', va='center', wrap=True)
    fig.suptitle(' ', fontsize=16, y = 1)
    plt.tight_layout()

    saveFig(plot_dir, data_name, opt, prefix =f'{S}_ProReg2overlay')
    if bool(opt.shf) == True: plt.show()
    plt.close()



def plot_FMPMMP_zoom(data_df, hss, hsd, plot_dir, numberofiteration, data_name, name_suffix, S, opt):
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

    saveFig(plot_dir, data_name, opt, prefix =f'{S}_ProRegZoom')
    if bool(opt.shf) == True: plt.show()
    plt.close()

######################################################
## 2. Stimulated Annealing
######################################################

def simuAnneal(param_dict, opt):
    S_record = []
    param_record = []
    T = 100
    for i in range(opt.ops):
        print(f"{i}th step----------------------------------------")
        if i == 0 :
            start_t = time.time()
            S = one_opt_step(param_dict, opt)
            S_record.append(S)
            param_record.append(param_dict)
            time_used = time.time()-start_t
            print(f" S = {S}")
        else: 
            start_t = time.time()
            T = 0.95*T
            S_old = S_record[-1]
            param_dict_new = updatePara(param_dict, opt)
            print(f"    {param_dict_new}")
            S = one_opt_step(param_dict_new, opt)
            delta_S = S - S_old
            #print(f" S = {S}, delta_S = {delta_S}")
            if delta_S < 0:
                param_dict = param_dict_new
                S_record.append(S)
                param_record.append(param_dict)
                time_used = time.time()-start_t
                print(f" accept,  S = {S}, delta_S = {delta_S}")
            else:
                prob = math.exp(-delta_S/T)
                print(f" prob = {prob}")
                accept = random.choices([0,1], weights=[(1-prob),prob], k=1)[0]
                print(f"  accept?: {accept}")
                if accept == 0: 
                    #time_used = time.time()-start_t
                    #print(f"  reject, time used = {time_used}")
                    print(f"  reject,  S = {S}, delta_S = {delta_S}")
                    continue
                else:
                    param_dict = param_dict_new
                    S_record.append(S)
                    param_record.append(param_dict)
                    #time_used = time.time()-start_t
                    #print(f"  accept, time used = {time_used}")
                    print(f"  accept,  S = {S}, delta_S = {delta_S}")
    return S_record, param_record

##### Functions in SimuAnneal()

def one_opt_step(param_dict, opt):
    listM4, listtime2 =Gillespie_1step(param_dict, opt)
    data_df = data_to_df(listtime2,listM4)
    S = obj_func(data_df, param_dict, opt)
    return S




def Gillespie_1step(param_dict, opt):
    print(" Gillespie start")
    start_time_g = time.time()
    listM4, listtime2 = [], []
    a1 = param_dict['a1']
    a2 = param_dict['a2']
    a5 = param_dict['a5']
    a6 = param_dict['a6']
    a7 = param_dict['a7']
    h1 = param_dict['h1']
    h2 = param_dict['h2']
    h5 = param_dict['h5']
    c1 = param_dict['c1']
    c3 = param_dict['c3']
    d1 = param_dict['d1']
    d3 = param_dict['d3']
    Decay1 = param_dict['Decay1']
    Decay2 = param_dict['Decay2']
    Decay4 = param_dict['Decay4']
    Decay5 = param_dict['Decay5']
    Decay6 = param_dict['Decay6']
    Decay7 = param_dict['Decay7']
    Decay8 = param_dict['Decay8']
    leakage = param_dict['leakage']
    n = param_dict['hillcoeff']

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
    #Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec'])

    listM = np.array([param_dict["init_HSFA1"],
                      param_dict["init_HSPR"],
                      param_dict["init_C_HSFA1_HSPR"],
                      param_dict["init_MMP"],
                      param_dict["init_FMP"],
                      param_dict["init_C_HSPR_MMP"],
                      param_dict["init_HSFB"]])
    listM2 =[listM]
    Time=0
    listtime =[Time]
    counter = 0
    while Time < int(opt.tsp): 
        if counter % 5000 ==0 and counter != 0:
            print(f"  Progress: {float(Time*100/int(opt.tsp))}%", end='\r')
        HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM
        if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
        else: d4 = param_dict['d4_norm']
            
        #HSFa1 andHSFA2 may makes complex
        #increase in HSFA1 by transcription and dessociation from the complex C_HSFA1_HSPR
        R_HSFA1_inc=leakage+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n) # + d1*C_HSFA1_HSPR
        #decrease in HSFA1 by association to the 1st complex C_HSFA1_HSPR and decay in the protein
        R_HSFA1_dec= Decay1*HSFA1
        #increase in HSPR by transcription and dess
        R_HSPR_inc= leakage+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n)
        #decrease in HSPR by transcription and dess **-> should be decay
        R_HSPR_dec= Decay2*HSPR
        #increase in C_HSFA1_HSPR association to the 1st complex
        R_C_HSFA1_HSPR_inc=c1*HSFA1*HSPR
        #decrease in C_HSFA1_HSPR dissociation from the 1st complex and degradation of the complex as a whole (?)
        R_C_HSFA1_HSPR_dec1=d1*C_HSFA1_HSPR
        R_C_HSFA1_HSPR_dec2=Decay7*C_HSFA1_HSPR
        #increase in MMP when 2nd complex decreases
        R_MMP_inc= d4*FMP
        #decrease in MMP by 2nd complex increases (may be we want to change the slope of dexcay later)
        R_MMP_dec= Decay5*MMP
        #increase in FMP by FMP production and MMP to FMP
        R_FMP_inc=a7  #how to write the production?
        #decrease in FMP by decay or
        R_FMP_dec= Decay6*FMP
        #increase in HSPR_MMP by association to the 2nd complex
        R_C_HSPR_MMP_inc=c3*HSPR*MMP #how to write the production?
        #decrease in HSPR_MMP by dissociation from the 2nd complex
        R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP
        R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP
        R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP
        #increase in HSFB by transcription with TF HSFA1 and HSFB
        R_HSFB_inc=leakage+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
        #decrease in HSFB by transcription and dess
        R_HSFB_dec=Decay4*HSFB
        listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec])
        TotR = np.sum(listR) #production of the MRNA 
        if counter >= 5000 and counter <=5010: 
            print(f"    sum of rate = {TotR}\n    reaction rate: {listR}")
        
        Rn = random.random() #getting random numbers
        Tau=-math.log(Rn)/TotR #when the next thing happen
        #Rn2= random.uniform(0,TotR) # for the next random number
        # HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB
        Outcome = random.choices(Stoich, weights = listR, k=1)
        #print(f"listM before: {listM}")
        #print(f"outcome: {Outcome} \n {Outcome[0]}")
        listM = listM+Outcome[0] ### why does it add term-by-term??? -> because listM is a np.array
        #print(f"listM after: {listM}")
        #exit()
        last_time = Time
        Time+=Tau # the new time the time before the step +the time to happen the next step ()
        counter += 1
        # print (Time,listM)
        if int(Time) == int(last_time) + opt.spf:
            listtime.append(int(Time)) #this is to add stuff to the list
            listM2.append(listM)
    listM4.append(listM2)
    listtime2.append(listtime)
    end_time = Time
    #param_dict['end_time'] = end_time
    print(f"    simulation time = {time.time()-start_time_g}")
    return listM4, listtime2

def data_to_df(listtime2,listM4):
    listM6, listM7 = [],[]
    for Iteration_Identifier, (time_list, iter_conc_list) in enumerate(zip(listtime2,listM4)):
        for time_step, conc_list in zip(time_list, iter_conc_list):
            listM7 = [f"Iteration {Iteration_Identifier}"]+ [time_step] + conc_list.tolist()
            #print(f"listM7: {listM7}")
            #print(f"conc_list: {conc_list.tolist()}")
            listM6.append(listM7)
    data_df = pd.DataFrame(listM6, columns = ['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB'])
    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    data_df['totalMMP'] = data_df['MMP'] + data_df['C_HSPR_MMP']
    return data_df

def obj_func(data_df, param_dict, opt):
    #ss1_df, ssHS_df, ss3_df = df_split(data_df, param_dict, opt)
    # high FMP throughout
    fmp = data_df['FMP'].mean()
    w_fmp = 1
    # low total MMP through out
    mmp_t = data_df['totalMMP'].mean()
    w_mmp_t = 1

    S = - w_fmp*fmp + w_mmp_t*mmp_t
    return S




def updatePara_1(param_dict, opt):
    #start_t = time.time()
    #print(" update param start")
    param_dict['a1'] = random.uniform(1,30) # max A1 transcription rate, default = 10, search range 1-30
    param_dict['a2'] = 5*10**(random.uniform(1,3)) # max HSPR transcription rate, default = 100, search range 50-5000
    param_dict['a5'] = random.uniform(1,30) # max HSFB transcription rate, default = 5, search range 1-30
    param_dict['a6'] = math.exp(random.uniform(-4.6,1)) # refolding rate from MMP-HSPR, default = 0.2, search range = 0.01-1
    param_dict['a7'] = pow(10, random.uniform(1, 4)) # folded protein production rate, default = 10
    param_dict['h1'] = random.uniform(1,10)
    param_dict['h2'] = random.uniform(1,10)
    param_dict['h5'] = random.uniform(1,10)
    param_dict['c1'] = random.uniform(5,30) # 5-30
    param_dict['c3'] = random.uniform(1,5)
    param_dict['d1'] = math.exp(random.uniform(-4.6,1)) #search range = 0.01-1
    param_dict['d3'] = math.exp(random.uniform(-4.6,1))#search range = 0.01-1
    param_dict['d4_norm'] = math.exp(random.uniform(-4.6,1)) # search range = 0.01-1
    param_dict['d4_heat'] = param_dict['d4_norm']*5
    param_dict['Decay1'] = param_dict['Decay2'] = param_dict['Decay4'] = param_dict['Decay6'] = param_dict['Decay7'] = param_dict['Decay8'] = pow(10, random.uniform(-3, -1))

    param_dict['Decay5'] = param_dict['Decay1']*pow(10,random.uniform(1,3))#MMP decay rate
    param_dict['hillcoeff']  = random.choices([1,2],k=1)[0]
    param_dict['leakage'] = 0.001
    #time_used = time.time()-start_t
    param_sum = sum(param_dict.keys())
    print(f" updated param, param_sum = {param_sum}")
    return param_dict


def updatePara(param_dict, opt):
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
## Save output
#################################################################



def save_param(S_record, param_record, opt, param_dir):
    date = datetime.now().date()
    data_file = f"{param_dir}/SimuAnneal_{date}_step{opt.ops}.csv"
    data_file = get_unique_filename(data_file)
    #print(data_file)
    keys = param_record[0].keys()

    with open(data_file, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)

        header_row = ['S'] + list(keys)
        csv_writer.writerow(header_row)
        for (dict, S) in zip(param_record, S_record):
            row = [S] + list(dict.values())
            csv_writer.writerow(row)






#################################################################
## Other Functions
#################################################################








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

