#!/usr/bin/env python3
# script: HSPR_AZ_tidy.py
# author: Rituparna Goswami, Enrico Sandro Colizzi, Alexis Jiayi Zeng
script_usage="""
usage
    HSPR_AZ_hpc.py -nit <numberInteration> -tsp <timeStep> [options]

version
    HSPR_AZ_hpc.py 0.0.2 (alpha)

dependencies
    Python v3.9.7, Scipy v1.11.2, NumPy v1.22.2, viennarna v2.5.1, Matplotlib v3.5.1, pandas v2.1.0

description
    Re-organised the original code in functions. Introduced parser & flags for easy change of iteration numbers and time step. (HSPR_AZ_tidy.py)

    Stored parameters in a dictionary. Changed naming convention of output files (HSPR_AZ_v2.py)

    Combined plotting function. Introduced progress reporter. Output format either csv or pcl. Optional figure saving (HSPR_AZ_v3.py)

    Introduced options to directly import simulation data for plotting by updating opt. Changed saveGilData() from list*3 to list*2 (HSPR_AZ_v5.py) 

    Save data at customisable time step - default = 1. Changed Hill Coeff to 2. Histogram plotting cleaned (HSPR_AZ_v6.py, Nov 22nd 2023) 

################################################################

--numberInteration,-nit
    The number of interations for Gillespi simulation

--importParamDataset,-ids
    Dataset name from which parameters are extracted. e.g. from SA '2024-02-24_step2_time900_hss600_hsd50\11.792037432365467' from simuData 'Exp3_Para_2023-11-26_numIter1_Time20000.002736231276_HSstart10000_HSduration5000' (default: )

--modelName,-mdn
    which model version or Gillespie Function to use (default: )

--timeStep,-tsp
    The time duration for each Gillespi simulation run/iteration (default:1000)

--heatShockStart,-hss
    The time point at which a heat shock is introduced (default:600)

--heatShockStart2,-hs2
    The time point at which a second heat shock is introduced (default: 0)

--heatShockDuration,-hsd
    The duration of heat shock introduced (default: 30)

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

--leakage_A1,-lga
    Trancription leakage for HSFA1 (default: 0.001)

--leakage_B,-lgb
    Trancription leakage for HSFB (default: 0.001)

--leakage_HSPR,-lgh
    Trancription leakage for HSPR (default: 0.001)

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
    the decay rate of species except for MMP (default:0.04)

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
import multiprocessing as mp
import multiprocessing as mp


def main(opt):
    print("Step1: Specify output directory")
    data_dir, plot_dir, param_rootdir = dir_gen()

    print("Step2: Specify parameters")
    if bool(opt.ids) == False: # de novo setting
        param_dict = param_spec(opt)
    else: 
        param_dict, opt = load_Param_fromFile(param_rootdir, data_dir, opt)
    
    print("Step3: Simulation begins")
    ## no multiprocessing
    if opt.mdn == 'woA2':
        listM4, listtime2, numberofiteration, end_time, rr_list2, model_name = gillespie_woA2(param_dict, opt)
    elif opt.mdn == 'replaceA1':
        listM4, listtime2, numberofiteration, end_time, rr_list2, model_name = gillespie_woA2_replaceA1(param_dict, opt)
    elif opt.mdn == 'd1upCons':
        listM4, listtime2, numberofiteration, end_time, rr_list2, model_name = gillespie_woA2_d1HS(param_dict, opt)
    else: 
        print('unexpected or unspecified model name')
        exit()
    listM6 = combine_data(listtime2, listM4, rr_list2, opt)

    ## with multiprocessing
    #listM2, listtime, end_time = gillespie_woA2_mp(param_dict, opt)
    #listM6, numberofiteration, end_time = parallel_gillespie_woA2(param_dict, opt)

    print("Step4: Combine and save data")
    if bool(opt.ids) == False:
        if opt.mdn == "replaceA1":
            data_file = saveGilData_replace(listM6, data_dir, numberofiteration, end_time, model_name, opt)
        else: 
            data_file = saveGilData_2(listM6, data_dir, numberofiteration, end_time, model_name, opt)
        param_outfile = saveParam(param_dict, data_dir, numberofiteration, end_time, model_name, opt)
    else:
        saveData_oneSAstep(listM6, param_rootdir, numberofiteration, end_time, opt)






##################################################################
## To Extract From Existing File
##################################################################

def SA_param_extract_csv(param_rootdir, data_dir, opt):
    if os.path.exists(f"{param_rootdir}/{opt.ids}"):
        para_csv_name = f"{param_rootdir}/{opt.ids}"
    elif os.path.exists(f"{data_dir}/Exp3_Para_{opt.ids}"):
        para_csv_name = f"{data_dir}/Exp3_Para_{opt.ids}"
        model_name = "woA2"
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
        numeric_data = [float(val) for val in data]
        param_dict = dict(zip(headers, numeric_data))
    #for key, val in param_dict.items():
    #    print(f"{key} - {type(val)}")
    print(param_dict)
    S = param_dict['S']
    return param_dict, opt


def SA_param_extract_pcl(param_rootdir, opt):
    para_pcl_name = f"{param_rootdir}/{opt.ids}.pcl"
    S, cost_func, param_dict = loadData(para_pcl_name)
    print(f"S: {S}, param_dict: {param_dict}")
    param_dict['S'] = S
    return param_dict

def load_Param_fromFile(param_rootdir, data_dir, opt):
    try: 
        S, cost_func, param_dict = loadData(f"{param_rootdir}/{opt.ids}.pcl")
        opt.S = S
        opt.cost_func = cost_func
    except FileNotFoundError:
        if os.path.exists(f"{data_dir}/Exp3_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/Exp3_Para_{opt.ids}.csv"
            model_name = "woA2"
        elif os.path.exists(f"{data_dir}/replaceA1_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/replaceA1_Para_{opt.ids}.csv"
            model_name = "replaceA1"
        elif os.path.exists(f"{data_dir}/woA2_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/woA2_Para_{opt.ids}.csv"
            model_name = "woA2"
        elif os.path.exists(f"{data_dir}/d1upCons_Para_{opt.ids}.csv"):
            para_csv_name = f"{data_dir}/d1upCons_Para_{opt.ids}.csv"
            model_name = "d1upCons"
        with open(para_csv_name, 'r') as param_file:
            csv_reader = csv.reader(param_file)
            headers = next(csv_reader)
            data = next(csv_reader)
            numeric_data = [float(val) for val in data]
            param_dict = dict(zip(headers, numeric_data))
    if not 'hstart2' in param_dict: param_dict['hstart2'] = 0
    param_dict['model_name'], opt.mdn = model_name, model_name
    param_dict['numberofiteration'] = int(opt.nit)
    return param_dict, opt



#######################################################################
## 1. Parameter specification
#######################################################################
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
        #'init_HSFA2': 1,
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
        'c1': float(opt.aah), # binding rate between A1 and HSPR
        'c3': float(opt.amh), # binding rate between MMP and HSPR
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
    if opt.mdn == 'replaceA1':
        param_dict['c2'] = float(opt.rma) # MMP replace A1 in complex with HSPR
        param_dict['c4'] = float(opt.ram) # A1 replace MMP in complex with HSPR
    elif opt.mdn == 'd1upCons':
        param_dict['d1_HS'] == float(opt.hda)
    print(param_dict)
    return param_dict

##########################################################################
## 2. Generate Output Directories
##########################################################################
def dir_gen():
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)

    data_dir = os.path.join(partiii_dir,"Ritu_simulation_data")
    if not os.path.isdir(data_dir): os.makedirs(data_dir, 0o777)
    plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)
    param_rootdir = os.path.join(partiii_dir,"Param_Optimisation")
    if not os.path.isdir(param_rootdir): os.makedirs(plot_dir, 0o777)
    return data_dir, plot_dir, param_rootdir

############################################################################
## 3. Gillespi Simulation
############################################################################

def gillespi_archive(param_dict, opt):
    listM4=[]
    listtime2=[]
    numberofiteration = param_dict["numberofiteration"]
    for i in range(numberofiteration):    
        print(f" iteration: {i}\n")
        listM = np.array([param_dict["init_HSFA1"],
                          param_dict["init_HSPR"],
                          param_dict["init_C_HSFA1_HSPR"],
                          param_dict["init_MMP"],
                          param_dict["init_FMP"],
                          param_dict["init_C_HSPR_MMP"],
                          param_dict["init_HSFA2"],
                          param_dict["init_HSFB"]])
        listM2 =[listM]
        Time=0
        listtime =[Time]

        counter = 0
        while Time < int(opt.tsp): 
            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')

            a1 = param_dict['a1']
            a2 = param_dict['a2']
            a3 = param_dict['a3']
            a4 = param_dict['a4']
            a5 = param_dict['a5']
            a6 = param_dict['a6']
            a7 = param_dict['a7']
            a8 = param_dict['a8']
            h1 = param_dict['h1']
            h2 = param_dict['h2']
            h3 = param_dict['h3']
            h4 = param_dict['h4']
            h5 = param_dict['h5']
            h6 = param_dict['h6']
            c1 = param_dict['c1']
            c3 = param_dict['c3']
            d1 = param_dict['d1']
            d3 = param_dict['d3']
            Decay1 = param_dict['Decay1']
            Decay2 = param_dict['Decay2']
            Decay3 = param_dict['Decay3']
            Decay4 = param_dict['Decay4']
            Decay6 = param_dict['Decay6']
            Decay7 = param_dict['Decay7']
            Decay8 = param_dict['Decay8']
            Decay5 = param_dict['Decay5']
            leakage = param_dict['leakage']
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
            else: d4 = param_dict['d4_norm']
                
            #HSFa1 andHSFA2 may makes complex
            #increase in HSFA1 by transcription and dessociation from the complex C_HSFA1_HSPR
            R_HSFA1_inc=leakage+a1*HSFA1/(h1+HSFA1+HSFB)+ a3*HSFA2/(h3+HSFA2+HSFB) # + d1*C_HSFA1_HSPR
            #decrease in HSFA1 by association to the 1st complex C_HSFA1_HSPR and decay in the protein
            R_HSFA1_dec= Decay1*HSFA1
            #increase in HSPR by transcription and dess
            R_HSPR_inc= leakage+a2*HSFA1/(h2+HSFA1+HSFB)+ a3*HSFA2/(h3+HSFA2+HSFB)
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
            #increase in HSFA2 by transcription with TF HSFA1 
            R_HSFA2_inc=leakage+a4*HSFA1/(h4+HSFA1+HSFB) # + a8*HSFA2/(h6+HSFA2+HSFB)
            #decrease in HSFA2 by transcription and dess
            R_HSFA2_dec=Decay3*HSFA2
            #increase in HSFB by transcription with TF HSFA1 and HSFB
            R_HSFB_inc=leakage+a5*HSFA1/(h5+HSFA1+HSFB)
            #decrease in HSFB by transcription and dess
            R_HSFB_dec=Decay4*HSFB


            listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3, R_HSFA2_inc,R_HSFA2_dec,R_HSFB_inc,R_HSFB_dec])

            TotR = sum(listR) #production of the MRNA 
            Rn = random.random() #getting random numbers
            Tau=-math.log(Rn)/TotR #when the next thing happen
            #Rn2= random.uniform(0,TotR) # for the next random number
            # HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB
            Stoich = [[1,0,0,0,0,0,0,0], [-1,0,0,0,0,0,0,0], [0,1,0,0,0,0,0,0], [0,-1,0,0,0,0,0,0], [-1,-1,1,0,0,0,0,0], [1,1,-1,0,0,0,0,0],[0,0,-1,0,0,0,0,0],
                    [0,0,0,1,-1,0,0,0], [0,0,0,-1,0,0,0,0], [0,0,0,0,1,0,0,0], [0,0,0,0,-1,0,0,0], [0,-1,0,-1,0,1,0,0], 
                    [0,1,0,1,0,-1,0,0], #R_C_HSPR_MMP_dec2 = dissociation of the complex to form free HSPR and MMP -> dissociation
                    [0,1,0,0,1,-1,0,0], #R_C_HSPR_MMP_dec2 = dissociation of the complex to form free HSPR and FMP -> refolding step
                    [0,0,0,0,0,-1,0,0], #R_C_HSPR_MMP_dec3, complex decrease by 1, decay 8
                    [0,0,0,0,0,0,1,0], [0,0,0,0,0,0,-1,0], [0,0,0,0,0,0,0,1], [0,0,0,0,0,0,0,-1]]

            Outcome = random.choices(Stoich, weights = listR, k=1)
            #print(f"listM before: {listM}")
            #print(f"outcome: {Outcome} \n {Outcome[0]}")
            listM = listM+Outcome[0] ### why does it add term-by-term??? -> because listM is a np.array
            #print(f"listM after: {listM}")
            #exit()
            Time+=Tau # the new time the time before the step +the time to happen the next step ()
            counter += 1
            # print (Time,listM)
            listtime.append(Time) #this is to add stuff to the list
            listM2.append(listM)
        listM4.append(listM2)
        listtime2.append(listtime)
        end_time = Time
    return listM4, listtime2, numberofiteration, end_time

def gillespie_woA2_mp(param_dict, opt):
    listM = np.array([param_dict["init_HSFA1"], param_dict["init_HSPR"], param_dict["init_C_HSFA1_HSPR"], param_dict["init_MMP"], param_dict["init_FMP"], param_dict["init_C_HSPR_MMP"], param_dict["init_HSFB"]])
    a1 = param_dict['a1']
    a2 = param_dict['a2']
    #a3 = param_dict['a3']
    #a4 = param_dict['a4']
    a5 = param_dict['a5']
    a6 = param_dict['a6']
    a7 = param_dict['a7']
    #a8 = param_dict['a8']
    h1 = param_dict['h1']
    h2 = param_dict['h2']
    #h3 = param_dict['h3']
    #h4 = param_dict['h4']
    h5 = param_dict['h5']
    #h6 = param_dict['h6']
    c1 = param_dict['c1']
    c3 = param_dict['c3']
    d1 = param_dict['d1']
    d3 = param_dict['d3']
    Decay1 = param_dict['Decay1']
    Decay2 = param_dict['Decay2']
    #Decay3 = param_dict['Decay3']
    Decay4 = param_dict['Decay4']
    Decay6 = param_dict['Decay6']
    Decay7 = param_dict['Decay7']
    Decay8 = param_dict['Decay8']
    Decay5 = param_dict['Decay5']
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

    listM2 =[listM]
    Time=0
    listtime =[Time]
    counter = 0

    while Time < int(opt.tsp): 
        HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM
        if counter % 5000 ==0 and counter != 0:
            print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')
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
        TotR = sum(listR) #production of the MRNA 
        Rn = random.random() #getting random numbers
        Tau=-math.log(Rn)/TotR #when the next thing happen
        #Rn2= random.uniform(0,TotR) # for the next random number
        # HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB

        Stoich_df = pd.DataFrame(Stoich, columns= ['HSFA1', 'HSPR', 'C_HSFA1_HSPR', 'MMP', 'FMP', 'C_HSPR_MMP', 'HSFB'], index=['R_HSFA1_inc', 'R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc','R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3', 'R_HSFB_inc', 'R_HSFB_dec'])
        print(Stoich_df)
        exit()
        Outcome = random.choices(Stoich, weights = listR, k=1)
        #print(f"listM before: {listM}")
        #print(f"outcome: {Outcome} \n {Outcome[0]}")
        listM = listM+Outcome[0] ### why does it add term-by-term??? -> because listM is a np.array
        print(f"listM after: {listM}")
        #exit()
        last_time = Time
        Time+=Tau # the new time the time before the step +the time to happen the next step ()
        counter += 1
        # print (Time,listM)
        if int(Time) == int(last_time) + opt.spf:
            listtime.append(Time) #this is to add stuff to the list
            listM2.append(listM)
    end_time = Time
    return listM2, listtime, end_time

def parallel_gillespie_woA2(param_dict, opt):
    numberofiteration = param_dict["numberofiteration"]
    arg = [(param_dict,opt)]*opt.thr
    #print(arg)
    with mp.Pool(processes= opt.thr) as pool:
        results = pool.starmap(gillespie_woA2_mp, arg)
    listM4 = []
    listtime2 = []
    listM6 = []
    listM7 = []
    all_end_time = []
    for i, iter_result in enumerate(results):
        listM2, listtime, end_time = iter_result
        all_end_time.append(end_time)
        listM4.append(listM2)
        listtime2.append(listtime)
    for time_step, conc_list in zip(listtime2, listM4):
        listM7 = [f"Iteration {i}"]+ [time_step] + conc_list
        listM6.append(listM7)
    param_dict['end_time'] = max(all_end_time)
    max_end_time = max(all_end_time)
    return listM6, numberofiteration, max_end_time

###### No multiprocessing, original, start
def gillespie_woA2(param_dict, opt):
    model_name = 'woA2'
    listM4=[]
    listtime2=[]
    rr_list2 = []
    numberofiteration = int(opt.nit)
    
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
    d1 = float(param_dict['d1'])
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
    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
        listM = np.array([int(param_dict["init_HSFA1"]),
                      int(param_dict["init_HSPR"]),
                      int(param_dict["init_C_HSFA1_HSPR"]),
                      int(param_dict["init_MMP"]),
                      int(param_dict["init_FMP"]),
                      int(param_dict["init_C_HSPR_MMP"]),
                      int(param_dict["init_HSFB"])])
        listM2 =[listM]
        Time=0
        listtime =[Time]
        rr_list = []
        counter = 0

        while Time < int(opt.tsp): 
            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): 
                d4 = param_dict['d4_heat']
            elif bool(opt.hs2) == True:
                if Time >= int(opt.hs2) and Time <= int(opt.hs2) + int(opt.hsd): d4 = param_dict['d4_heat']
                else: d4 = param_dict['d4_norm']
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


            listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec])
            #print(R_MMP_inc)


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

            #listtime.append(Time)
            #listM2.append(listM)
            #rr_list.append(listR)

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
    return listM4, listtime2, numberofiteration, end_time, rr_list2, model_name



def gillespie_woA2_d1HS(param_dict, opt):
    model_name = 'd1upCons'
    listM4=[]
    listtime2=[]
    rr_list2 = []
    numberofiteration = int(opt.nit)
    if float(opt.hda) != float(opt.dah):
        print(f"HS-induced d1 change: normal d1: {opt.dah}, d1 during HS: {opt.hda} ")
    
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
    d1 = float(param_dict['d1'])
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
    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
        listM = np.array([int(param_dict["init_HSFA1"]),
                      int(param_dict["init_HSPR"]),
                      int(param_dict["init_C_HSFA1_HSPR"]),
                      int(param_dict["init_MMP"]),
                      int(param_dict["init_FMP"]),
                      int(param_dict["init_C_HSPR_MMP"]),
                      int(param_dict["init_HSFB"])])
        listM2 =[listM]
        Time=0
        listtime =[Time]
        rr_list = []
        counter = 0

        while Time < int(opt.tsp): 
            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%", end='\r')
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): 
                d4 = param_dict['d4_heat']
                if float(opt.hda) != float(opt.dah):
                    d1 = float(opt.hda)
            elif bool(opt.hs2) == True:
                if Time >= int(opt.hs2) and Time <= int(opt.hs2) + int(opt.hsd): d4 = param_dict['d4_heat']
                else: d4 = param_dict['d4_norm']
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


            listR = np.array([R_HSFA1_inc, R_HSFA1_dec, R_HSPR_inc, R_HSPR_dec, R_C_HSFA1_HSPR_inc, R_C_HSFA1_HSPR_dec1,R_C_HSFA1_HSPR_dec2,R_MMP_inc,R_MMP_dec,R_FMP_inc,R_FMP_dec,R_C_HSPR_MMP_inc,R_C_HSPR_MMP_dec1, R_C_HSPR_MMP_dec2, R_C_HSPR_MMP_dec3,R_HSFB_inc,R_HSFB_dec])
            #print(R_MMP_inc)


            TotR = sum(listR) #production of the MRNA 
            Rn = random.random() #getting random numbers
            Tau=-math.log(Rn)/TotR 
            Outcome = random.choices(Stoich, weights = listR, k=1)
            listM = listM+Outcome[0] 
            last_time = Time
            Time+=Tau 
            counter += 1

            if "{:.1f}".format(Time) == "{:.1f}".format(last_time + opt.spf):
                listtime.append("{:.1f}".format(Time)) #this is to add stuff to the list
                listM2.append(listM)
                rr_list.append(listR)


        listM4.append(listM2)
        listtime2.append(listtime)
        rr_list2.append(rr_list)

        end_time = Time
        param_dict['end_time'] = end_time
        #print(rr_list)
    return listM4, listtime2, numberofiteration, end_time, rr_list2, model_name


def gillespie_woA2_replaceA1(param_dict, opt):
    model_name = "replaceA1"
    listM4=[]
    listtime2=[]
    rr_list2 = []
    numberofiteration = int(opt.nit)
    
    a1 = float(param_dict['a1'])
    a2 = float(param_dict['a2'])
    a5 = float(param_dict['a5'])
    a6 = float(param_dict['a6'])
    a7 = float(param_dict['a7'])
    h1 = float(param_dict['h1'])
    h2 = float(param_dict['h2'])
    h5 = float(param_dict['h5'])
    c1 = float(param_dict['c1'])
    c2 = float(param_dict['c2'])
    c3 = float(param_dict['c3'])
    c4 = float(param_dict['c4'])
    d1 = float(param_dict['d1'])
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
    n = int(float(param_dict['hillcoeff']))
    
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
        listM = np.array([int(float(param_dict["init_HSFA1"])),
                      int(float(param_dict["init_HSPR"])),
                      int(float(param_dict["init_C_HSFA1_HSPR"])),
                      int(float(param_dict["init_MMP"])),
                      int(float(param_dict["init_FMP"])),
                      int(float(param_dict["init_C_HSPR_MMP"])),
                      int(float(param_dict["init_HSFB"]))])
        listM2 =[listM]
        Time=0
        listtime =[Time]
        rr_list = []
        counter = 0

        while Time < int(opt.tsp): 
            
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
            elif bool(opt.hs2) == True:
                if Time >= int(opt.hs2) and Time <= int(opt.hs2) + int(opt.hsd): d4 = param_dict['d4_heat']
                else: d4 = param_dict['d4_norm']
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
            TotR = sum(listR) 
            Rn = random.random() 
            Tau=-math.log(Rn)/TotR 
            Outcome = random.choices(Stoich, weights = listR, k=1)
            listM = listM+Outcome[0] 
            last_time = Time
            Time+=Tau 
            counter += 1

            ## Capping Reaction Rates
            rate_dict ={'R_HSFA1_inc': R_HSFA1_inc,'R_HSFA1_dec': R_HSFA1_dec, 'R_HSPR_inc': R_HSPR_inc, 'R_HSPR_dec':R_HSPR_dec, 'R_C_HSFA1_HSPR_inc': R_C_HSFA1_HSPR_inc, 'R_C_HSFA1_HSPR_dec1': R_C_HSFA1_HSPR_dec1,'R_C_HSFA1_HSPR_dec2': R_C_HSFA1_HSPR_dec2,'R_MMP_inc': R_MMP_inc,'R_MMP_dec': R_MMP_dec,'R_FMP_inc': R_FMP_inc, 'R_FMP_dec': R_FMP_dec,'R_C_HSPR_MMP_inc': R_C_HSPR_MMP_inc,'R_C_HSPR_MMP_dec1': R_C_HSPR_MMP_dec1, 'R_C_HSPR_MMP_dec2': R_C_HSPR_MMP_dec2, 'R_C_HSPR_MMP_dec3': R_C_HSPR_MMP_dec3,'R_HSFB_inc': R_HSFB_inc,'R_HSFB_dec': R_HSFB_dec,'MMP_replace_A1HSPR': MMP_replace_A1HSPR,'A1_replace_MMPHSPR': A1_replace_MMPHSPR}

            max_rate = max(rate_dict, key=rate_dict.get)

            if counter % 5000 ==0 and counter != 0:
                print(f"  Progress: {int(Time*100/int(opt.tsp))}%, TotR: {TotR}, max_rate: {max_rate} = {rate_dict[max_rate]}", end='\r')


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
    return listM4, listtime2, numberofiteration, end_time, rr_list2, model_name



def combine_data(listtime2, listM4, rr_list2, opt):
    #to combine the data 
    listM6 = []
    listM7 = []
    # listM = list of protein conc at a single time point
    # listM2 = list of listM, storing conc at each time point in a single iteration
    # listM4 =list of listM2, storing different iterations
    for Iteration_Identifier, (time_list, iter_conc_list, rate_list) in enumerate(zip(listtime2, listM4, rr_list2)):
        #print(f"ratelist length {len(rate_list)}")
        #print(f"listM2 length {len(iter_conc_list)}")
        for time_step, conc_list, listR in zip(time_list[:-1], iter_conc_list[:-1],rate_list):
            listM7 = [f"Iteration {Iteration_Identifier}"]+ [time_step] + conc_list.tolist() + listR.tolist()
            #print(f"listM7: {listM7}")
            #print(f"conc_list: {conc_list.tolist()}")
            #print(f"listR: {len(listR)}")
            listM6.append(listM7)
    return listM6

###### No multiprocessing, original, end

## the original function
def saveGilData_2(list, data_dir, numberofiteration, end_time, model_name, opt):
    # Name output file
    date = datetime.now().date()
    if opt.ofm == "csv":
        data_file = f"{data_dir}/{model_name}_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
        data_file = get_unique_filename(data_file)
        print(data_file)
        # Open the CSV file in write mode with the specified directory and file name
        with open(data_file, 'w') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec'])
            csv_writer.writerows(list) #how different it is to use .writerow and .writerows
    #elif opt.ofm == "pcl":
    #    headers = ['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB', 'R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec']
    #    data_df = pd.DataFrame(list, columns=headers)
    #    #print(data_df.shape)
    #    data_df.columns = headers
    #    #print(data_df)
    #    data_file = f"{data_dir}/Exp3_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.pcl"
    #    data_file = get_unique_filename(data_file)
    #    print(data_file)
    #    saveData(data_df, data_file)
    print(f" Gillespi Simulation Output Saved as {opt.ofm}")
    return data_file


def saveGilData_replace(list, data_dir, numberofiteration, end_time, model_name,opt):
    # Name output file
    date = datetime.now().date()
    if bool(opt.hs2) == True:
        data_file = f"{data_dir}/{model_name}_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_hs2-{opt.hs2}_HSduration{opt.hsd}.csv"
    else:
        data_file = f"{data_dir}/{model_name}_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
    data_file = get_unique_filename(data_file)
    print(data_file)
    # Open the CSV file in write mode with the specified directory and file name
    with open(data_file, 'w') as csvfile:
        # Create a CSV writer object
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
        csv_writer.writerows(list) #how different it is to use .writerow and .writerows


def saveGilData(list, data_dir, numberofiteration, end_time, opt):
    # Name output file
    date = datetime.now().date()

    if opt.ofm == "csv":
        data_file = f"{data_dir}/Exp3_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_decay8{opt.dmh}.csv"
        data_file = get_unique_filename(data_file)
        print(data_file)
        # Open the CSV file in write mode with the specified directory and file name
        with open(data_file, 'w') as csvfile:
            # Create a CSV writer object
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFA2','HSFB'])
            csv_writer.writerows(list) #how different it is to use .writerow and .writerows
    elif opt.ofm == "pcl":
        headers = ['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFA2','HSFB']
        data_df = pd.DataFrame(list, columns=headers)
        #print(data_df.shape)
        data_df.columns = headers
        #print(data_df)
        data_file = f"{data_dir}/Exp3_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_decay8-{opt.dmh}.pcl"
        data_file = get_unique_filename(data_file)
        print(data_file)
        saveData(data_df, data_file)
    print(f" Gillespi Simulation Output Saved as {opt.ofm}")
    return data_file


def saveParam(param_dict, data_dir, numberofiteration, end_time, model_name, opt):
    date = datetime.now().date()
    if bool(opt.hs2) == True:
        param_name = f"{data_dir}/{model_name}_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSstart2{opt.hs2}_HSduration{opt.hsd}.csv"
    else:
        param_name = f"{data_dir}/{model_name}_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
    param_outfile = get_unique_filename(param_name)
    header = param_dict.keys()
    with open(param_outfile, 'w', newline='') as csvfile_2:
        # Create a CSV writer object
        writer = csv.DictWriter(csvfile_2, fieldnames=header)
        # Write the header
        writer.writeheader()
        # Write the parameter values
        writer.writerow(param_dict)
    #if opt.ofm =="pcl":
    #    param_name = f"{data_dir}/Exp3_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.pcl"
    #    param_outfile = get_unique_filename(param_name)
    #    saveData(param_dict, param_outfile)
    print(f" Parameters Saved as {opt.ofm}")
    return param_outfile


def saveData_oneSAstep(listM6, param_rootdir, numberofiteration, end_time, opt):
    date = datetime.now().date()
    data_file = f"{param_rootdir}/{opt.ids}_SimuData_{date}_{opt.mdn}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
    data_file = get_unique_filename(data_file)
    data_name = f"{date}_{opt.mdn}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
    with open(data_file, 'w') as csvfile:
        csv_writer = csv.writer(csvfile)
        if opt.mdn == 'replaceA1':
            csv_writer.writerow(['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec','MMP_replace_A1HSPR','A1_replace_MMPHSPR'])
        else: 
            csv_writer.writerow(['Iteration_Identifier', 'time','HSFA1','HSPR','C_HSFA1_HSPR','MMP', 'FMP', 'C_HSPR_MMP','HSFB','R_HSFA1_inc','R_HSFA1_dec', 'R_HSPR_inc', 'R_HSPR_dec', 'R_C_HSFA1_HSPR_inc', 'R_C_HSFA1_HSPR_dec1','R_C_HSFA1_HSPR_dec2','R_MMP_inc','R_MMP_dec','R_FMP_inc', 'R_FMP_dec','R_C_HSPR_MMP_inc','R_C_HSPR_MMP_dec1', 'R_C_HSPR_MMP_dec2', 'R_C_HSPR_MMP_dec3','R_HSFB_inc','R_HSFB_dec'])
        csv_writer.writerows(listM6) 



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

