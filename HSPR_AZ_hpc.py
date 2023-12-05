#!/usr/bin/env python3
# script: HSPR_AZ_tidy.py
# author: Rituparna Goswami, Enrico Sandro Colizzi, Alexis Jiayi Zeng
script_usage="""
usage
    HSPR_AZ_v3.py -nit <numberInteration> -tsp <timeStep> [options]

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

--numberInteration,-nit
    The number of interations for Gillespi simulation

--timeStep,-tsp
    The time duration for each Gillespi simulation run/iteration (default:20000)

--heatShockStart,-hss
    The time point at which a heat shock is introduced (default:10000)

--heatShockDuration,-hsd
    The duration of heat shock introduced (default: 5000)

--misfoldRateNormal,-mfn
    The formation rate of misfolded protein from folded protein under normal temperature (default: 0.01)

--misfoldRateHS,-mfh
    The formation rate of misfolded protein from folded protein under heat shock condition (default: 0.05)

--decayMMP_HSPR,-dmh
    The decay rate of MMP-HSPR complex (default: 0.01)

--assoMMP_HSPR,-amh
    The association rate between MMP and HSPR (default: 0.5)

--hillCoeff,-hco
    The Hill Coefficient (default: 1)

--A2positiveAutoReg,-a2p
    Whether HSFA2 positively regulates itself in the model (default: 0)

--leakage,-lkg
    Trancription leakage (default: 0.01)

--outputFormat,-ofm
    Whether to save Gillespi simulation output as csv or picklefile (default: csv)

--samplingFreq,-spf
    How often is simulation datasaved (default: 1)

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
    print("Step1: Specify output directory")
    data_dir, plot_dir = dir_gen()

    print("Step2: Specify parameters")
    param_dict = param_spec(opt)
    
    print("Step3: Simulation begins")
    listM4, listtime2, numberofiteration, end_time = gillespie_save_1_tsp(param_dict, opt)
    #listM4, listtime2, numberofiteration, end_time = gillespie_store(param_dict, opt, data_dir)
    print("Step4: Combine and save data")
    listM6 = combine_data(listtime2, listM4, opt)
    data_file = saveGilData_2(listM6, data_dir, numberofiteration, end_time, opt)
    param_outfile = saveParam(param_dict, data_dir, numberofiteration, end_time, opt)




##################################################################
## To Extract From Existing File
##################################################################

def extract_para_from_name(filename, opt):
    data_file = str(filename)
    print(f" {data_file}")
    # Define a pattern to match the relevant parts of the filename
    pattern = re.compile(r"Exp3_SimuData_\d+-\d+-\d+_numIter(\d+)_Time([\d.]+)_HSstart(\d+)_HSduration(\d+)\.(pcl|csv)")
    #pattern = re.compile(r"Exp3_SimuData_\d+-\d+-\d+_numIter(\d+)_Time([\d.]+)_HSstart(\d+)_HSduration(\d+)(?:_decay8-([\d.]+))?\.(pcl|csv)")
    # Use the pattern to extract values
    match = pattern.match(data_file)
    if match:
        numberofiteration = int(match.group(1))
        end_time = float(match.group(2))
        opt.hss = int(match.group(3))
        opt.hsd = int(match.group(4))
        #opt.dmh = float(match.group(5))
        opt.ofm = str(match.group(5))
        print(" Extracted parameters")
        print("     Number of Iteration:", numberofiteration)
        print("     End Time:", end_time)
        print("     HSstart:", opt.hss)
        print("     HSduration:", opt.hsd)
        #print("     decay8:", opt.dmh)
        print("     File Extension:", opt.ofm)
    else:
        print("Filename does not match the expected pattern.")
    return data_file, numberofiteration, end_time, opt



#######################################################################
## 1. Parameter specification
#######################################################################
def param_spec(opt):
    param_dict = {
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
        'h6': 1,
        ## association rates
        'c1': 10.0,
        'c3': float(opt.amh), #between MMP and HSPR
        ## decay rates
        'd1': 0.1, # decay path 1 of A1-HSPR
        'd3': 0.01, # dissociation rate of MMP-HSPR
        'd4_heat': float(opt.mfh),
        'd4_norm': float(opt.mfn),
        'Decay1': 0.01,
        'Decay2': 0.01, # decay of free HSPR
        'Decay3': 0.01,
        'Decay4': 0.01,
        'Decay6': 0.01,
        'Decay7': 0.01, # decay path 2 of A1-HSPR
        'Decay8': float(opt.dmh), # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage': float(opt.lkg),
        'numberofiteration': int(opt.nit),
        'hillcoeff': int(opt.hco),
        'hstart':int(opt.hss),
        'hdura':int(opt.hsd)
    }
    #print(param_dict['a1'])  # prints the value of a1
    #print(param_dict['init_HSFA1'])  # prints the value of init_HSFA1
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
    return data_dir, plot_dir

############################################################################
## 3. Gillespi Simulation
############################################################################

def gillespie_store(param_dict, opt, data_dir):
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
            R_C_HSPR_MMP_dec1=d3*C_HSPR_MMP # complex dissociation
            R_C_HSPR_MMP_dec2=a6*C_HSPR_MMP # refolding step
            R_C_HSPR_MMP_dec3=Decay8*C_HSPR_MMP # complex degraded
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
        saveGilData_2(combine_data(listtime2, listM4, opt), data_dir, i, end_time, opt)
        print(f"iteration {i} saved")
    return listM4, listtime2, numberofiteration, end_time

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

def gillespie_save_1_tsp(param_dict, opt):
    listM4=[]
    listtime2=[]
    numberofiteration = param_dict["numberofiteration"]
    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
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
            n = param_dict['hillcoeff']
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
            else: d4 = param_dict['d4_norm']
                
            #HSFa1 andHSFA2 may makes complex
            #increase in HSFA1 by transcription and dessociation from the complex C_HSFA1_HSPR
            R_HSFA1_inc=leakage+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n)+ a3*HSFA2**n/(h3**n+HSFA2**n+HSFB**n) # + d1*C_HSFA1_HSPR
            #decrease in HSFA1 by association to the 1st complex C_HSFA1_HSPR and decay in the protein
            R_HSFA1_dec= Decay1*HSFA1
            #increase in HSPR by transcription and dess
            R_HSPR_inc= leakage+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n)+ a3*HSFA2**n/(h3**n+HSFA2**n+HSFB**n)
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
            if bool(opt.a2p) == False:
                #increase in HSFA2 by transcription with TF HSFA1 
                R_HSFA2_inc=leakage+a4*HSFA1**n/(h4**n+HSFA1**n+HSFB**n) # + a8*HSFA2/(h6+HSFA2+HSFB)
            else:
                #increase in HSFA2 by transcription with TF HSFA1 and HSFA2 itself
                R_HSFA2_inc=leakage+a4*HSFA1**n/(h4**n+HSFA1**n+HSFB**n) + a8*HSFA2**n/(h6**n+HSFA2**n+HSFB**n)
            #decrease in HSFA2 by transcription and dess
            R_HSFA2_dec=Decay3*HSFA2
            #increase in HSFB by transcription with TF HSFA1 and HSFB
            R_HSFB_inc=leakage+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
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
            last_time = Time
            Time+=Tau # the new time the time before the step +the time to happen the next step ()
            counter += 1
            # print (Time,listM)
            if int(Time) == int(last_time) + opt.spf:
                listtime.append(Time) #this is to add stuff to the list
                listM2.append(listM)
        listM4.append(listM2)
        listtime2.append(listtime)
        end_time = Time
        param_dict['end_time'] = end_time
    return listM4, listtime2, numberofiteration, end_time

def gillespie_woA2(param_dict, opt):
    listM4=[]
    listtime2=[]
    numberofiteration = param_dict["numberofiteration"]
    for i in range(numberofiteration):    
        print(f" \n iteration: {i}")
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
            n = param_dict['hillcoeff']
            HSFA1, HSPR, C_HSFA1_HSPR, MMP, FMP, C_HSPR_MMP, HSFA2, HSFB = listM

            if Time >= int(opt.hss) and Time <= int(opt.hss) + int(opt.hsd): d4 = param_dict['d4_heat']
            else: d4 = param_dict['d4_norm']
                
            #HSFa1 andHSFA2 may makes complex
            #increase in HSFA1 by transcription and dessociation from the complex C_HSFA1_HSPR
            R_HSFA1_inc=leakage+a1*HSFA1**n/(h1**n+HSFA1**n+HSFB**n)+ a3*HSFA2**n/(h3**n+HSFA2**n+HSFB**n) # + d1*C_HSFA1_HSPR
            #decrease in HSFA1 by association to the 1st complex C_HSFA1_HSPR and decay in the protein
            R_HSFA1_dec= Decay1*HSFA1
            #increase in HSPR by transcription and dess
            R_HSPR_inc= leakage+a2*HSFA1**n/(h2**n+HSFA1**n+HSFB**n)+ a3*HSFA2**n/(h3**n+HSFA2**n+HSFB**n)
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
            if bool(opt.a2p) == False:
                #increase in HSFA2 by transcription with TF HSFA1 
                R_HSFA2_inc=leakage+a4*HSFA1**n/(h4**n+HSFA1**n+HSFB**n) # + a8*HSFA2/(h6+HSFA2+HSFB)
            else:
                #increase in HSFA2 by transcription with TF HSFA1 and HSFA2 itself
                R_HSFA2_inc=leakage+a4*HSFA1**n/(h4**n+HSFA1**n+HSFB**n) + a8*HSFA2**n/(h6**n+HSFA2**n+HSFB**n)
            #decrease in HSFA2 by transcription and dess
            R_HSFA2_dec=Decay3*HSFA2
            #increase in HSFB by transcription with TF HSFA1 and HSFB
            R_HSFB_inc=leakage+a5*HSFA1**n/(h5**n+HSFA1**n+HSFB**n)
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
            last_time = Time
            Time+=Tau # the new time the time before the step +the time to happen the next step ()
            counter += 1
            # print (Time,listM)
            if int(Time) == int(last_time) + opt.spf:
                listtime.append(Time) #this is to add stuff to the list
                listM2.append(listM)
        listM4.append(listM2)
        listtime2.append(listtime)
        end_time = Time
        param_dict['end_time'] = end_time
    return listM4, listtime2, numberofiteration, end_time

def combine_data(listtime2, listM4, opt):
    #to combine the data 
    listM6 = []
    listM7 = []
    # listM = list of protein conc at a single time point
    # listM2 = list of listM, storing conc at each time point in a single iteration
    # listM4 =list of listM2, storing different iterations
    for Iteration_Identifier, (time_list, iter_conc_list) in enumerate(zip(listtime2,listM4)):
        for time_step, conc_list in zip(time_list, iter_conc_list):
            listM7 = [f"Iteration {Iteration_Identifier}"]+ [time_step] + conc_list.tolist()
            #print(f"listM7: {listM7}")
            #print(f"conc_list: {conc_list.tolist()}")
            listM6.append(listM7)
    return listM6
    

## the original function
def saveGilData_2(list, data_dir, numberofiteration, end_time, opt):
    # Name output file
    date = datetime.now().date()
    if opt.ofm == "csv":
        data_file = f"{data_dir}/Exp3_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.csv"
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
        data_file = f"{data_dir}/Exp3_SimuData_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.pcl"
        data_file = get_unique_filename(data_file)
        print(data_file)
        saveData(data_df, data_file)
    print(f" Gillespi Simulation Output Saved as {opt.ofm}")
    return data_file


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
    
    if opt.ofm =="pcl":
        param_name = f"{data_dir}/Exp3_Para_{date}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}.pcl"
        param_outfile = get_unique_filename(param_name)
        saveData(param_dict, param_outfile)
    
    print(f" Parameters Saved as {opt.ofm}")

    return param_outfile

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




def plot_outcome(data_file, data_dir, plot_dir, numberofiteration, end_time, opt):
    ########################################
    ### Data tidying
    if opt.ofm == "csv":
        try: data_df = pd.read_csv(f"{data_file}")
        except Exception as e1: 
            try: data_df = pd.read_csv(f"{data_dir}/{data_file}")
            except Exception as e2: 
                print(f"Error 1: {e1} \nError 2: {e2}")
                exit()
    elif opt.ofm == "pcl":
        try: data_df = loadData(f"{data_file}")
        except Exception as e1:
            try: data_df = loadData(f"{data_dir}/{data_file}")
            except Exception as e2: 
                print(f"Error 1: {e1} \nError 2: {e2}")
                exit()
    data_df['totalHSPR'] = data_df['HSPR'] + data_df['C_HSFA1_HSPR'] + data_df['C_HSPR_MMP']
    #print(data_df)
    #print(data_df.shape)
    ### number of rows and columns for all iterations
    Rows = int(math.sqrt(numberofiteration))
    Columns = int(math.ceil(numberofiteration / Rows))
    grouped_data = data_df.groupby('Iteration_Identifier')

    ########################################
    ### Call ploting functions ###

    ### Plot trajectories of all species for all iterations
    plot_allvsTime_separate(data_df, grouped_data, plot_dir, numberofiteration, end_time, Rows, Columns, opt)

    ### Variability Analysis
    #plot_before_during_after_HS(data_df, grouped_data, plot_dir, numberofiteration, end_time, Rows, Columns, opt)

    ### Plot trajectory of total HSPR for all iterations
    #plot_totalHSPRvsTime_subplots(grouped_data, data_df, plot_dir, numberofiteration, end_time, Rows, Columns, opt)

    ### Plot overlayed trajectory of A1 concentrations for all trajectory
    #plot_A1vsTime_asOne(grouped_data, plot_dir, numberofiteration, end_time, opt)


def plot_before_during_after_HS(data_df, grouped_data, plot_dir, numberofiteration, end_time, Rows, Columns, opt):
    ss1_start = 1000
    ss1_end = int(opt.hss)
    ssHS_start = int(opt.hss) + 100
    ssHS_end = int(opt.hss) + int(opt.hsd)
    ss3_start = ssHS_end + 500
    ss3_end = end_time
    print(f"hss:{opt.hss}, hsd: {opt.hsd}")
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



    plot_HSPR_hist(totalHSPR_df_outlist, plot_dir, numberofiteration, end_time, opt)
    plot_CVsq_mean(totalHSPR_df_outlist, plot_dir, numberofiteration, end_time, opt)


def plot_CVsq_mean(totalHSPR_df_outlist, plot_dir, numberofiteration, end_time, opt):
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
        plot_name = f"{plot_dir}/CV-Mean_TotalHSPR_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{plot_dir}/CV-Mean_TotalHSPR_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        print(f"save figure {opt.sfg == True}")

    #plt.show()
    plt.close()



def plot_HSPR_hist(totalHSPR_df_outlist, plot_dir, numberofiteration, end_time, opt):

    print("Plot total HSPR histogram")
    fig = plt.figure(figsize=(12, 8))
    #plt.hist(ss1_df['mean_totalHSPR'], label="before HS", density=True,alpha=0.50, color="blue")
    #plt.hist(ssHS_df['mean_totalHSPR'], label="during HS", density=True, alpha=0.50, color="red")
    #plt.hist(ss3_df['mean_totalHSPR'], label="after HS", density=True, alpha=0.50, color="orange")

    for df in totalHSPR_df_outlist:
        plt.hist(df['mean_totalHSPR'], label="before HS", density=True,alpha=0.50)

    plt.title("Distribution of mean total HSPR")
    plt.xlabel("total HSPR")
    plt.ylabel("Frequency")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.) # Adjust the figure size to accommodate the legend
    plt.subplots_adjust(right=0.8)  # Increase the right margin
    plt.tight_layout()

    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/Hist_TotalHSPR_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{plot_dir}/Hist_TotalHSPR_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        print(f"save figure {opt.sfg == True}")
    #plt.show()
    plt.close()


def plot_allvsTime_separate(data_df, grouped_data, plot_dir, numberofiteration, end_time, Rows, Columns, opt):

    print(" Plot trajectories of all species for all iterations")
    conc_col = data_df.drop(columns = ["time", "Iteration_Identifier"])

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
    fig.suptitle('Plot of all concentrations vs time for all iterations separately')
    plt.tight_layout()


    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/separate_allConc_vs_time_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        plot_name = f"{plot_dir}/separate_allConc_vs_time_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

        print(f" save figure {opt.sfg == True}")

    #plt.show()
    plt.close()




def plot_totalHSPRvsTime_subplots(grouped_data, data_df, plot_dir, numberofiteration, end_time, Rows, Columns, opt):
    print("Plot trajectory of total HSPR for all iterations")
    if numberofiteration == 1:
        # If only one subplot, create a single subplot without flattening
        fig, ax = plt.subplots(figsize=(15, 10))
        ax.plot(data_df['time'], data_df['totalHSPR']) 
        ax.set_xlabel('time')
        ax.set_ylabel('totalHSPR')
        ax.legend(loc="upper right")
        ax.set_title(f"iteration 0")
    else:
        # If more than one subplot, create a subplot grid
        fig, ax = plt.subplots(nrows=numberofiteration, figsize=(15, 10))
        ax = ax.flatten()

        # Iterate through grouped data and plot on each subplot
        for (Iteration_Identifier, group_data), ax in zip(grouped_data, ax):
            ax.plot(group_data['time'], group_data['totalHSPR'], label='{}'.format(Iteration_Identifier))
            ax.set_xlabel('time')
            ax.set_ylabel('totalHSPR')
            ax.set_title(f"{Iteration_Identifier}")
            ax.legend(loc="upper right")

        # Set the title for the entire plot
    fig.suptitle('Plot of time vs total HSPR for all Iterations separately')
    plt.tight_layout()
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/separate_totalHSPR_vs_time_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/separate_totalHSPR_vs_time_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")

    #plt.show()
    plt.close()





def plot_A1vsTime_asOne(grouped_data, plot_dir, numberofiteration, end_time, opt):
    fig, ax1 = plt.subplots(figsize=(15,10))  # Set the figure size 
    for Iteration_Identifier, group_data in grouped_data:
        ax1.plot(group_data['time'], group_data['HSFA1'], label='{}'.format(Iteration_Identifier))
        ax1.set_xlabel('time')
        ax1.legend()
        ax1.set_ylabel('HSFA1')
        ax1.set_title('Plot of HSFA1 vs time for all Iterations')
    plt.tight_layout()
    if bool(opt.sfg) == True:
        plot_name = f"{plot_dir}/allAsOne_{ax1.get_ylabel()}_vs_{ax1.get_xlabel()}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.pdf"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
        plot_name = f"{plot_dir}/allAsOne_{ax1.get_ylabel()}_vs_{ax1.get_xlabel()}_numIter{numberofiteration}_Time{end_time}_HSstart{opt.hss}_HSduration{opt.hsd}_hil-{opt.hco}_a2p-{opt.a2p}_decay8-{opt.dmh}.svg"
        unique_plot_name = get_unique_filename(plot_name)
        plt.savefig(f"{unique_plot_name}")
    #plt.show()
    plt.close()





















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

