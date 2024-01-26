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

--optSteps,-ops
    how many steps the simulated annealing algorithm is gonna take (default:10)

--timeStep,-tsp
    The time duration for each Gillespi simulation run/iteration (default:10000)

--heatShockStart,-hss
    The time point at which a heat shock is introduced (default:3000)

--heatShockDuration,-hsd
    The duration of heat shock introduced (default: 2500)

--misfoldRateNormal,-mfn
    The formation rate of misfolded protein from folded protein under normal temperature (default: 0.01)

--misfoldRateHS,-mfh
    The formation rate of misfolded protein from folded protein under heat shock condition (default: 0.05)

--decayMMP_HSPR,-dmh
    The decay rate 8 of MMP-HSPR complex (default: 0.01)

--assoMMP_HSPR,-amh
    The association rate between MMP and HSPR, c3 (default: 0.5)

--hillCoeff,-hco
    The Hill Coefficient (default: 1)

--A2positiveAutoReg,-a2p
    Whether HSFA2 positively regulates itself in the model (default: 0)

--leakage,-lkg
    Trancription leakage (default: 0.01)

--hilHalfSaturation,-hhs
    The conc of inducer/repressor at half-max transcriptional rate (default: 1.0)

--maxA1,-ma1
    a1, the max transcription rate of A1 (default: 10.0)

--decayA1,-da1
    decay1, decay rate of free A1 (default: 0.01)

--foldedProduction,-fpp
    a7, folded protein production rate (default: 10)

--foldedDecay,-fpd
    decay6, decay rate of folded protein (default: 0.01)

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

    param_dir, plot_dir = dir_gen()
    param_dict = param_spec(opt)
    #print(param_dict)

    S_record, param_record = simuAnneal(param_dict, opt)

    save_param(S_record, param_record, opt, param_dir)


######################################################
## 1. Specify directory and initial params
######################################################

def dir_gen():
    cwd = os.getcwd() #GRN_heatshock_Arabidopsis
    partiii_dir = os.path.dirname(cwd)

    param_dir = os.path.join(partiii_dir,"Param_Optimisation")
    if not os.path.isdir(param_dir): os.makedirs(param_dir, 0o777)
    plot_dir = os.path.join(partiii_dir,"Gillespi_plots")
    if not os.path.isdir(plot_dir): os.makedirs(plot_dir, 0o777)
    return param_dir, plot_dir

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
        'a1': float(opt.ma1),
        'a2': 100.0,
        'a5': 5.0,
        'a6': 0.2, # refolding rate from MMP-HSPR
        'a7': int(opt.fpp), #folded protein production rate
        ## Ka in Hill equation
        'h1': float(opt.hhs),
        'h2': float(opt.hhs),
        'h5': float(opt.hhs),
        ## association rates
        'c1': 10.0, #between A1 and HSPR
        'c3': float(opt.amh), #between MMP and HSPR
        ## decay rates
        'd1': 0.1, # decay path 1 of A1-HSPR
        'd3': 0.01, # dissociation rate of MMP-HSPR
        'd4_heat': float(opt.mfh),
        'd4_norm': float(opt.mfn),
        'Decay1': 0.01,
        'Decay2': 0.01, # decay of free HSPR
        'Decay4': 0.01,
        'Decay6': float(opt.fpd),
        'Decay7': 0.01, # decay path 2 of A1-HSPR
        'Decay8': float(opt.dmh), # decay of MMP-HSPR. Make sense for it to be higher than normal complexes/proteins
        'Decay5': 0.1,
        ####
        'leakage': float(opt.lkg), #default = 0.01
        'hillcoeff': int(opt.hco), #default = 1
        'hstart':int(opt.hss),
        'hduration':int(opt.hsd)
    }
    #print(param_dict['a1'])  # prints the value of a1
    #print(param_dict['init_HSFA1'])  # prints the value of init_HSFA1
    R_sum = 0
    list = ['a1','a2','a5','a6','a7','h1','h2','h5','c1','c3','d1','d3','Decay1','Decay2','Decay4','Decay5','Decay6','Decay7','Decay8','leakage','hillcoeff']
    for it in list:
        R_sum += param_dict[it]
    print(f"R_sum: {R_sum}")
    return param_dict

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

