# -*- coding: utf-8 -*-
"""
Created on Wed Oct 19 12:14:57 2022

@author: Jake Barrett

Modified by: DMAS Group 7
"""

import warnings

# library installs
import os
import numpy as np
import pandas as pd
import random
from random import seed
from tqdm import tqdm
seed(1)
import matplotlib.pyplot as plt

from openness_simulations_helper_functions import *

pd.options.mode.chained_assignment = None  # default='warn'

warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

#random.Random(3)

# SIMULATION STEPS

# 1 initialise parameters


# allocation paradigm - fixed, random, opt

# parameters for rules - O, G, p
O = np.round(N_STRONG_NODES_J/2) # group sway

# number of rounds
T = 10

# initial opinions - currently random, adjusting this is a big to-do

opinions_0 = np.repeat(range(10),np.repeat(10,10),axis=0)
random.Random(3).shuffle(opinions_0)
opinions_base = pd.DataFrame(opinions_0,columns=["opinions_0"])

# we want a distribution where: some individuals can have value already above the threshold, but most 
# participants are closed-minded
np.random.seed(0)

prop_opm = 0.1

# 2 create dataset from parameters (using code from IP - add in randomised OPM attribute)
    # 2a read in optimised protocol

opt_protocol = pd.read_csv('100_20_200_protocol.csv')
# Data generated through table allocation code - single demographic (a1 or a2), with proportions 0.2 and 0.8

# 3 create update rules: function taking a given table grouping and updating status

opinions_update = True
exp_included = True

extreme_index = random.sample(range(N_STRONG_NODES_I),20)

# Experts: random, central, extreme, switching between extremes periodically, none
n_iterations = 10
extreme_index = random.sample(range(N_STRONG_NODES_I),20)
experts_random = np.random.choice(np.arange(0,10),size=200)
experts_extreme = np.repeat(8,200)
experts_periodic = np.tile(np.concatenate((np.repeat(1,5),np.repeat(8,5))),20)

print("Accounting for randomisation in expert trials")
# need to account for randomisation in CAS trials
for i in range(n_iterations): # a row for each iteration
    if i == 0:
        experts_random_update = [list(experts_random)]
        experts_extreme_update = [list(experts_extreme)]
        experts_periodic_update = [list(experts_periodic)]
    else:
        experts_random_update.append(list(experts_random))
        experts_extreme_update.append(list(experts_extreme))
        experts_periodic_update.append(list(experts_periodic))
      
# Participants: random opinions, bipartisan opinions,commited extremists
opinions_bimodal = opinions_base.copy()
opinions_bimodal['opinions_0'] = np.array(random.choices([0,1,2,7,8,9],k=N_STRONG_NODES_I))
opinions_extreme = opinions_base.copy()
opinions_extreme.loc[extreme_index[0:10]]=0
opinions_extreme.loc[extreme_index[11:20]]=9

# initialise with sensible parameters
input_data = opt_protocol
prop_opm=0.1
T=200
rho=0.01 # Updated value to match paper
demog_cols = ['a']
# variable: prop_opm=0.1 (OPM True) or 1 (OPM False),
# variable: opinions_base=opinions_base, opinions_bimodal, opinions_extreme
allocation='opt'
pt=0.001
O=7
C=2 # two demographic levels in this example
opinions_update=True
exp_included = True
# variable: exp_weight = 0.1 (x_x_low_x_x) or 0.25 (x_x_high_x_x)
# variable: movement_speed = 0.1 (x_x_low_x_x) or 0.25 (x_x_high_x_x)
# variable: OPM_to_CM=True (x_x_x_x_true) or False (x_x_x_x_false)
O2=7
# variable: extremists = True (x_x_x_commit_x) or False otherwise
# variable: extreme_index = range(20) (x_x_x_commit_x) or 0 otherwise
# naming convention: opinion model_experts_exp_weight_initial opinions_opm used

paradigm_values = {
    'dg': "DeGroot",
    'wm': "weighted median",
    #'bc': "bounded confidence"
}

exp_values = {
    'random': experts_random_update,
#    'extreme': experts_extreme_update,
    'periodic': experts_periodic_update
}

participant_values = {
    'random': [opinions_base, False, 0],
    'bipart': [opinions_bimodal, False, 0],
#    'extreme': [opinions_extreme, True, extreme_index]
}

# This is atrocious
speed_values = {
    'slower': [
        0.075, 
        0.075
    ]
}

opm_values = {
    'opm': True,
    'nopm': False
}

trial = 0
for mode in paradigm_values:
    for exp in exp_values:
        for participant in participant_values:
            for speed in speed_values:
                for opm in opm_values:
                    string = str(mode)+"_"+str(exp)+"_"+str(participant)+"_"+str(speed)+"_"+str(opm)
                    print("test for trial " + str(trial) + ": " + str(string))
                    output = avg_over_trials(input_data,T,prop_opm,
                                             participant_values[participant][0],
                                             allocation,demog_cols,rho,C,O,n_iterations,opinions_update,exp_included,
                                             exp_values[exp],speed_values[speed][0],speed_values[speed][1],paradigm_values[mode],
                                             opm_values[opm],
                                             pt,O2,
                                             participant_values[participant][1],participant_values[participant][2])
                    
                    # Plot the evolution of opinions
                    # I'm skipping this since a error appears to occur with different hyperparameters
                    # double_plot(output,T,'opt',n_iterations,True,True,0, save_path=f"output/{string}")
                    
                    # Get statistics on the final opinions
                    values = opinion_analysis(output,True,np.array(exp_values[exp][0]),n_iterations,participant_values[participant][2])
                    if trial == 0:
                        final_data = {string:values}
                    else:
                        final_data[string]=values
                    trial += 1

output_2 = pd.DataFrame(final_data).transpose().reset_index()

# Convert to csv
output_2.to_csv('output/summary.csv')