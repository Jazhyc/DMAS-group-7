# -*- coding: utf-8 -*-
"""
Created on Thu Nov 10 13:17:53 2022

@author: Jake Barrett
"""
import os
import numpy as np
import pandas as pd
import random
from random import seed
seed(1)
import matplotlib.pyplot as plt
import seaborn as sns
import copy
from tqdm import tqdm
import diptest
from weightedstats import weighted_median

# number of strong nodes - fixed at 20%
N_STRONG_NODES_I = 100
N_STRONG_NODES_J=10

#!pip install unidip


def opm_status(row):
    if row['opm_0']<opm_trigger:
        val = 0
    else: 
        val = 1
    return val

def create_data(input_data,T,prop_opm,opinions_base,allocation,demog_cols,extremists,extreme_index):
    opt_rounds = input_data[[col for col in input_data if col.startswith('allocation')]]
        # 2b create base for fixed and random allocations - retain the same strong nodes
    base_data = input_data[['id']+demog_cols].copy()
        # 2c create table allocations
    # fixed allocations: need to partition once, then replicate - can just take first allocation of optimised
    for round in range(T):
        round_name = "allocation_"+str(round+1)
        if(allocation == "fixed"):
            base_data = pd.concat([base_data,opt_rounds['allocation_1']],axis=1)
            base_data.columns = [*base_data.columns[:-1], round_name]
        if(allocation == "random"):
            # want to shuffle vector of tables from allocation_1 in opt_protocol
            shuffle_data = opt_rounds['allocation_1'].copy()
            random.shuffle(shuffle_data)
            base_data = pd.concat([base_data,shuffle_data],axis=1)
            base_data.columns = [*base_data.columns[:-1], round_name]
        if(allocation == "opt"):
            base_data = pd.concat([base_data,opt_rounds[round_name]],axis=1)
        # 2d attach new variables based on set parameters - initial parameters, initial openness

    #base_data = pd.concat([base_data,opinions_base],axis=1)
    base_data['opinions_0'] = opinions_base
    opm_0 = random.choices([0,1],weights=[1-prop_opm,prop_opm],k=N_STRONG_NODES_I)
    opm_base = pd.DataFrame(opm_0,columns=["opm_0"])
    base_data = pd.concat([base_data,opm_base],axis=1)
    # participants can have status 0 (closed, always been closed), 1 (open), and 2 (closed, having previously been open)
    
    #base_data['opm_status_0'] = base_data.apply(opm_status, axis=1) # CONTINUOUS CASE
    base_data['opm_status_0'] = base_data['opm_0']

    # also want to track when specific rules are being triggered
    base_data['first_O'] = 0
    base_data['first_rho'] = 0
    base_data['first_pt'] = 0
    base_data['first_O2'] = 0
    base_data['OPM_to_CM_trigger'] = 0
    if(extremists==True):
        # set the corresponding index of 'opm_to_cm_trigger' to 1: no change of opm/opinion
        base_data.loc[extreme_index,'OPM_to_CM_trigger'] = 1
    
    return base_data

# data = strict_reg_change, round_no=11,table_no=1,R=0.05,B=5,O=7,opinions_update=True,exp_included=True,exp_opinion=4.5,exp_weight=0.25,movement_speed=0.25

# need a diversity calculator for second OPM rule. For our simple example, demog_cols is simply ['a'], and as the allocation is optimised there
# is identical diversity per table per round

def diverse_calc(table, demog_cols, rho, C):
    # Calculate the proportion of each group
    P_values = table.groupby(demog_cols).size() / len(table)
    
    # Calculate the NGV diversity metric
    NGV = C * (1 - sum(P_values ** 2)) / (C - 1)
    
    # Return the product of rho and NGV
    return rho * NGV


# testing pt rule
#data = sim_trial(opt_protocol,5,prop_opm,opinions_base,'opt',demog_cols,rho,C,O,
#              opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
#              OPM_to_CM,pt,O2,False,0)
#round_no = 1
#table_no = 1

def opm_update(data, round_no, table_no, demog_cols, rho, C, O, opinions_update, 
               exp_included, exp_opinion, exp_weight, movement_speed, mode,
               OPM_to_CM, pt, O2):

    # Define column names
    opm_status_col = f'opm_status_{round_no-1}'
    new_opm_status_col = f'opm_status_{round_no}'
    table_col = f'allocation_{round_no}'
    opinion_col = f'opinions_{round_no-1}'
    new_opinion_col = f'opinions_{round_no}'

    # Pre-slice the data to reduce frequent .loc calls
    selected_columns = ['id', table_col, opm_status_col, opinion_col, 
                        'first_rho', 'first_O', 'first_pt', 'first_O2', 
                        'OPM_to_CM_trigger'] + demog_cols

    table = data.loc[data[table_col] == table_no, selected_columns].copy()

    # Convert to NumPy arrays for faster operations
    opm_status_values = table[opm_status_col].values
    opinion_values = table[opinion_col].values

    # Batch assignment using NumPy arrays
    table[new_opm_status_col] = opm_status_values
    table['prior_open'] = opm_status_values
    table[new_opinion_col] = opinion_values

    # CRITERIA 1: Encounter more than O who are already open
    if OPM_to_CM:
        no_open = table['prior_open'].sum()

        if no_open >= O:
            mask = (table[opm_status_col].values == 0) & (table['OPM_to_CM_trigger'].values == 0)
            table[new_opm_status_col].values[mask] = 1
            table['first_O'].values[table['first_O'].values == 0] = round_no

        # Calculate diversity with demography
        rho_NGV = diverse_calc(table, demog_cols, rho, C)

        # Use np.random.choice for vectorized operation
        table['temp_prob'] = np.random.choice([0, 1], size=table.shape[0], p=[1 - rho_NGV, rho_NGV])

        # First rho update
        mask_first_rho = table['first_rho'].values == 0
        table['first_rho'].values[mask_first_rho] = table['temp_prob'].values[mask_first_rho] * round_no

        # Update OPM status based on temp_prob
        mask_opm_status = (table[opm_status_col].values == 0) & (table['temp_prob'].values == 1) & (table['OPM_to_CM_trigger'].values == 0)
        table[new_opm_status_col].values[mask_opm_status] = 1

    # Criteria 2: Open to Closed transitions
    if OPM_to_CM:
        no_open_to_closed = table['OPM_to_CM_trigger'].sum()

        if no_open_to_closed >= O2:
            # Change from open to closed if O2 conditions met
            mask_open_to_closed = table[opm_status_col].values == 1
            table[new_opm_status_col].values[mask_open_to_closed] = 0
            mask_first_O2 = (table[opm_status_col].values == 1) & (table['first_O2'].values == 0)
            table['first_O2'].values[mask_first_O2] = round_no

        # Calculate OPM time and probabilities using pt
        OPM_triggers = table[['first_O', 'first_rho']].values
        table['OPM_time'] = round_no - np.minimum.reduce(np.where(OPM_triggers > 0, OPM_triggers, round_no), axis=1)

        # Calculate 'pt' and probabilities based on OPM_time
        table['pt'] = 1 - (1 / (1 + pt * table['OPM_time']))
        table['temp_prob'] = np.array([np.random.choice([1, 0], p=[1 - p, p]) for p in table['pt']], dtype=float)

        # Update first_pt for rows meeting the criteria
        mask_first_pt = (table[opm_status_col].values == 1) & (table['first_pt'].values == 0)
        table['first_pt'].values[mask_first_pt] = (1 - table['temp_prob'].values[mask_first_pt]) * round_no

        # Update OPM status based on temp_prob
        mask_new_opm_status = (table[new_opm_status_col].values == 1) & (table[opm_status_col].values == 1)
        table[new_opm_status_col].values[mask_new_opm_status] = table['temp_prob'].values[mask_new_opm_status]

        # Update OPM_to_CM_trigger where applicable
        mask_opm_to_cm_trigger = (table[opm_status_col].values == 1) & (table[new_opm_status_col].values == 0)
        table['OPM_to_CM_trigger'].values[mask_opm_to_cm_trigger] = 1

    # If opinions need to be updated
    if opinions_update:
        table_new_opinions = opinion_update(table, round_no, table_no, exp_included, exp_opinion, exp_weight, movement_speed, mode)
        
        # Instead of merging, use update to minimize merge overhead
        table.update(table_new_opinions)

    # Return relevant columns
    return table[['id', 'first_rho', 'first_O', 'first_pt', 'first_O2', new_opm_status_col, new_opinion_col, 'OPM_to_CM_trigger']]


# table_data = table
# table_data = r1.loc[r1.allocation_1==0]
# table_data = run_data_0.loc[(run_data_0.allocation_58==4)&(run_data_0.iteration==0)], exp_included=False,exp_weight=0.1,movement_speed=0.1



def update_opinions_de_groot(opinions, opm_status, exp_included, exp_opinion, exp_weight, movement_speed):
    n_agents = len(opinions)
    total_opinions = np.sum(opinions)
    avg_excl = (total_opinions - opinions) / (n_agents - 1)

    if exp_included:
        avg_with_exp = exp_weight * exp_opinion + (1 - exp_weight) * avg_excl
        new_opinions = opinions + opm_status * movement_speed * (avg_with_exp - opinions)
    else:
        new_opinions = opinions + opm_status * movement_speed * (avg_excl - opinions)

    return new_opinions


def update_opinions_bounded_confidence(opinions, opm_status, exp_included, exp_opinion, exp_weight, movement_speed, nhood):
    n_agents = len(opinions)
    new_opinions = opinions.copy()

    for i in range(n_agents):
        opinion_diff = np.abs(opinions - opinions[i])
        in_nhood = opinion_diff <= nhood

        nhood_sum = np.sum(opinions[in_nhood])
        nhood_count = np.sum(in_nhood)

        # Exclude self from neighborhood
        nhood_sum -= opinions[i]
        nhood_count -= 1

        if nhood_count > 0:
            avg_excl = nhood_sum / nhood_count
        else:
            avg_excl = opinions[i]

        if exp_included and np.abs(exp_opinion - opinions[i]) <= nhood:
            avg_with_exp = exp_weight * exp_opinion + (1 - exp_weight) * avg_excl
            new_opinions[i] = movement_speed * avg_with_exp + (1 - movement_speed) * opinions[i]
        else:
            new_opinions[i] = movement_speed * avg_excl + (1 - movement_speed) * opinions[i]

    return new_opinions

def update_opinions_weighted_median(opinions, opm_status, exp_included, exp_opinion, exp_weight, movement_speed):
    n_agents = len(opinions)
    new_opinions = opinions.copy()

    if exp_included:
        base_weights = [movement_speed * exp_weight] + n_agents * [movement_speed * (1 - exp_weight) / n_agents-1]
        opinions = [exp_opinion, *opinions]
    else:
        base_weights = n_agents * [movement_speed * (1 - exp_weight) / n_agents - 1]

    for agent_idx in range(n_agents):

        # Opinion remains the same when not open-minded
        if not opm_status[agent_idx]:
            continue

        weights = base_weights.copy()
        weights[agent_idx+1] = 1 - movement_speed
        new_opinions[agent_idx] = weighted_median([exp_opinion, *opinions], weights=weights)
    return new_opinions

def opinion_update(table_data, round_no, table_no, exp_included, exp_opinion, exp_weight, movement_speed, mode, nhood=None):
    """
    Optimized version of the opinion update function.
    
    Parameters:
    -----------
    table_data : pd.DataFrame
        DataFrame containing the agent data. Must include columns for agent IDs, allocation, opm status, and opinions.
    round_no : int
        The current round number.
    table_no : int
        The current table number (not used in this function but kept for consistency).
    exp_included : bool
        Whether expert opinion is included in the opinion update process.
    exp_opinion : float
        The expert's opinion value.
    exp_weight : float
        The weight given to the expert's opinion (if included).
    movement_speed : float
        A parameter that determines how much the agents' opinions move towards the new calculated average.
    mode : str
        The mode of opinion dynamics: either "DeGroot" or "bounded confidence".
    nhood : float, optional
        The neighborhood radius used in the bounded confidence model. Only required if mode is "bounded confidence".
    
    Returns:
    --------
    pd.DataFrame
        A DataFrame with the updated opinions for each agent.
    
    Raises:
    -------
    ValueError
        If the mode is not recognized or if nhood is not provided for bounded confidence mode.
    """
    new_opm_status_col = f'opm_status_{round_no}'
    opinion_col = f'opinions_{round_no - 1}'
    new_opinion_col = f'opinions_{round_no}'

    opinions = table_data[opinion_col].values
    opm_status = table_data[new_opm_status_col].values

    if mode == "DeGroot":
        new_opinions = update_opinions_de_groot(opinions, opm_status, exp_included, exp_opinion, exp_weight, movement_speed)
    elif mode == "bounded confidence":
        if nhood is None:
            raise ValueError("Neighborhood radius (nhood) must be specified for bounded confidence mode.")
        new_opinions = update_opinions_bounded_confidence(opinions, opm_status, exp_included, exp_opinion, exp_weight, movement_speed, nhood)
    elif mode == "weighted median":
        new_opinions = update_opinions_weighted_median(opinions, opm_status, exp_included, exp_opinion, exp_weight,
                                                       movement_speed)
    else:
        raise ValueError(f"Unknown mode: {mode}. Must be 'DeGroot' or 'bounded confidence'.")

    return {'id': table_data['id'], new_opinion_col: new_opinions}


round_no=1
table_no=0
def round_update(data,round_no,demog_cols,rho,C,O,
                 opinions_update,exp_included,exp_opinion,exp_weight,movement_speed,mode,
                 OPM_to_CM,pt,O2):
    #print("generating for round "+str(round_no))
    for table_no in range(N_STRONG_NODES_J):
        #print("table "+str(table_no))
        new_opm = opm_update(data,round_no,table_no,demog_cols,rho,C,O,
                             opinions_update, exp_included, exp_opinion, exp_weight, movement_speed,mode,
                             OPM_to_CM,pt,O2)
        if(table_no == 0):
            new_frame = new_opm
        else:
            new_frame = pd.concat([new_frame,new_opm])
        #print("new table added "+str(table_no))
    # merge new info onto original info
    prior_data = data.drop(['first_rho','first_O','first_pt','first_O2','OPM_to_CM_trigger'], axis=1)
    post_data = pd.merge(prior_data,new_frame,how='left',on='id')
    #print("# OPM_to_CM_transitions in round "+str(round_no)+ " = " +str(sum(post_data.OPM_to_CM_trigger)))
    return post_data




def sim_trial(input_data,T,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,
              opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
              OPM_to_CM,pt,O2,
              extremists,extreme_index):
    # simulate the run T times to get average behaviour
    data_update = create_data(input_data,T,prop_opm,opinions_base,allocation,demog_cols,
                              extremists,extreme_index)    
    
    #print(data_update.columns)
    for t in range(T):
        #print(t)
        
        data_update = round_update(data_update,t+1,demog_cols,rho,C,O,
                                   opinions_update,exp_included,exp_opinion_array[t],exp_weight,movement_speed,mode,
                                   OPM_to_CM,pt,O2)
        
        if(sum(data_update.OPM_to_CM_trigger)==N_STRONG_NODES_I):
            #print(sum(data_update.OPM_to_CM_trigger))
            break
    return(data_update)
    
def avg_over_trials(input_data,T,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,n_iterations,
                    opinions_update,exp_included,exp_opinion_array,exp_weight,movement_speed,mode,
                    OPM_to_CM,pt,O2,
                    extremists,extreme_index):
    
    for i in tqdm(range(n_iterations), desc="Iteration"):
        #print("Iteration: "+str(i))
        
        # Check if expert opinion array is a list of lists
        if type(exp_opinion_array[0]) == list:
            exp_opinion = exp_opinion_array[i]
        else:
            exp_opinion = exp_opinion_array
        
        # Change: Pass whole expert opinion array instead of just one value
        temp_data = sim_trial(input_data,T,prop_opm,opinions_base,allocation,demog_cols,rho,C,O,
                              opinions_update,exp_included,exp_opinion,exp_weight,movement_speed,mode,
                              OPM_to_CM,pt,O2,
                              extremists,extreme_index)
        temp_data['iteration']=i
        if(i==0):
            output = temp_data
        else:
            output = pd.concat([output,temp_data])
    return(output)
    
    
# openness_simulations.py line 104: input_frame = latex_plot_good, T=200, allocation = 'opt', scale_to_complete=True, n_iterations=5, importance = True
# exp_asmyptote = experts_extreme_update[0]

def plot_opm_overview(input_frame, n_iterations, scale_to_complete, T, N_STRONG_NODES_I):
    """
    Plots the overview of OPM (Open-mindedness) status over iterations.

    Parameters:
    - input_frame (pd.DataFrame): The input DataFrame containing OPM status and iteration data.
    - n_iterations (int): The number of iterations.
    - scale_to_complete (bool): Whether to scale the plot to completion.
    - T (int): The total number of iterations.
    - N_STRONG_NODES_I (int): The number of strong nodes.

    Returns:
    - None
    """
    # Add in: if importance, need to only look at when the rules triggered a change
    # 4a: each round, how many opm?
    if n_iterations == 1:
        input_frame['iteration'] = 1

    opm_overview = pd.melt(input_frame[[col for col in input_frame if col.startswith(('opm_status_', 'iteration'))]], id_vars=['iteration'])
    # Replace NaNs in value
    opm_overview = opm_overview.replace(np.nan, 0)
    opm_overview['variable'] = opm_overview.variable.str.replace('opm_status_', '')

    # Remove outliers, e.g., if only one or two are left to become OPM - after last iteration has converged
    opm_min = opm_overview.groupby(['variable', 'iteration']).sum().reset_index()
    opm_min.variable = opm_min.variable.astype(float)
    if len(opm_min.loc[opm_min.value == 0,].groupby('iteration')['variable']) == 0:
        first_0 = T
    else:
        first_0 = max(opm_min.loc[opm_min.value == 0,].groupby('iteration')['variable'].min())  # This will never be 0, as always some start with OPM=1

    opm_grouped = opm_overview.groupby('variable').sum().reset_index()
    opm_grouped.variable = opm_grouped.variable.astype(float)
    late_entrants = sum(opm_grouped.loc[opm_grouped.variable >= first_0, "value"])
    time_to_consensus = max(opm_grouped.loc[opm_grouped.value != 0, "variable"]) + 1
    if time_to_consensus != first_0:
        print("Note: not all individuals were OPM by the time we reached 0 aggregate (" + str(late_entrants) + " individuals remaining CM)")
        time_to_consensus = first_0

    if scale_to_complete:
        opm_grouped = opm_grouped.loc[opm_grouped['variable'] <= time_to_consensus,]
        opm_grouped.variable = 100 * opm_grouped['variable'] / time_to_consensus

    opm_grouped = opm_grouped.sort_values('variable').reset_index()[['variable', 'value']]
    # Standardize based on number of iterations we're averaging over
    opm_grouped.value = opm_grouped['value'] / n_iterations

    # Add error bars - one standard error away
    if n_iterations > 1:
        opm_round_values = opm_overview.groupby(['iteration', 'variable']).sum().reset_index()
        opm_round_values['variable'] = opm_round_values.variable.str.replace('opm_status_', '')
        opm_round_values.variable = opm_round_values.variable.astype(float)
        if scale_to_complete:
            opm_round_values.variable = 100 * opm_round_values['variable'] / time_to_consensus

        opm_errors = pd.merge(opm_round_values, opm_grouped, on='variable')
        opm_errors['num'] = (opm_errors['value_x'] - opm_errors['value_y']) ** 2
        # Standard error for every 'variable' (i.e., round number)
        opm_error_values = opm_errors.groupby('variable').sum().reset_index()[['variable', 'num']]
        opm_error_values['num'] = np.sqrt(opm_error_values['num'] / n_iterations)

    opm_grouped.columns = ['Round', 'Cumulative OPM']
    plt.plot(opm_grouped['Round'], opm_grouped['Cumulative OPM'])
    if n_iterations > 1:
        plt.errorbar(opm_grouped['Round'], opm_grouped['Cumulative OPM'], yerr=opm_error_values['num'])

    plt.ylim(0, N_STRONG_NODES_I)
    plt.ylabel('Cumulative # agents with OPM status 1')
    if scale_to_complete:
        x_lim = 100
    else:
        x_lim = T + 1
    plt.xlim(0, x_lim)
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')

    print("Plot 1 complete")
    plt.show()
    plt.close()
    
    return opm_grouped, time_to_consensus

def plot_opm_influence(input_frame, n_iterations, scale_to_complete, time_to_consensus, N_STRONG_NODES_I, importance):
    """
    Plots the influence of different OPM (Open-mindedness) rules over iterations.

    Parameters:
    - input_frame (pd.DataFrame): The input DataFrame containing OPM status and iteration data.
    - n_iterations (int): The number of iterations.
    - scale_to_complete (bool): Whether to scale the plot to completion.
    - time_to_consensus (int): The time to consensus.
    - N_STRONG_NODES_I (int): The number of strong nodes.
    - importance (bool): Whether to consider the importance of rules.

    Returns:
    - prob_grouped (pd.DataFrame): The grouped DataFrame containing the influence of different rules.
    """
    # 4b: how many because of prob
    if importance:
        # for each individual, only need to keep their lowest first
        prob_inspect = input_frame[['first_O', 'first_rho']]
        # 0 out any entries that are not the lowest - NEED TO INCLUDE pi_0
        prob_inspect['lowest'] = np.where(prob_inspect > 0, prob_inspect, np.inf).min(axis=1)
        prob_inspect.loc[prob_inspect.first_O != prob_inspect.lowest, 'first_O'] = 0
        prob_inspect.loc[prob_inspect.first_rho != prob_inspect.lowest, 'first_rho'] = 0
        prob_bind = pd.concat([input_frame['opm_0'], prob_inspect], axis=1)
        # 0 out those who were initialised OPM
        prob_bind.loc[prob_bind['opm_0'] == 1, 'first_O'] = 0
        prob_bind.loc[prob_bind['opm_0'] == 1, 'first_rho'] = 0
        prob_melt = pd.melt(prob_bind[['first_rho', 'first_O']])
        # OPM to CM
        prob_inspect_2 = input_frame[['first_pt', 'first_O2']]
        # 0 out any entries that are not the lowest - NEED TO INCLUDE pi_0
        prob_inspect_2['lowest'] = np.where(prob_inspect_2 > 0, prob_inspect_2, np.inf).min(axis=1)
        prob_inspect_2.loc[prob_inspect_2.first_pt != prob_inspect_2.lowest, 'first_pt'] = 0
        prob_inspect_2.loc[prob_inspect_2.first_O2 != prob_inspect_2.lowest, 'first_O2'] = 0
        prob_melt_2 = pd.melt(prob_inspect_2[['first_pt', 'first_O2']])
        prob_melt_all = pd.concat([prob_melt, prob_melt_2])
    else:
        prob_melt = pd.melt(input_frame[['first_rho', 'first_O']])
        prob_melt_2 = pd.melt(input_frame[['first_pt', 'first_O2']])
        prob_melt_all = pd.concat([prob_melt, prob_melt_2])

    prob_grouped = prob_melt_all.groupby(['variable', 'value']).size().reset_index()
    prob_grouped = prob_grouped[prob_grouped.value != 0]
    prob_grouped = prob_grouped.pivot(index='value', columns='variable', values=0).reset_index()
    # replace na values
    prob_grouped = prob_grouped.fillna(0)
    # need to factor in for when a variable has no influence
    if 'first_rho' not in prob_grouped.columns:
        prob_grouped['first_rho'] = 0
    if 'first_O' not in prob_grouped.columns:
        prob_grouped['first_O'] = 0
    if 'first_pt' not in prob_grouped.columns:
        prob_grouped['first_pt'] = 0
    if 'first_O2' not in prob_grouped.columns:
        prob_grouped['first_O2'] = 0

    prob_grouped['rho_csum'] = prob_grouped.first_rho.cumsum()
    prob_grouped['O_csum'] = prob_grouped.first_O.cumsum()
    prob_grouped['pt_csum'] = prob_grouped.first_pt.cumsum()
    prob_grouped['O2_csum'] = prob_grouped.first_O2.cumsum()
    prob_grouped = prob_grouped.rename(columns={'value': 'Round'})

    if scale_to_complete:
        prob_grouped.Round = prob_grouped.Round.astype(float)
        prob_grouped.Round = 100 * prob_grouped['Round'] / time_to_consensus

    plt.plot(prob_grouped['Round'], prob_grouped['rho_csum'] / n_iterations, label="rho", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O_csum'] / n_iterations, label="O", color="orange")
    plt.legend()
    plt.ylim(0, N_STRONG_NODES_I)
    if importance:
        plt.ylabel('cumulative # agents becoming OPM due to each rule')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0, 100 if scale_to_complete else time_to_consensus + 1)

    print("Plot 2 complete")
    plt.show()
    plt.close()
    
    return prob_grouped

def plot_simulation_results(prob_grouped, n_iterations, N_STRONG_NODES_I, importance, scale_to_complete, x_lim):
    """
    Plots the simulation results based on the provided data.

    Parameters:
    prob_grouped (DataFrame): DataFrame containing the grouped probabilities.
    n_iterations (int): Number of iterations for normalization.
    N_STRONG_NODES_I (int): Y-axis limit for the plot.
    importance (bool): Flag to determine the ylabel.
    scale_to_complete (bool): Flag to determine the xlabel.
    x_lim (int): X-axis limit for the plot.

    Returns:
    None
    """
    plt.plot(prob_grouped['Round'], prob_grouped['pt_csum']/n_iterations, '--', label="R", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O2_csum']/n_iterations, '--', label="O'", color="orange")
    # plt.plot(prob_grouped['Round'], prob_grouped['B_csum']/n_iterations, '--', label="B", color="blue")
    plt.legend()
    # plt.title('Influence of different CM rules for table allocation setting: '+str(allocation))
    plt.ylim(0, N_STRONG_NODES_I)
    if importance:
        plt.ylabel('cumulative # agents becoming CPM due to each rule')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0, x_lim)
    
    print("Plot 3 complete")
    plt.show()
    plt.close()
    
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_opinion_shift(input_frame, n_iterations, exp_asymptote, scale_to_complete, time_to_consensus, x_lim):
    """
    Plots the opinion shift over time based on the provided data.

    Parameters:
    input_frame (DataFrame): DataFrame containing the input data.
    n_iterations (int): Number of iterations for normalization.
    exp_asymptote (int or list): Expert asymptote values.
    scale_to_complete (bool): Flag to determine the xlabel.
    time_to_consensus (int): Time to consensus for scaling rounds.
    x_lim (int): X-axis limit for the plot.

    Returns:
    None
    """
    # 4c: add in opinion shift - just average opinions? Don't want to truncate by time step
    opinion_overview = pd.melt(input_frame[[col for col in input_frame if col.startswith(('opinions_','iteration','id'))]], id_vars=['iteration','id'])
    # need to replace final NaNs with final opinions for each individual
    final_opinions = opinion_overview.groupby(['iteration','id']).last().reset_index()[['iteration','id','value']]
    final_opinions.columns = ['iteration','id','value_fill']
    opinion_overview = pd.merge(opinion_overview, final_opinions, on=['iteration','id'], how='left')
    opinion_overview.value.fillna(opinion_overview.value_fill, inplace=True)
    opinion_overview = opinion_overview.drop('value_fill', axis=1)
    opinion_overview['variable'] = opinion_overview.variable.str.replace('opinions_', '')
    opinion_grouped = opinion_overview.groupby('variable').mean().reset_index()
    opinion_grouped.variable = opinion_grouped.variable.astype(float)
    
    if n_iterations > 1:
        opinion_round_values = opinion_overview.groupby(['iteration','variable']).mean().reset_index()
        opinion_round_values.variable = opinion_round_values.variable.astype(float)
        opinion_errors = pd.merge(opinion_round_values, opinion_grouped[['variable','value']], on='variable')
        opinion_errors['num'] = (opinion_errors['value_x'] - opinion_errors['value_y'])**2
        # want a standard error for every 'variable' (i.e. round number)
        opinion_error_values = opinion_errors.groupby('variable').sum().reset_index()[['variable','num']]
        opinion_error_values['num'] = np.sqrt(opinion_error_values['num'] / n_iterations)
        
    # Next function requires errors so something needs to be returned
    else:
        opinion_error_values = None
    
    opinion_grouped = opinion_grouped.sort_values('variable').reset_index()[['variable','value']]
    opinion_grouped.columns = ['Round', 'Average opinion']
    
    # add 75% bars
    opinion_75 = pd.melt(input_frame[[col for col in input_frame if col.startswith(('opinions_','iteration','id'))]], id_vars=['iteration','id'])
    opinion_75 = pd.merge(opinion_75, final_opinions, on=['iteration','id'], how='left')
    opinion_75.value.fillna(opinion_75.value_fill, inplace=True)
    opinion_75 = opinion_75.drop('value_fill', axis=1)
    opinion_75['variable'] = opinion_75.variable.str.replace('opinions_', '')
    opinion_75.variable = opinion_75.variable.astype(float)
    # moving to a single standard deviation away
    sd = opinion_75.groupby(['variable','iteration']).agg("var").reset_index()
    sd['value'] = np.sqrt(sd['value'])
    sd = sd.groupby('variable')['value'].mean()
    opinion_grouped['sd'] = sd
    opinion_grouped['lq'] = opinion_grouped['Average opinion'] - opinion_grouped['sd']
    opinion_grouped['uq'] = opinion_grouped['Average opinion'] + opinion_grouped['sd']
    
    if type(exp_asymptote) == int:
        exp_asymptote_copy = [exp_asymptote] * opinion_grouped.shape[0]
    else:
        exp_asymptote_copy = exp_asymptote.copy()
        # Keep on extending expert asymptote until end of data, fill with last value
        if len(exp_asymptote_copy) < opinion_grouped.shape[0]:
            while len(exp_asymptote_copy) != opinion_grouped.shape[0]: 
                exp_asymptote_copy.append(exp_asymptote_copy[-1])
        elif len(exp_asymptote_copy) >= opinion_grouped.shape[0]:
            exp_asymptote_copy = exp_asymptote_copy[:opinion_grouped.shape[0]]
    
    opinion_grouped['expert'] = exp_asymptote_copy[0:opinion_grouped.shape[0]]
    
    if scale_to_complete:
        opinion_grouped.Round = 100 * opinion_grouped['Round'] / time_to_consensus
        
    plt.plot(opinion_grouped['Round'], opinion_grouped['Average opinion'], color='red')
    # need to add 75% error bars
    plt.plot(opinion_grouped['Round'], opinion_grouped['lq'], '--', color='blue')
    plt.plot(opinion_grouped['Round'], opinion_grouped['uq'], '--', color='blue')
    if exp_asymptote_copy != "-":
        plt.plot(opinion_grouped['Round'], opinion_grouped['expert'], linestyle="", marker="o")
    if n_iterations > 1:
        plt.errorbar(opinion_grouped['Round'], opinion_grouped['Average opinion'], yerr=opinion_error_values['num'], color="red")
    plt.ylim(0, 10)
    plt.ylabel('average opinion (with LQ+UQ)')
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('t')
    plt.xlim(0, x_lim)
    
    print("Plot 4 complete")
    plt.show()
    plt.close()
    
    return opinion_error_values


def plot_opm_and_prob_grouped(opm_grouped, prob_grouped, opm_error_values, n_iterations, N_STRONG_NODES_I, scale_to_complete, T, importance):
    """
    Plots the OPM and probability grouped data.

    Parameters:
    opm_grouped (DataFrame): DataFrame containing the grouped OPM data.
    prob_grouped (DataFrame): DataFrame containing the grouped probability data.
    opm_error_values (DataFrame): DataFrame containing the error values for OPM.
    n_iterations (int): Number of iterations for normalization.
    N_STRONG_NODES_I (int): Y-axis limit for the plot.
    scale_to_complete (bool): Flag to determine the xlabel.
    T (int): Total number of rounds.
    importance (bool): Flag to determine the ylabel.

    Returns:
    None
    """
    plt.plot(opm_grouped['Round'], opm_grouped['Cumulative OPM'])
    if n_iterations > 1:
        plt.errorbar(opm_grouped['Round'], opm_grouped['Cumulative OPM'], yerr=opm_error_values['num'], color='orange', label='# agents OPM')
    plt.legend()
    plt.ylim(0, N_STRONG_NODES_I)
    if scale_to_complete:
        x_lim = 100
    else:
        x_lim = T + 1
    plt.xlim(0, x_lim)
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    
    plt.plot(prob_grouped['Round'], prob_grouped['rho_csum'] / n_iterations, label="rho", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O_csum'] / n_iterations, label="O", color="orange")
    plt.legend()
    plt.ylim(0, N_STRONG_NODES_I)
    if importance:
        plt.ylabel('cumulative # agents becoming OPM due to each rule')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0, x_lim)
    
    plt.plot(prob_grouped['Round'], prob_grouped['pt_csum'] / n_iterations, '--', label="p", color="red")
    plt.plot(prob_grouped['Round'], prob_grouped['O2_csum'] / n_iterations, '--', label="O'", color="orange")
    plt.legend()
    plt.ylim(0, N_STRONG_NODES_I)
    if importance:
        plt.ylabel('cumulative # agents')
    else:
        plt.ylabel('cumulative # agents triggering each rule')
    if scale_to_complete:
        plt.xlabel('% of rounds until convergence')
    else:
        plt.xlabel('# rounds')
    plt.xlim(0, x_lim)
    
    print("Plot 5 complete")
    plt.show()

def double_plot(input_frame,T,allocation,n_iterations,scale_to_complete,importance,exp_asymptote):
    
    opm_grouped, time_to_consensus = plot_opm_overview(input_frame, n_iterations, scale_to_complete, T, N_STRONG_NODES_I)
    
    prob_grouped = plot_opm_influence(input_frame, n_iterations, scale_to_complete, T, N_STRONG_NODES_I, importance)
    
    plot_simulation_results(prob_grouped, n_iterations, N_STRONG_NODES_I, importance, scale_to_complete, x_lim=T)
    
    opm_error_values = plot_opinion_shift(input_frame, n_iterations, exp_asymptote, scale_to_complete, time_to_consensus, x_lim=T)
    
    plot_opm_and_prob_grouped(opm_grouped, prob_grouped, opm_error_values, n_iterations, N_STRONG_NODES_I, scale_to_complete, T, importance)
       
    
# data = isolate_fit_opt
    
from scipy.optimize import curve_fit
    
def sigmoid(x, L ,x0, k, b):
    y = L / (1 + np.exp(-k*(x-x0))) + b
    return (y)

# data = curve_fit_data_2
pad_0=False
scale_to_100 =False
method = 'opt'
importance=True
def fit_exp_to_O(data,pad_0,scale_to_100,method,importance):
    #https://rowannicholls.github.io/python/curve_fitting/exponential.html
    if importance:
        # for each individual, only count O if it is their first trigger
        prob_inspect = data[['first_pt','first_O']]
        # 0 out any entries that are not the lowest - NEED TO INCLUDE pi_0
        prob_inspect['lowest']=np.where(prob_inspect>0,prob_inspect,np.inf).min(axis=1)
        prob_inspect.loc[prob_inspect.first_O!=prob_inspect.lowest,'first_O'] = 0
        prob_bind=pd.concat([data[['iteration','trial','opm_0']],prob_inspect['first_O']],axis=1)
        # 0 out those who were initialised OPM
        prob_bind.loc[prob_bind['opm_0']==1,'first_O']=0
        prob_grouped = prob_bind[prob_bind.first_O!=0].groupby(['iteration','trial','first_O']).size().reset_index()
    else:
        prob_grouped = data[data.first_O!=0].groupby(['iteration','trial','first_O']).size().reset_index()
    prob_grouped.columns = ['iteration','trial','Round','ct']
    # add in max for each iteration
    prob_grouped_max_round = prob_grouped.groupby(['iteration','trial']).agg(max).reset_index()[['iteration','trial','Round']]
    prob_grouped_max_round.columns = ['iteration','trial','max_round']
    prob_grouped_min_round = prob_grouped.groupby(['iteration','trial']).agg(min).reset_index()[['iteration','trial','Round']]
    prob_grouped_min_round.columns = ['iteration','trial','min_round']
    prob_grouped = prob_grouped.merge(prob_grouped_max_round,on=['iteration','trial'],how='left')
    prob_grouped = prob_grouped.merge(prob_grouped_min_round,on=['iteration','trial'],how='left')
    prob_grouped = prob_grouped[['iteration','trial','Round','ct','max_round']]
    # pad 0s for each iteration
    if(pad_0):
        # add a row for each value under min_round
        # create dictionary of unique iteration/trial/min_rounds
        for i in range(prob_grouped_min_round.shape[0]):
            # add min_round rows, if min round is not 0
            n_reps = prob_grouped_min_round.min_round[i]
            if n_reps>=1:
                new_frame = pd.DataFrame(index=range(n_reps),columns=['iteration','trial','Round','ct','max_round'])
                new_frame.iteration = prob_grouped_min_round.iteration[i]
                new_frame.trial = prob_grouped_min_round.trial[i]
                new_frame.Round = range(n_reps)
                new_frame.ct = 0
                new_frame.max_round = prob_grouped_max_round.max_round[i]
                if i==0:
                    pad_frame = new_frame
                else:
                    pad_frame = pd.concat([pad_frame,new_frame])
        prob_grouped = pd.concat([prob_grouped,pad_frame]).sort_values(["iteration","trial","Round"])
    prob_grouped['O_csum'] = prob_grouped.groupby(['iteration','trial']).ct.cumsum()
    if scale_to_100==True:
        prob_scale = prob_grouped.groupby(['iteration','trial']).agg(max).reset_index()[['iteration','trial','O_csum']]
        prob_scale.columns=['iteration','trial','agent_scale']
        prob_grouped = prob_grouped.merge(prob_scale,on=['iteration','trial'],how="left")
    else:
        prob_grouped['agent_scale']=100
    prob_grouped['O_csum']=100*prob_grouped.O_csum/prob_grouped.agent_scale
    # need to normalise over 0-1 scale
    prob_grouped.Round = 100*prob_grouped.Round/prob_grouped.max_round
    prob_grouped = prob_grouped[['Round','ct','O_csum']].sort_values(['Round'])
    # fit polynomial curve to data: 
    x = prob_grouped.Round
    y = prob_grouped.O_csum
    p_exp = np.polyfit(x,np.log(y),1)
    a = np.exp(p_exp[1])
    b = p_exp[0]
    x_fitted = x.sort_values()
    y_fitted_exp = a*np.exp(b*x_fitted)
    # fit sigmoid
    p0 = [max(y), np.median(x),1,min(y)] # this is an mandatory initial guess
    popt, pcov = curve_fit(sigmoid, x, y,p0, method='dogbox',maxfev=5000)
    y_fitted_sig = sigmoid(x_fitted, *popt)
    ax = plt.axes()
    ax.scatter(x, y, label='Raw data')
    ax.plot(x_fitted, y_fitted_exp, 'k', label='Fitted exponential curve')
    ax.plot(x_fitted, y_fitted_sig, 'k', label='Fitted sigmoid curve',color='red')
    if(importance):
        ylab="% agents becoming OPM by triggering O rule"
    else:
        ylab="% of agents triggering O rule"
    ax.set_title('Convergence to OPM for various parameter settings: '+str(method))
    ax.set_ylabel(str(ylab))
    ax.set_ylim(0, 100)
    ax.set_xlabel('% of time to convergence')
    ax.legend()



#! Needs verification
def get_intervals(data):
    # Uses diptest to get the lower and upper bounds of the dip test
    dip, results = diptest.dipstat(data, full_output=True, sort_x=True, allow_zero=True)
    lower_indices = results['lo']
    upper_indices = results['hi']
    
    intervals = [lower_indices, upper_indices]
    return intervals



import pandas as pd
import numpy as np

def calculate_R1(bound_opinions):
    """
    Calculate R1: Difference between mean final opinion and mean initial opinion.
    """
    mean_final = bound_opinions.loc[bound_opinions['type'] == "Last"]['value'].mean()
    mean_initial = bound_opinions.loc[bound_opinions['type'] == "First"]['value'].mean()
    return mean_final - mean_initial

def calculate_R2(bound_opinions, expert_input, max_round):
    """
    Calculate R2: Difference between mean initial and final deviation from expert opinion.
    """
    expert_redux = expert_input[0:int(max_round)]
    mean_expert = expert_redux.mean()
    
    R2a = np.mean(abs(bound_opinions.loc[bound_opinions['type'] == "First"]['value'] - mean_expert))
    R2b = np.mean(abs(bound_opinions.loc[bound_opinions['type'] == "Last"]['value'] - mean_expert))
    
    return R2a, R2b, R2a - R2b

def calculate_R3(bound_opinions, n_iterations):
    """
    Calculate R3: Differences in the number of extreme opinions (N8, N9, N0, N1) across iterations.
    """
    N8_0 = bound_opinions.loc[(bound_opinions.type == 'First') & (bound_opinions.value >= 8)].shape[0]
    N8_T = bound_opinions.loc[(bound_opinions.type == 'Last') & (bound_opinions.value >= 8)].shape[0]
    
    N9_0 = bound_opinions.loc[(bound_opinions.type == 'First') & (bound_opinions.value >= 8.5)].shape[0]
    N9_T = bound_opinions.loc[(bound_opinions.type == 'Last') & (bound_opinions.value >= 8.5)].shape[0]
    
    N1_0 = bound_opinions.loc[(bound_opinions.type == 'First') & (bound_opinions.value <= 1)].shape[0]
    N1_T = bound_opinions.loc[(bound_opinions.type == 'Last') & (bound_opinions.value <= 1)].shape[0]
    
    N0_0 = bound_opinions.loc[(bound_opinions.type == 'First') & (bound_opinions.value <= 0.5)].shape[0]
    N0_T = bound_opinions.loc[(bound_opinions.type == 'Last') & (bound_opinions.value <= 0.5)].shape[0]
    
    R3a = (N8_0 - N8_T) / n_iterations
    R3b = (N9_0 - N9_T) / n_iterations
    R3c = (N1_0 - N1_T) / n_iterations
    R3d = (N0_0 - N0_T) / n_iterations
    
    return R3a, R3b, R3c, R3d

def calculate_R4(bound_opinions, extreme_index, n_iterations):
    """
    Calculate R4: Number of unique modes in the final distribution of opinions.
    """
    if extreme_index == 0:
        modal_data = bound_opinions.loc[bound_opinions.type == 'Last']
    else:
        modal_data = bound_opinions.loc[(bound_opinions.type == 'Last') & (~bound_opinions.id.isin(extreme_index))]
    
    data = np.sort(modal_data['value'], axis=0)
    intervals = get_intervals(data)
    
    R4 = len(intervals)
    
    modes = []
    for iteration in range(n_iterations):
        iter_modal_data = np.sort(modal_data.loc[modal_data.iteration == iteration]['value'], axis=0)
        intervals = get_intervals(iter_modal_data)
        modes.append(len(intervals))
    
    individual_mode_count = len([x for x in modes if x == R4])
    
    return R4, individual_mode_count / n_iterations

def calculate_R5(bound_opinions):
    """
    Calculate R5: Variance of the final opinions across iterations.
    """
    R5 = np.mean((bound_opinions.loc[bound_opinions['type'] == "Last"].groupby('iteration')['value'].std().dropna())**2)
    return R5

def calculate_avg_runtime(bound_opinions):
    """
    Calculate the average runtime for all iterations.
    """
    avg_T = bound_opinions.loc[bound_opinions['type'] == "Last"][['variable', 'iteration']].groupby('iteration').max().mean()[0]
    return avg_T

def calculate_avg_flux_times(bound_opinions):
    """
    Calculate average flux start time, end time, and time spent in flux.
    """
    avg_t_opm = bound_opinions.loc[bound_opinions['type'] == "Start"]['variable'].mean()
    avg_t_cm = bound_opinions.loc[bound_opinions['type'] == "Mid"]['variable'].mean()
    avg_flux = avg_t_cm - avg_t_opm
    return avg_t_opm, avg_t_cm, avg_flux

def opinion_analysis(data, experts, expert_input, n_iterations, extreme_index):
    """
    Perform opinion analysis on the provided data and return a dictionary with calculated metrics.
    """
    opinion_overview = pd.melt(data[[col for col in data if col.startswith(('opinions_', 'iteration', 'id'))]], id_vars=['iteration', 'id'])
    opinion_overview['variable'] = opinion_overview.variable.str.replace('opinions_', '')
    opinion_overview.variable = opinion_overview.variable.astype(float)
    
    max_round = max(opinion_overview.variable)
    opinion_overview = opinion_overview[opinion_overview['value'].notna()].sort_values(by='variable')
    
    bound_opinions = pd.concat([opinion_overview.groupby(['iteration', 'id']).head(1), opinion_overview.groupby(['iteration', 'id']).tail(1)]).drop_duplicates().sort_values(['iteration', 'id']).reset_index(drop=True)
    bound_opinions['type'] = np.where(bound_opinions['variable'] == 0, "First", "Last")
    
    R1 = calculate_R1(bound_opinions)
    R2a, R2b, R2 = calculate_R2(bound_opinions, expert_input, max_round)
    R3a, R3b, R3c, R3d = calculate_R3(bound_opinions, n_iterations)
    R4, R4_uniqueness = calculate_R4(bound_opinions, extreme_index, n_iterations)
    R5 = calculate_R5(bound_opinions)
    avg_T = calculate_avg_runtime(bound_opinions)
    avg_t_opm, avg_t_cm, avg_flux = calculate_avg_flux_times(bound_opinions)
    
    return {
        "R1": R1,
        "R2a": R2a,
        "R2b": R2b,
        "R2": R2,
        "R3a (8)": R3a,
        "R3b (9)": R3b,
        "R3c (1)": R3c,
        "R3d (0)": R3d,
        "R4": R4,
        "R4 uniqueness": R4_uniqueness,
        "R5": R5,
        "Average runtime": avg_T,
        "Average t OPM": avg_t_opm,
        "Average t CM": avg_t_cm,
        "Average time in flux": avg_flux
    }





def kde_plot(opinions,title,legend_status):
    long_data = pd.wide_to_long(opinions,stubnames='opinions',i=['iteration','id'],j='round',sep="_").reset_index()[['iteration','id','round','opinions']]
    # only want first and last opinions per id and iterations
    long_data = long_data.dropna()
    long_data_grouped = long_data.groupby(['iteration','id'])
    filter_data = pd.concat([long_data_grouped.head(1),long_data_grouped.tail(1)])
    filter_data['round'].loc[filter_data['round']==0]="First"
    filter_data['round'].loc[filter_data['round']!="First"]="Last"
    for value in ['First','Last']:
        subset = filter_data[filter_data['round']==value]
        sns.distplot(subset['opinions'],hist=False,kde=True,kde_kws={'linewidth':3,'clip': (0.0, 9.0)},label=value)
    if legend_status == True:
        plt.legend(prop={'size': 12}, title = 'Round')
    #plt.title(title)
    plt.xlabel('Opinion')
    plt.ylabel('Density')
    plt.ylim(0,0.5)
    # fix ylim at 0, 0.5
    plt.show()
    plt.close()



