############################################################################################################################################################################
##### This file writes out the parameter files necessary for conducting resampling simulations. In particular, it includes the following:
#####  (i) results from original run (posteriors and data). 
#####           It is saved as an array allResults, with each index holding a result dictionary corresponding to each user
#####  (ii) baseline parameters for resampling: what parameters should be used in reward calculation during the parametric bootstrap? 
#####           This will tell you under "posterior", "0TxEffect", "prior", and "0TxEffect_beta_i" which has a list with indices in it. 
#####           The 0TxEffect_beta_i was added to 0 out coef for var index i, according to F_KEYS. This was added for the 0'ing out in interestingness bootstrap!
#####           It is stored as a 2d array where first dimension is on the user index, and the second holds the params in an np array
#####  (iii) residual matrix: the residuals for each user based on the posterior fit. 
#####           It is a matrix of NUSERS x T
#####  Notes: residual matrix and rest are calculated only at available and non nan reward times.
############################################################################################################################################################################

# %%
import pandas as pd
import pickle as pkl
import numpy as np
import rpy2.robjects as robjects
from collections import OrderedDict
import scipy.stats as stats
import scipy.linalg as linalg
import argparse
import os
import copy
import skfda.representation.basis as basis
import itertools
import rpy2.robjects.packages as rpackages
from rpy2.robjects.packages import importr
rpackages.importr('fda')
import random
from main import calculate_posterior_avail, calculate_posterior_maineffect, calculate_posterior_unavail, calculate_posteriors,load_data,load_priors
import math

def rindex(l, value):
    for i in reversed(range(len(l))):
        if l[i]==value:
            return i
    return -1

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))

# %%
PRIOR_DATA_PATH = "./init/bandit-prior.RData"
PRIOR_NEW_DATA_PATH = "./init/bandit-spec-new.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5
T=NDAYS*NTIMES

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)

robjects.r['load'](PRIOR_NEW_DATA_PATH)
banditSpec=robjects.r['bandit.spec'] 
PSED=banditSpec.rx2("p.sed")[0]
W=banditSpec.rx2("weight.est")[0]
GAMMA=banditSpec.rx2("gamma")[0]
etaInit=banditSpec.rx2("eta.init")
alpha0_pmean = np.array(banditSpec.rx2("mu0"))
alpha0_psd = np.array(banditSpec.rx2("Sigma0"))

from dosage_checks import determine_user_state

# %%
def calculate_posterior_l2Reg(alpha_sigma, alpha_mu, beta_sigma, beta_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, day):
    "calc post of l2 model at end of study"
    # Get indices with non nan rewards
    avail_idx = np.logical_and(~np.isnan(reward_matrix), ~np.isnan(reward_matrix))

    R = reward_matrix[avail_idx]
    A = action_matrix[avail_idx].reshape(-1, 1)
    P = prob_matrix[avail_idx].reshape(-1, 1)
    F = fs_matrix[avail_idx]
    G = gs_matrix[avail_idx]

    # Calculate prior mu and sigma
    prior_mu = np.hstack((alpha_mu, beta_mu))
    prior_sigma = linalg.block_diag(alpha_sigma, beta_sigma)

    # If there are no available datapoints, return the prior
    if(len(R) == 0):
        return prior_mu, prior_sigma#beta_mu, beta_sigma

    # Calculate X and Y
    X = np.hstack((G, A * F))
    Y = R

    # Calculate posterior mu and sigma
    post_mu, post_sigma = calculate_posteriors(X, Y, prior_mu, prior_sigma, sigma)
    return post_mu, post_sigma

# %%
def calculate_posterior_avail(alpha_sigma, alpha_mu, beta_sigma, beta_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, prior_sigma):
    '''Calculate the posterior distribution when user is available'''

    # Get indices with non nan rewards, and where availability is 1
    #avail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 1)
    avail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 1)

    R = reward_matrix[avail_idx]
    A = action_matrix[avail_idx].reshape(-1, 1)
    P = prob_matrix[avail_idx].reshape(-1, 1)
    F = fs_matrix[avail_idx]
    G = gs_matrix[avail_idx]

    # Calculate prior mu and sigma
    prior_mu = np.hstack((alpha_mu, beta_mu, beta_mu))
    prior_sigma = linalg.block_diag(alpha_sigma, beta_sigma, beta_sigma)

    # If there are no available datapoints, return the prior
    if(len(R) == 0):
        return prior_mu, prior_sigma#beta_mu, beta_sigma

    # Calculate X and Y
    X = np.hstack((G, P * F, (A - P) * F))
    Y = R

    # Calculate posterior mu and sigma
    post_mu, post_sigma = calculate_posteriors(X, Y, prior_mu, prior_sigma, sigma)
    return post_mu, post_sigma

# %%
def run_algorithm(data,user):
    '''Run the algorithm for each user and each bootstrap'''
    rewards=data[:,5]
    rewards=list(rewards[~np.isnan(rewards)])
    imputeRewardValue=sum(rewards)/len(rewards)

    # Load priors
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu = load_priors()

    # Posterior initialized using priors
    post_alpha0_mu, post_alpha0_sigma = np.copy(alpha0_pmean), np.copy(alpha0_psd)
    post_alpha1_mu, post_alpha1_sigma = np.copy(alpha1_pmean), np.copy(alpha1_psd)
    post_actionCenter_mu, post_actionCenter_sigma = np.copy(prior_mu), np.copy(prior_sigma)    

    # get inverses
    alpha0_sigmaInv=np.linalg.inv(alpha0_psd)
    alpha1_sigmaInv=np.linalg.inv(alpha1_psd)
    post_actionCenter_sigmaInv=np.linalg.inv(prior_sigma)

    # DS to store availability, probabilities, features, actions, posteriors and rewards
    availability_matrix = np.zeros((NDAYS * NTIMES))
    prob_matrix = np.zeros((NDAYS * NTIMES))
    reward_matrix = np.zeros((NDAYS * NTIMES))
    action_matrix = np.zeros((NDAYS * NTIMES))
    fs_matrix = np.zeros((NDAYS * NTIMES, F_LEN))
    gs_matrix = np.zeros((NDAYS * NTIMES, G_LEN))

    # Posterior matrices
    # alpha0
    post_alpha0_mu_matrix = np.zeros((NDAYS * NTIMES, G_LEN))
    post_alpha0_sigma_matrix = np.zeros((NDAYS * NTIMES, G_LEN, G_LEN))

    # alpha1
    post_alpha1_mu_matrix = np.zeros((NDAYS * NTIMES, G_LEN))
    post_alpha1_sigma_matrix = np.zeros((NDAYS * NTIMES, G_LEN , G_LEN))

    # beta/action centered
    post_actionCenter_mu_matrix = np.zeros((NDAYS * NTIMES, 2*F_LEN +G_LEN))
    post_actionCenter_sigma_matrix = np.zeros((NDAYS * NTIMES, 2*F_LEN +G_LEN ,  2*F_LEN +G_LEN ))

    dosage = data[0][6]

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            #availability, fs, gs, reward, prob_fsb, action = determine_user_state(data[ts])
            availability, fs, gs, dosage, reward, prob_fsb, action = determine_user_state(data[ts], dosage, action_matrix[ts-1], useOldDosage=False)
            #not use old dosage means recompute `as intended`, 
            #use old dosage means used the one in the data

            # Save user's availability
            availability_matrix[ts] = availability

            # Save probability, features, action and reward
            prob_matrix[ts] = prob_fsb
            action_matrix[ts] = action

            # Save features and state
            reward_matrix[ts] = reward

            fs_matrix[ts] = fs
            gs_matrix[ts] = gs

            post_alpha0_mu_matrix[ts] = post_alpha0_mu
            post_alpha0_sigma_matrix[ts] = post_alpha0_sigma

            post_alpha1_mu_matrix[ts] = post_alpha1_mu
            post_alpha1_sigma_matrix[ts] = post_alpha1_sigma

            post_actionCenter_mu_matrix[ts] = post_actionCenter_mu
            post_actionCenter_sigma_matrix[ts] = post_actionCenter_sigma

        # Update posteriors at the end of the day
        post_actionCenter_mu, post_actionCenter_sigma = calculate_posterior_avail(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], post_actionCenter_sigmaInv)
    
        post_alpha0_mu, post_alpha0_sigma = calculate_posterior_unavail(alpha0_psd, alpha0_pmean, sigma, availability_matrix[:ts + 1], 
                                                                            reward_matrix[:ts + 1], gs_matrix[:ts + 1], alpha0_sigmaInv)

        post_alpha1_mu, post_alpha0_sigma = calculate_posterior_maineffect(alpha1_psd, alpha1_pmean, sigma, availability_matrix[:ts + 1], 
                                                                                        reward_matrix[:ts + 1], action_matrix[:ts + 1], gs_matrix[:ts + 1], alpha1_sigmaInv)

    #idx = np.where(data[:T,2]==1)[0]
    ndata=rindex(data[:T, 2], 1)+1#np.where(data[:T,1]==0)[0] #first time the user is out
    print('user '+str(user)+ '. last ts '+str(ndata)) # last time that we are available, since ending index is exlcusive
    l2Reg_mu, l2Reg_sigma = calculate_posterior_l2Reg(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ndata], prob_matrix[:ndata], reward_matrix[:ndata], 
                                                                  action_matrix[:ndata], fs_matrix[:ndata], gs_matrix[:ndata], day)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_actionCenter_mu_matrix, "post_beta_sigma": post_actionCenter_sigma_matrix,
            "l2Reg_mu": l2Reg_mu, "l2Reg_sigma": l2Reg_sigma,
            "fs": fs_matrix, "gs": gs_matrix}
    
    # Save results
    return result

def initial_run():
    data = load_data()
    allResults=[]
    for i in range(NUSERS):
        allResults.append(run_algorithm(data[i], i))
    return allResults,data


def get_residual_pairs(results, baseline="Prior"):
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu = load_priors()
    residual_matrix = np.zeros((NUSERS, NDAYS * NTIMES))
    baseline_thetas=[]

    for user in range(NUSERS):
        # we care about l2 fit for bootstrapping
        posterior_user_T=results[user]['l2Reg_mu'] 
        alpha=posterior_user_T[:G_LEN].flatten()
        beta=posterior_user_T[-F_LEN:].flatten()

        #idx = np.where(data[:T,2]==1)[0]
        ndata=rindex(results[user]['availability'][:T], 1)+1#np.where(data[:T,1]==0)[0] #first time the user is out
        last_day=math.floor((ndata-1)/NTIMES) #ndata is the ts index of the last available time, reset the -1, then /NTIMES to get the day of the last available time.

        # calculate residuals
        for day in range(NDAYS): #will only matter for up to last_day^^
            for time in range(NTIMES):
                ts = (day) * 5 + time
                gs=results[user]['gs'][ts]
                fs=results[user]['fs'][ts]
                action=results[user]['action'][ts]
                reward=results[user]['reward'][ts]
                available=results[user]['availability'][ts]
        
                # get estimated reward, using l2 fit.
                estimated_reward = gs @ alpha + action * (fs @ beta)
                if available and not np.isnan(reward):
                        residual_matrix[user, ts]=reward-estimated_reward

        # set the 'baseline' or parameter vector with which we generate environment rewards in bootstrap. In particular, the L2 fit at time T is used.
        baseline_theta=np.zeros(F_LEN+G_LEN)
        # set beta
        if baseline=="Prior":
            baseline_theta[-F_LEN:]=prior_mu[-F_LEN:]
        elif baseline=="ZeroAtAll": # to be explicit
            baseline_theta[-F_LEN:]=np.zeros(prior_mu[-F_LEN:].shape)
        elif baseline=="Posterior":
            baseline_theta[-F_LEN:]=beta
        # 0 out coef at baseline coef
        elif baseline in F_KEYS:
            baseline_theta[-F_LEN:]=beta
            index=F_KEYS.index(baseline)
            baseline_theta[G_LEN+index]=0.0

        # set alpha
        baseline_theta[:G_LEN]=alpha

        baseline_thetas.append(baseline_theta)
    return residual_matrix, baseline_thetas

################################################################################################################

np.random.seed(0)
random.seed(0)

result,data=initial_run()

res_matrix, baseline_prior = get_residual_pairs(result, "Prior")
res_matrix, baseline_Posterior=get_residual_pairs(result, "Posterior")
res_matrix, baseline_0Tx=get_residual_pairs(result, "ZeroAtAll")
baseline_txEffectForInteresting=[]
for i in range(1,5):
    res_matrix, baseline_i=get_residual_pairs(result, F_KEYS[i])
    baseline_txEffectForInteresting.append(baseline_i)

baselines={"prior": baseline_prior, "posterior": baseline_Posterior, "all0TxEffect": baseline_0Tx, "0TxEffect_beta_i": baseline_txEffectForInteresting}

# # write result!
with open('./init/baseline_parameters.pkl', 'wb') as handle:
    pkl.dump(baselines, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
with open('./init/residual_matrix.pkl', 'wb') as handle:
    pkl.dump(res_matrix, handle, protocol=pkl.HIGHEST_PROTOCOL)
        
with open('./init/original_result_91.pkl', 'wb') as handle:
    pkl.dump(result, handle, protocol=pkl.HIGHEST_PROTOCOL)

