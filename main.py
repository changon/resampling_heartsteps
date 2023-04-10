### Main runner code for running Personalized HeartSteps algorithm (Liao et al, 2020) ###

# %%
import pandas as pd
import pickle5 as pkl 
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

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))

# %%
PKL_DATA_PATH = "./init/all91.pkl"
PRIOR_DATA_PATH = "./init/bandit-prior.RData"
PRIOR_NEW_DATA_PATH = "./init/bandit-spec-new.RData"
NDAYS = 90
NUSERS = 91
NTIMES = 5

LAMBDA = 0.95
MAX_ITERS=100

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)

E0 = 0.2
E1 = 0.2

MIN_DOSAGE = 0
MAX_DOSAGE = 20

NBASIS=50

# %%
# create the dosage basis
dosage_basis = basis.BSpline((MIN_DOSAGE, MAX_DOSAGE), n_basis=NBASIS, order=4)

# create the dosage grid
dosage_grid = np.arange(MIN_DOSAGE, MAX_DOSAGE + .1, .1)
DOSAGE_GRID_LEN = len(dosage_grid)

# evaluate the dosage values using the basis
dosage_eval = dosage_basis.evaluate(dosage_grid)
next_dosage_eval0 = dosage_basis.evaluate(dosage_grid * 0.95).squeeze().T
next_dosage_eval1 = dosage_basis.evaluate(dosage_grid * 0.95 + 1).squeeze().T
dosage_eval = dosage_eval[:, :, 0]

# setup dosage matrix instead of np.repeat in calculating marginal rewards
dosage_matrix = []
for dosage in dosage_grid:
    dosageI = np.repeat(dosage/20.0, NTIMES*3000)
    dosage_matrix.append(dosageI)
dosage_matrix = np.matrix(dosage_matrix)
    
# %%
# partial dosage ols solutions used in eta proxy update (in particular, where value updates are done via function approximation)
dosage_OLS_soln=np.linalg.inv(np.matmul(dosage_eval,dosage_eval.T))#(X'X)^{-1}#50 x 50
dosage_OLS_soln=np.matmul(dosage_OLS_soln, dosage_eval)#(X'X)^{-1}X'# 50 x 201

# %%
# load in prior data results like H1 (eta.init), w, and gamma tuned by peng
robjects.r['load'](PRIOR_NEW_DATA_PATH)
banditSpec=robjects.r['bandit.spec'] 
PSED=banditSpec.rx2("p.sed")[0]
W=banditSpec.rx2("weight.est")[0]
GAMMA=banditSpec.rx2("gamma")[0]
etaInit=banditSpec.rx2("eta.init")

# %%
# Load data
def load_data():
    with open(PKL_DATA_PATH, "rb") as f:
        data = pkl.load(f)
    return data

# %%
# Load initial run result
def load_initial_run(residual_path, baseline_thetas_path, baseline):
    with open(residual_path, "rb") as f:
        residual_matrix = pkl.load(f)
    with open(baseline_thetas_path, "rb") as f:
        baseline_pickle = pkl.load(f)
    if baseline == "Prior":
        baseline_thetas = baseline_pickle["prior"]
        print(baseline)
        print(baseline_thetas[0:5])
    elif baseline == "Posterior":
        baseline_thetas = baseline_pickle["posterior"]
        print(baseline)
        print(baseline_thetas[0:5])
    elif baseline == "Zero":
        baseline_thetas = baseline_pickle["all0TxEffect"]
        print(baseline)
        print(baseline_thetas[0:5])
    elif baseline in F_KEYS:
        index=F_KEYS.index(baseline)#.index in 3.9, .equals in 3.7
        print(baseline)
        baseline_thetas = baseline_pickle["0TxEffect_beta_i"][index]
        print(baseline_thetas[0:5])
    else:
        raise ValueError("Invalid baseline")

    return residual_matrix, baseline_thetas

# %%
def determine_user_state(data, dosage, last_action):
    '''Determine the state of each user at each time point'''
    availability = data[2]

    features = {}

    features["engagement"] = data[7]
    features["other_location"] = data[8]
    # features["work_location"] = data[9]
    features["variation"] = data[10]
    features["temperature"] = data[11]
    features["logpresteps"] = data[12]
    features["sqrt_totalsteps"] = data[13]
    features["prior_anti"] = data[14]

    # calculating dosage
    newdosage = LAMBDA * dosage + (1 if (features["prior_anti"] == 1 or last_action == 1) else 0)

    # standardizing the dosage
    features["dosage"] = newdosage / 20.0

    features["intercept"] = 1

    fs = np.array([features[k] for k in F_KEYS])
    gs = np.array([features[k] for k in G_KEYS])

    reward = data[5]

    return availability, fs, gs, newdosage, reward

# %%
def load_priors():
    '''Load priors from RData file'''
    robjects.r['load'](PRIOR_DATA_PATH)
    priors = robjects.r['bandit.prior']
    alpha0_pmean = np.array(banditSpec.rx2("mu0"))
    alpha0_psd = np.array(banditSpec.rx2("Sigma0"))
    alpha_pmean = np.array(priors.rx2("mu1"))
    alpha_psd = np.array(priors.rx2("Sigma1"))
    beta_pmean = np.array(priors.rx2("mu2"))
    beta_psd = np.array(priors.rx2("Sigma2"))
    sigma = float(priors.rx2("sigma")[0])

    prior_sigma = linalg.block_diag(alpha_psd, beta_psd, beta_psd)
    prior_mu = np.concatenate([alpha_pmean, beta_pmean, beta_pmean])

    return alpha0_pmean, alpha0_psd, alpha_pmean, alpha_psd, beta_pmean, beta_psd, sigma, prior_sigma, prior_mu

# %%
def get_priors_alpha_beta(post_mu, post_sigma):
    '''Get alpha and beta priors from mu and sigma'''
    alpha_pmean = post_mu[:G_LEN].flatten()
    alpha_psd = post_sigma[:G_LEN, :G_LEN]
    beta_pmean = post_mu[-F_LEN:].flatten()
    beta_psd = post_sigma[-F_LEN:, -F_LEN:]

    return alpha_pmean, alpha_psd, beta_pmean, beta_psd

# %%
def sample_lr_params(alpha_pmean, alpha_psd, beta_pmean, beta_psd, sigma):
    '''Sample alpha, beta and noise from priors for BLR'''

    alpha0 = np.random.multivariate_normal(alpha_pmean, alpha_psd)
    alpha1 = np.random.multivariate_normal(beta_pmean, beta_psd)
    beta = np.random.multivariate_normal(beta_pmean, beta_psd)
    et = np.random.normal(0, np.sqrt(sigma**2))

    return alpha0, alpha1, beta, et

# %%
def clip(x, E0=E0, E1=E1):
    '''Clipping function'''
    #return min(1 - E0, max(x, E1))
    return min(1-E1, E0+max(x-.5,0)*(1-E0)/.5)

# %%
def calculate_post_prob(ts, data, fs, beta_pmean, beta_psd, eta = 0):
    '''Calculate the posterior probability of Pr(fs * b > eta)'''

    # First 7 days, use 0.2 or 0.25 as in the data
    if ts < 35:
        #return data[ts][3]
        #if data[ts][3] == .2 or data[ts][3] == .25:
        #    return data[ts][3]
        #else:
        #    return .25
        return .25

    # Calculate the mean of the fs*beta distribution
    fs_beta_mean = fs.T.dot(beta_pmean)

    # Calculate the variance of the fs*beta distribution
    fs_beta_cov = fs.T @ beta_psd @ fs

    # Calculate the probability of Pr(fs * b > eta) using cdf
    post_prob = 1 - stats.norm.cdf(eta, fs_beta_mean, np.sqrt(fs_beta_cov))

    # Clip the probability
    phi_prob = clip(post_prob)

    return phi_prob

# %%
def calculate_reward(ts, fs, gs, action, baseline_theta, residual_matrix):
    '''Calculate the reward for a given action'''

    # Get alpha and betas from the baseline
    alpha0 = baseline_theta[:G_LEN].flatten()
    beta   = baseline_theta[-F_LEN:].flatten()

    # Calculate reward
    estimated_reward = (gs @ alpha0) + action * (fs @ beta) #for dosage as baseline
    reward = residual_matrix[ts] + estimated_reward # this residual matrix will either by the one from original data or a resampled with replacemnet version if user-specific
    #temporary sanity check with no noise!

    return reward

# %%
def calculate_posteriors(X, Y, prior_mu, sigmaInv, sigma):
    '''Calculate the posterior mu and sigma'''

    # Calculate posterior sigma
    post_sigma = (sigma**2) * np.linalg.inv(X.T @ X + (sigma**2) * sigmaInv)

    # Calculate posterior mu
    post_mu = (post_sigma @ ((X.T @ Y)/(sigma**2) + sigmaInv @ prior_mu) )

    return post_mu, post_sigma

# %%
def calculate_posterior_avail(alpha_sigma, alpha_mu, beta_sigma, beta_mu, sigma, availability_matrix, prob_matrix, reward_matrix, action_matrix, fs_matrix, gs_matrix, beta_sigmaInv):
    '''Calculate the posterior distribution when user is available'''

    # Get indices with non nan rewards, and where availability is 1
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
        return prior_mu, prior_sigma

    # Calculate X and Y
    X = np.hstack((G, P * F, (A - P) * F))
    Y = R

    # Calculate posterior mu and sigma
    post_mu, post_sigma = calculate_posteriors(X, Y, prior_mu, beta_sigmaInv, sigma)

    # Get the posterior beta mu and sigma
    #post_beta_mu, post_beta_sigma = post_mu[-F_LEN:], post_sigma[-F_LEN:, -F_LEN:]
    #return post_beta_mu, post_beta_sigma
    return post_mu, post_sigma

# %%
def calculate_posterior_unavail(prior_alpha0_sigma, prior_alpha0_mu, sigma, availability_matrix, reward_matrix, gs_matrix, alpha0_sigmaInv):
    '''Calculate the posterior distribution for the case when there are no available timesloday'''

    # Get the index of unavailable timeslots and non nan rewards
    unavail_idx = np.logical_and(~np.isnan(reward_matrix), availability_matrix == 0)

    # the feature matrix, and reward matrix
    X = gs_matrix[unavail_idx]
    Y = reward_matrix[unavail_idx]

    # If there are no unavailable datapoints, return the prior
    if len(Y) == 0:
        return prior_alpha0_mu, prior_alpha0_sigma

    # Calculate posterior mu and sigma
    post_alpha0_mu, post_alpha0_sigma = calculate_posteriors(X, Y, prior_alpha0_mu, alpha0_sigmaInv, sigma)

    return post_alpha0_mu, post_alpha0_sigma

# %%
def calculate_posterior_maineffect(prior_alpha1_sigma, prior_alpha1_mu, sigma, availability_matrix, reward_matrix, action_matrix, gs_matrix, alpha1_sigmaInv):
    '''Calculate the posterior distribution for the case when user is available but we don't take action (action = 0)'''

    # Get the index of available timeslots with action = 0, and non nan rewards
    maineff_idx = np.logical_and.reduce((availability_matrix == 1, action_matrix == 0, ~np.isnan(reward_matrix)))

    # the feature matrix, and reward matrix
    X = gs_matrix[maineff_idx]
    Y = reward_matrix[maineff_idx]

    # If there are no unavailable datapoints, return the prior
    if len(Y) == 0:
        return prior_alpha1_mu, prior_alpha1_sigma

    # Calculate posterior mu and sigma
    post_alpha1_mu, post_alpha1_sigma = calculate_posteriors(X, Y, prior_alpha1_mu, alpha1_sigmaInv, sigma)

    return post_alpha1_mu, post_alpha1_sigma

# %%
def select_action(p):
    '''Select action from bernoulli distribution with probability p'''
    return stats.bernoulli.rvs(p)

# %%
# later update on the fly with state_prob dict
def get_state_probabilities(fs, gs):
    '''Compute the probability of occurence for each state, given the history until timeslot ts'''

    # Dict to first store occurence counts
    state_prob = {}

    # Remove the dosage from the state
    fm = np.delete(fs, 1, axis=1)
    gm = np.delete(gs, 4, axis=1)

    # Count occurences
    for i in range(len(fs)):
        # Remove the dosage from the state
        key = str(np.concatenate([fm[i], gm[i]]))
        if key in state_prob:
            state_prob[key] += 1
        else:
            state_prob[key] = 1

    # Normalize to get probabilities
    for key in state_prob.keys():
        state_prob[key] /= len(fs)

    # return vector of probabilities to easily multiply in reward calculations later
    pZ = []
    for i in range(len(fs)):
        key = str(np.concatenate([fm[i], gm[i]]))
        pZ.append(state_prob[key])
    
    return np.array(pZ)

# %%
def get_empirical_rewards_estimate(target_availability, target_action, fs_matrix, gs_matrix, pZ, beta_mu, alpha0_mu, alpha1_mu):
    '''Calculate the empirical reward estimate'''
    rewardEstimates=[]

    fs = np.copy(fs_matrix)
    gs = np.copy(gs_matrix)

    # Compute r(x, a) i.e. r_{target_availability}(x, target_action)
    for i in range(DOSAGE_GRID_LEN):
        # modifying feature matrices to have the dosage from the dosage_grid
        fs[:, 1] = dosage_matrix[i,:len(fs)]#np.repeat(dosage_grid[i] / 20.0, len(fs))
        gs[:, 4] = dosage_matrix[i,:len(fs)]#np.repeat(dosage_grid[i] / 20.0, len(fs))
        # using target values instead of action[t] for speedup
        fittedReward = target_availability * (gs @ alpha1_mu * pZ + target_action * (fs @ beta_mu * pZ)) + \
                        (1-target_availability) * (gs @ alpha0_mu * pZ)

        rewardEstimates.append(np.sum(fittedReward))
    return rewardEstimates

# %%
# gets the \sum_{x', i'} \tau(x' \mid x, a)*f_pAvail(i')*V(x',i')
def get_value_summand(dosage_index, action, pavail, theta0, theta1, psed=PSED, lamb=LAMBDA):
    summand = 0
    basis_representation0 = next_dosage_eval0[dosage_index, :]
    basis_representation1 = next_dosage_eval1[dosage_index, :]
    #when action==0
    if action == 0:
        # case: x'=\lambda*dosage+1,i'=1
        V_1_1 = basis_representation1 @ theta1
        summand += (psed)   * pavail     * V_1_1

        # case: x'=\lambda*dosage+1,i'=0. index into V_old with or without offset depending on availability
        V_1_0 = basis_representation1 @ theta0
        summand += (psed)   * (1-pavail) * V_1_0

        # case: x'=\lambda*dosage,i'=1
        V_0_1 = basis_representation0 @ theta1
        summand += (1-psed) * pavail     * V_0_1

        # case: x'=\lambda*dosage,i'=0. index into V_old with or without offset depending on availability
        V_0_0 = basis_representation0 @ theta0
        summand += (1-psed) * (1-pavail) * V_0_0

    # when action == 1
    else:
        # case: x'=\lambda*dosage+1,i'=1
        V_1_1 = basis_representation1 @ theta1
        summand += pavail     * V_1_1

        # case: x'=\lambda*dosage+1,i'=0. index into V_old with or without offset depending on availability
        V_1_0 = basis_representation1 @ theta0
        summand += (1-pavail) * V_1_0 
    return summand

# %%
def bellman_backup(action_matrix, fs_matrix, gs_matrix, post_mu, p_avail_avg, theta0, theta1, reward_available0_action0, reward_available1_action0, reward_available1_action1, gamma=GAMMA):
    # set values based on OLS estimates
    V = [0]*(2*DOSAGE_GRID_LEN)

    # now go through each state and bellman update it
    # each V[i+()] corresponds to formula max_a [r_i(x,a) + value_summand]
    for i in range(DOSAGE_GRID_LEN):

        # bellman update on avail0 case
        r00 = reward_available0_action0[i]
        V[i] = r00 + gamma * get_value_summand(i, 0, p_avail_avg, theta0, theta1)

        # bellman update on avail1 case
        r10 = reward_available1_action0[i]
        v0 = r10 + gamma * get_value_summand(i, 0, p_avail_avg, theta0,theta1)

        # bellman update on avail1 action1 case
        r11 = reward_available1_action1[i]
        v1  = r11 + gamma * get_value_summand(i, 1, p_avail_avg, theta0, theta1)
        v = max(v1, v0)
        V[i + DOSAGE_GRID_LEN] = v

    return np.array(V)

# %%
def calculate_value_functions(availability_matrix, action_matrix, fs_matrix, gs_matrix, reward_matrix, beta_mu, alpha0_mu, alpha1_mu, ts):
    '''Calculate eta for a given dosage'''

    # Remove datapoints with nan rewards
    non_nanidx = ~np.isnan(reward_matrix)
    AV = availability_matrix[non_nanidx]
    A = action_matrix[non_nanidx]
    F = fs_matrix[non_nanidx, :]
    G = gs_matrix[non_nanidx, :]
    R = reward_matrix[non_nanidx]

    # estimate ECDF of Z
    pZ = get_state_probabilities(F, G)

    # calculate mean p(availability)
    p_avail_avg=0.0
    if len(AV)!=0:
        p_avail_avg = np.mean(AV)

    # get rewards vectors for each case: r_i(x,a)
    #r_0(x,0)
    r00 = get_empirical_rewards_estimate(0, 0, F, G, pZ, beta_mu, alpha0_mu, alpha1_mu)

    #r_1(x,0)
    r10 = get_empirical_rewards_estimate(1, 0, F, G, pZ, beta_mu, alpha0_mu, alpha1_mu)

    #r_1(x,1)
    r11 = get_empirical_rewards_estimate(1, 1, F, G, pZ, beta_mu, alpha0_mu, alpha1_mu)

    # get initial value estimates for V(dosage, i)
    # init to 0's! have ordering be dosage_grid(0), dosage_grid(1). dosage_grid(i) means dosagegrid x availability=i
    V = np.zeros(DOSAGE_GRID_LEN * 2)
    theta0 = np.zeros(NBASIS)
    theta1 = np.zeros(NBASIS)

    epsilon = 1e-2
    delta = 10
    iters = 0

    while delta > epsilon and iters < MAX_ITERS:
        # store the old V value
        V_old = V

        # get OLS Estimate
        theta0 = np.matmul(dosage_OLS_soln, V[:DOSAGE_GRID_LEN])
        theta1 = np.matmul(dosage_OLS_soln, V[DOSAGE_GRID_LEN:])

        # update value function
        V = bellman_backup(A, F, G, beta_mu, p_avail_avg, theta0, theta1, r00, r10, r11)
        
        # compute the loss/delta
        delta = np.amax(np.abs(V - V_old))

        iters = iters+1

    return theta0, theta1, p_avail_avg

# %%

def calculate_eta(theta0, theta1, dosage, p_avail, ts, psed=PSED, w=W, gamma=GAMMA, lamb=LAMBDA):
    eta1=etaInit(float(dosage))[0]*(gamma)/(1-gamma)
    # If less than 10 time steps, use etaInit from HeartStepsV1
    if ts < 10:
        return eta1

    cur_dosage_eval0 = dosage_basis.evaluate(dosage * lamb)
    cur_dosage_eval1 = dosage_basis.evaluate(dosage * lamb + 1)

    # Calculate etaHat using peng's
    thetabar = theta0 * (1 - p_avail) + theta1 * (p_avail)
    val = np.sum(thetabar * (cur_dosage_eval0 - cur_dosage_eval1).squeeze().T)

    # Peng most likely got this wrong (He used 1-gamma instead of gamma)
    etaHat = val * (1-psed) * (gamma)
    # etaHat = val * (1-psed) * (1 - gamma)

    eta = w * etaHat + (1-w) * eta1

    return eta

# %%
def run_algorithm(data, user, boot_num, user_specific, residual_matrix, baseline_theta, output_dir=None, log_dir=None):
    '''Run the algorithm for each user and each bootstrap'''

    # Load priors
    alpha0_pmean, alpha0_psd, alpha1_pmean, alpha1_psd, beta_pmean, beta_psd, sigma, prior_sigma,prior_mu = load_priors()

    # Initializing dosage to first dosage value (can be non-zero if user was already in the trial)
    dosage = data[0][6]

    # Posterior initialized using priors
    post_alpha0_mu, post_alpha0_sigma = np.copy(alpha0_pmean), np.copy(alpha0_psd)
    post_alpha1_mu, post_alpha1_sigma = np.copy(alpha1_pmean), np.copy(alpha1_psd)
    post_beta_mu, post_beta_sigma = np.copy(prior_mu), np.copy(prior_sigma)

    # get inverses
    alpha0_sigmaInv=np.linalg.inv(alpha0_psd)
    alpha1_sigmaInv=np.linalg.inv(alpha1_psd)
    beta_sigmaInv=np.linalg.inv(prior_sigma)

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

    # beta
    post_beta_mu_matrix = np.zeros((NDAYS * NTIMES, 2*F_LEN+G_LEN ))
    post_beta_sigma_matrix = np.zeros((NDAYS * NTIMES, 2*F_LEN+G_LEN , 2*F_LEN+G_LEN ))

    eta = 0
    p_avail_avg = 0
    theta0, theta1 = np.zeros(NBASIS), np.zeros(NBASIS)

    p_avail_avgs_matrix = np.zeros((NDAYS * NTIMES))
    etas_matrix = np.zeros((NDAYS * NTIMES))
    all_theta0=[]
    all_theta1=[]

    for day in range(NDAYS):
        # loop for each decision time during the day
        for time in range(NTIMES):

            # Get the current timeslot
            ts = (day) * 5 + time
            
            # State of the user at time ts
            availability, fs, gs, dosage, reward = determine_user_state(data[ts], dosage, action_matrix[ts-1])

            # Save user's availability
            availability_matrix[ts] = availability

            # If user is available
            action, prob_fsb,eta = 0, 0,0
            # Calculate probability of (fs x beta) > n
            ### NOTE THAT eta and probs are calculated regardless of availability, and when availability=0, one would interpret eta/prob_fsb effectively as 0. This is done in case we want to calc effects during non avail times. ###
            eta = calculate_eta(theta0, theta1, dosage, p_avail_avg, ts)
            prob_fsb = calculate_post_prob(ts, data, fs, post_beta_mu[-F_LEN:], post_beta_sigma[-F_LEN:, -F_LEN:], eta)

            if availability == 1:
                # Sample action with probability prob_fsb from bernoulli distribution
                action = select_action(prob_fsb)

            # Bayesian LR to estimate reward
            reward = calculate_reward(ts, fs, gs, action, baseline_theta, residual_matrix)

            # Save probability, features, action and reward
            prob_matrix[ts] = prob_fsb
            action_matrix[ts] = action
            etas_matrix[ts]=eta
            p_avail_avgs_matrix[ts]=p_avail_avg
            
            all_theta0.append(theta0)
            all_theta1.append(theta1)

            # Save features and state
            reward_matrix[ts] = reward

            fs_matrix[ts] = fs
            gs_matrix[ts] = gs

            post_alpha0_mu_matrix[ts] = post_alpha0_mu
            post_alpha0_sigma_matrix[ts] = post_alpha0_sigma

            post_alpha1_mu_matrix[ts] = post_alpha1_mu
            post_alpha1_sigma_matrix[ts] = post_alpha1_sigma

            post_beta_mu_matrix[ts] = post_beta_mu
            post_beta_sigma_matrix[ts] = post_beta_sigma

        # Update posteriors at the end of the day
        post_beta_mu, post_beta_sigma = calculate_posterior_avail(alpha1_psd, alpha1_pmean, beta_psd, beta_pmean, sigma, 
                                                                  availability_matrix[:ts + 1], prob_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                                  action_matrix[:ts + 1], fs_matrix[:ts + 1], gs_matrix[:ts + 1], beta_sigmaInv)

        post_alpha0_mu, post_alpha0_sigma = calculate_posterior_unavail(alpha0_psd, alpha0_pmean, sigma, availability_matrix[:ts + 1], 
                                                                            reward_matrix[:ts + 1], gs_matrix[:ts + 1], alpha0_sigmaInv)

        post_alpha1_mu, post_alpha0_sigma = calculate_posterior_maineffect(alpha1_psd, alpha1_pmean, sigma, availability_matrix[:ts + 1], 
                                                                                        reward_matrix[:ts + 1], action_matrix[:ts + 1], gs_matrix[:ts + 1], alpha1_sigmaInv)

        # update value functions
        theta0, theta1, p_avail_avg = calculate_value_functions(availability_matrix[:ts + 1], action_matrix[:ts + 1], 
                                                    fs_matrix[:ts + 1], gs_matrix[:ts + 1], reward_matrix[:ts + 1], 
                                                    post_beta_mu[-F_LEN:], post_alpha0_mu, post_alpha1_mu, ts)

    result = {"availability": availability_matrix, "prob": prob_matrix, "action": action_matrix, "reward": reward_matrix,
            "post_alpha0_mu": post_alpha0_mu_matrix, "post_alpha0_sigma": post_alpha0_sigma_matrix,
            "post_alpha1_mu": post_alpha1_mu_matrix, "post_alpha1_sigma": post_alpha1_sigma_matrix,
            "post_beta_mu": post_beta_mu_matrix, "post_beta_sigma": post_beta_sigma_matrix,
            "fs": fs_matrix, "gs": gs_matrix,
            "etas": etas_matrix, "p_avail_avg": p_avail_avgs_matrix,
            "theta0": all_theta0, "theta1": all_theta1 
            }
    
    # Save results
    with open(output_dir + f"/results_{user}_{boot_num}.pkl", "wb") as f:
        pkl.dump(result, f)

def resample_user_residuals(residual_matrix, user):
    T= NDAYS * NTIMES
    resampled_indices = np.random.choice(range(T), T)
    residual_matrix=residual_matrix[resampled_indices]
    return residual_matrix

# %%
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=int, required=True, help="User number")
    parser.add_argument("-b", "--bootstrap", type=int, required=True, help="Bootstrap number")
    parser.add_argument("-s", "--seed", type=int, required=True, help="Random seed")
    parser.add_argument("-userBIdx", "--userBootstrapIndex", type=int, required=True, help="User's bs index")
    parser.add_argument("-pec", "--user_specific", default=False, type=bool, required=False, help="User specific experiment")
    parser.add_argument("-o", "--output", type=str, default="./output", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=False, help="Baseline of interest. If you want to run interestingness analysis, put in the F_KEY as a string to get thetas for that tx Effect coef 0ed out.")
    parser.add_argument("-rm", "--residual_matrix", type=str, default="./init/residual_matrix.pkl", required=False, help="Pickle file for residual matrix")
    parser.add_argument("-bp", "--baseline_theta", type=str, default="./init/baseline_parameters.pkl", required=False, help="Pickle file for baseline parameters")
    args = parser.parse_args()

    # Set random seed to the bootstrap number
    np.random.seed(args.seed)

    # Load initial run data
    residual_matrix, baseline_thetas = load_initial_run(args.residual_matrix, args.baseline_theta, args.baseline)

    # Load data
    data = load_data()

    # Prepare directory for output and logging
    print(args.user_specific)

    # note if baseline in F_KEYS, we have an interestingness bootstrap (by nature of baseline thetas used)
    output_dir = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"-NoBS", "Bootstrap-" + str(args.bootstrap), "user-"+str(args.userBootstrapIndex))
    log_dir = os.path.join(args.log, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"-NoBS", "Bootstrap-" + str(args.bootstrap), "user-"+str(args.userBootstrapIndex))

    if args.user_specific:
        output_dir = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"_User-"+str(args.user), "Bootstrap-" + str(args.bootstrap))
        log_dir = os.path.join(args.log, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"_User-"+str(args.user), "Bootstrap-" + str(args.bootstrap))
    print(output_dir)
    print(log_dir)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    residual_matrix=residual_matrix[args.user]
    # need to make it s.t. we are doing different seq of seeds in user specific case.
    #if args.user_specific:
    #    residual_matrix = resample_user_residuals(residual_matrix, args.user)

    # Run algorithm
    run_algorithm(data[args.user], args.user, args.bootstrap, args.user_specific, residual_matrix, baseline_thetas[args.user], output_dir, log_dir)
    print("finished")

# %%

if __name__ == "__main__":
    main()
