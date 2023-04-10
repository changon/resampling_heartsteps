
# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import math

# %%
NDAYS = 90
NUSERS = 91
NTIMES = 5
T=NDAYS*NTIMES

F_KEYS = ["intercept", "dosage", "engagement", "other_location", "variation"]
F_LEN = len(F_KEYS)
G_KEYS = ["intercept", "temperature", "logpresteps", "sqrt_totalsteps", "dosage", "engagement", "other_location", "variation"]
G_LEN = len(G_KEYS)
def rindex(l, value):
    for i in reversed(range(len(l))):
        if l[i]==value:
            return i
    return -1

## Int Score Computation, at day level sliding window
def computeMetricSlidingDay(result, indexFS, x=2, delta1=.5, delta2=.5, IGNORE_ETA=False):
    ndata=rindex(result['availability'][:T], 1)+1 # +1 for actual number of timepoints. 
    if ndata==0: #if no one is found
        print("no availability :(")
        return {'isEqualAvail':False, 'isEqualEngAvail':False, 'r1':None, 'r2':None, 'stdEffectRatio': None}

    last_day=math.floor((ndata-1)/NTIMES) #ndata is the ts index of the last available time
    
    # Generate list of standardized adv forecasts for this user.
    varValues=[] #binary 0/1 for \var (z)
    stdEffects=[] #the standardized effects for the user (\hat{\Delta}(\cdot))
    etas=[] # eta values

    effLastDay=min(last_day+1+1,90)
    for day in range(effLastDay): #want it to iterate to last day so +1. One more in case we stop before day 90 and can still forecast next day.
        for time in range(NTIMES):
            ts = (day) * 5 + time
            if result['availability'][ts]==1.:
                ## get mean/std and eta
                # the one at ts is really the posteriors from the last day, or (ts//5)-1 since we update at end of the day. 
                # Consequently, the values at day 0, or times 0-4, correspond to the prior params
                beta=result['post_beta_mu'][day*5][-len(F_KEYS):] 
                mean =result['fs'][ts] @ beta

                sigma=result['post_beta_sigma'][day*5][-len(F_KEYS):, -len(F_KEYS):]
                std=math.sqrt((result['fs'][ts] @ sigma.T) @ result['fs'][ts])
                
                eta=0 if IGNORE_ETA else result['etas'][ts]
                
                ## compute stdEffect
                etas.append(eta)
                stdEffect=(mean-eta)/std

                stdEffects.append(stdEffect)
                varValues.append(result['fs'][ts, indexFS])
            else: #if not available put none
                stdEffects.append("NONE")
                varValues.append("NONE")

    varValues=np.array(varValues)

    # for computing int scores, track a few variables
    ## int score 2, \var
    nSlidingWindows=NDAYS
    nInterestingDeterminedWindows=0
    nDeterminedWindows=0
    
    ## int score 1
    nSlidingWindows_intscore1=NDAYS
    nInterestingDeterminedWindows_intscore1=0
    nDeterminedWindows_intscore1=0

    avgVarEffAll=[]
    avgNonVarEffAll=[]
    determinedTimes=[]
    
    # loop through each day, and (1) form sliding windows, (2) check for G_{d,1}, (3) compute int_score
    for day in range(effLastDay):
        avail_idx_pre2=np.array([]) # var used for determining if an update occurred 2 days before?
        avail_idx_pre1=np.array([]) # --""-- 1 day before
        avail_idx_cur=np.array([]) # --""-- current day
        if day == 0: #length of sliding window is 2*NTIMES
            startTime=0
            endTime=NTIMES*2

            # check day 0 has any updates
            avail_idx_cur = np.logical_and(~np.isnan(result['reward'][0:NTIMES]), result['availability'][0:NTIMES] == 1)

        elif day >= 1 and day < last_day: #length of sliding window is 3*NTIMES
            startTime=day*NTIMES-NTIMES
            endTime=day*NTIMES+NTIMES*2

            if day>=2:
                avail_idx_pre2 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)]), result['availability'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)] == 1)
            avail_idx_pre1 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-NTIMES):(day*NTIMES)]), result['availability'][(day*NTIMES-NTIMES):(day*NTIMES)] == 1)
            avail_idx_cur = np.logical_and(~np.isnan(result['reward'][(day*NTIMES):(day*NTIMES+NTIMES)]), result['availability'][(day*NTIMES):(day*NTIMES+NTIMES)] == 1)

        else: #if last_day, length of sliding window is 2*NTIMES
            startTime=day*NTIMES-NTIMES
            endTime=day*NTIMES+NTIMES
            
            # check day lastday-1 has any updates
            avail_idx_pre2 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)]), result['availability'][(day*NTIMES-2*NTIMES):(day*NTIMES-NTIMES)] == 1)
            avail_idx_pre1 = np.logical_and(~np.isnan(result['reward'][(day*NTIMES-NTIMES):(day*NTIMES)]), result['availability'][(day*NTIMES-NTIMES):(day*NTIMES)] == 1)
            if day < 89:
                endTime=day*NTIMES+2*NTIMES
                avail_idx_cur = np.logical_and(~np.isnan(result['reward'][(day*NTIMES):(day*NTIMES+NTIMES)]), result['availability'][(day*NTIMES):(day*NTIMES+NTIMES)] == 1)

        ## Subset above varValues and forecasts to sliding window durations
        varWindow=varValues[startTime:endTime]
        forecastsWindow=stdEffects[startTime:endTime]

        # check that an update happened before any of the day windows in question. 
        enoughUpdates = (sum(avail_idx_pre2)>0) or (sum(avail_idx_pre1)>0) or (sum(avail_idx_cur)>0) #a function of non observed reward. 
        
        # G_{d,1} condition
        varIndices=np.where(varWindow==1)[0]
        nonVarIndices=np.where(varWindow==0)[0]
        nBlue=len(varIndices)
        nRed=len(nonVarIndices)
        isDetermined = (nBlue >=x) and (nRed >= x) and enoughUpdates

        #if G_{d,1} for intscore 2
        if isDetermined: 
            nDeterminedWindows=nDeterminedWindows+1
            determinedTimes.append(day)

            # calculate day-level avg forecasts
            avgVarEffect=np.mean(forecastsWindow[varIndices])
            avgNonVarEffect=np.mean(forecastsWindow[nonVarIndices])

            avgVarEffAll.append(avgVarEffect)
            avgNonVarEffAll.append(avgNonVarEffect)
            
            # compare to determine if an 'interesting' window
            if avgVarEffect > avgNonVarEffect:
                nInterestingDeterminedWindows=nInterestingDeterminedWindows+1

        # if G_{d,1} for intscore 1
        if sum(avail_idx_pre1) > 0 or day==0: # if an update occurred the day prior
            effects_intscore1=stdEffects[(day*NTIMES):(day*NTIMES+NTIMES)][effects_intscore1!="None"]
            #effects_intscore1=effects_intscore1[effects_intscore1!="None"]
            if len(effects_intscore1)>=2: #if we have enough data in the day window
                nDeterminedWindows_intscore1=nDeterminedWindows_intscore1+1
                if np.mean(effects_intscore1)>0: # if the window is interesting enough
                    nInterestingDeterminedWindows_intscore1=nInterestingDeterminedWindows_intscore1+1

    nUndeterminedSlidingWindows=nSlidingWindows-nDeterminedWindows
    nUndeterminedSlidingWindows_intscore1=nSlidingWindows_intscore1-nDeterminedWindows_intscore1

    # output int scores (one and two sided) and G_{d,1} fractions
    statistic={}
    # int score 2
    if nSlidingWindows >0 and nDeterminedWindows >0:
        statistic["r1"]=nUndeterminedSlidingWindows/nSlidingWindows
        statistic["rawR2"]=nInterestingDeterminedWindows/nDeterminedWindows
        statistic["r2"]=abs(nInterestingDeterminedWindows/nDeterminedWindows - 0.5)
        statistic["isInteresting_2"]=(statistic["r1"]<=delta1) and (statistic["r2"]>=delta2)
    else: 
        statistic["r1"]=None
        statistic["r2"]=None
        statistic["rawR2"]=None
        statistic["isInteresting_2"]=None
        
    # int score 1
    if nSlidingWindows_intscore1>0 and nDeterminedWindows_intscore1 > 0:
        statistic["r3"]=nUndeterminedSlidingWindows_intscore1/nSlidingWindows_intscore1#modified to just be for if there are enough updates.
        statistic["rawR4"]=nInterestingDeterminedWindows_intscore1/nDeterminedWindows_intscore1 #
        statistic["r4"]=abs(nInterestingDeterminedWindows_intscore1/nDeterminedWindows_intscore1 - 0.5) #
        statistic["isInteresting_1"]=(statistic["r3"]<=delta1) and (statistic["r4"]>=delta2)
    else:
        statistic["r3"]=None
        statistic["r4"]=None
        statistic["rawR4"]=None
        statistic["isInteresting_1"]=None
    
    # to reproduce twin curves of avg of engaged and not engaged effects at determiend times
    statistic["determinedTimes"]=determinedTimes
    statistic["avgNonValAll"]=avgNonEngageAll
    statistic["avgValAll"]=avgEngageAll
    
    # to reproduce plot of standardized posterior at engaged and not engaged states
    statistic['varValues']=varValues
    statistic['standardizedEffects']=stdEffects
    statistic['etas']=etas
    return statistic