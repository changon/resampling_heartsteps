#python slidingWindow_bootstrap.py -b 965 -bi other_location -pec True -u 1
#python slidingWindow_bootstrap.py -b 965 -bi other_location -pec False -u 1
# %%
import pandas as pd
import numpy as np
import pickle as pkl
from collections import OrderedDict
import argparse
import os
import matplotlib.pyplot as plt
import math

import matplotlib as mpl
import pylab
from matplotlib import rc

fix_plot_settings = True
if fix_plot_settings:
    plt.rc('font', family='serif')
    plt.rc('text', usetex=False)
    label_size = 10
    mpl.rcParams['xtick.labelsize'] = label_size 
    mpl.rcParams['ytick.labelsize'] = label_size 
    mpl.rcParams['axes.labelsize'] = label_size
    mpl.rcParams['axes.titlesize'] = label_size
    mpl.rcParams['figure.titlesize'] = label_size
    mpl.rcParams['lines.markersize'] = label_size
    mpl.rcParams['grid.linewidth'] = 2.5
    mpl.rcParams['legend.fontsize'] = label_size
    pylab.rcParams['xtick.major.pad']=5
    pylab.rcParams['ytick.major.pad']=5

    lss = ['--',  ':', '-.', '-', '--', '-.', ':', '-', '--', '-.', ':', '-']
    mss = ['>', 'o',  's', 'D', '>', 's', 'o', 'D', '>', 's', 'o', 'D']
    ms_size = [25, 20, 20, 20, 20, 20, 20, 20, 20, 20]
    colors = ['#e41a1c', '#0000cd', '#4daf4a',  'black' , 'magenta']
else:
    pass

np.set_printoptions(edgeitems=30, linewidth=100000, formatter=dict(float=lambda x: "%.7g" % x))


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


def readResult(outputDir, userSpec):
    results=[]
    collectedIds=[]
    if userSpec:
        B=500
        for b in range(B):
            d=os.path.join(outputDir, "Bootstrap-"+str(b))
            if os.path.exists(d):
                ff=os.listdir(d)
                if len(ff) > 0:
                    userFile = [f for f in ff if ".pkl" in f]
                    userFilePath=d+"/"+userFile[0]
                    userId=int(userFile[0].split("_")[1])
                    collectedIds.append(userId)
                    res_user=pkl.load(open(userFilePath,"rb"))
                    results.append(res_user)
    else:
        dirs = os.listdir(outputDir)
        dirs = [f for f in dirs if not os.path.isfile(outputDir+'/'+f)] #Filtering only the files.
        print(outputDir)
        print(dirs)
        for user_dir in dirs:
            if "-" in user_dir:
                userFile = os.listdir(outputDir+"/"+user_dir)
                userFile = [f for f in userFile if ".pkl" in f]
                if len(userFile) >0: #should really be ==1
                    userFilePath=outputDir+"/"+user_dir+"/"+userFile[0]
                    userId=int(userFile[0].split("_")[1])
                    res_user=pkl.load(open(userFilePath,"rb"))
                    results.append(res_user)
                    collectedIds.append(userId)
    print(len(results))
    return results, collectedIds

from slidingWindow_og import computeMetricSlidingDay, plotUserDay
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-u", "--user", type=int, required=False, help="User")
    parser.add_argument("-b", "--bootstrap", type=int, required=True, help="Bootstrap number")
    #parser.add_argument("-fsIndex", "--fsIndex", type=int, required=True, help="FS Index")
    parser.add_argument("-pec", "--user_specific", default=False, type=bool, required=False, help="User specific experiment")
    parser.add_argument("-o", "--output", type=str, default="./output", required=False, help="Output file directory name")
    parser.add_argument("-l", "--log", type=str, default="./log", required=False, help="Log file directory name")
    parser.add_argument("-bi", "--baseline", type=str, default="Prior", required=True, help="Baseline of interest. If you want to run interestingness analysis, put in the F_KEY as a string to get thetas for that tx Effect coef 0ed out.")
    parser.add_argument("-rm", "--residual_matrix", type=str, default="./init/residual_matrix.pkl", required=False, help="Pickle file for residual matrix")
    parser.add_argument("-bp", "--baseline_theta", type=str, default="./init/baseline_parameters.pkl", required=False, help="Pickle file for baseline parameters")
    args = parser.parse_args()

    print(args.user_specific)

    # note if baseline in F_KEYS, we have an interestingness bootstrap (by nature of baseline thetas used)
    # it will be ./output/Baseline-Prior_UserSpecific-False/Bootstrap-0/user-0/user-trueIndex-bootNum
    output_dir = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific), "Bootstrap-" + str(args.bootstrap))
    log_dir = os.path.join(args.log, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific), "Bootstrap-" + str(args.bootstrap))
    if args.user_specific:
        output_dir = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"_User-"+str(args.user))# , "Bootstrap-" + str(args.bootstrap))
        log_dir = os.path.join(args.log, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"_User-"+str(args.user))#, "Bootstrap-" + str(args.bootstrap))
    print(output_dir)
    print(log_dir)
    if not os.path.exists(output_dir):
        print("DNE!")
        os.makedirs(output_dir)
    if not os.path.exists(log_dir):
        print("DNE!")
        os.makedirs(log_dir)

    txEffectState = False #if posterior etc
    if args.baseline in F_KEYS: #if engagement,...
        indexFS=F_KEYS.index(args.baseline)
        txEffectState=True
    if args.baseline=="Zero":
        txEffectState=True
        indexFS=2

    # read in results from original run and bootstrap
    bs_result, realIds=readResult(output_dir, args.user_specific)
    nNone=0
    
    print(len(bs_result))
    #delta setting here does not matter. just computing r1/r2 here first.
    delta1=.75
    delta2=.4

    if txEffectState:
        r1s=[]
        r2s=[]
        rawR2s=[]
        r3s=[]
        r4s=[]
        rawR4s=[]
        effectRatios=[]
        effectIds=[]
        interestingR2s=[]
        interestingR1s=[]
        interestingIds=[]
        interestings=[]
        for i in range(len(bs_result)):
            result=computeMetricSlidingDay(bs_result[i], indexFS, delta1=delta1, delta2=delta2)
            #print("USER "+str(i))
            r1s.append(result['r1'])
            r2s.append(result['r2'])
            rawR2s.append(result['rawR2'])
            r3s.append(result['r3'])
            r4s.append(result['r4'])
            rawR4s.append(result['rawR4'])
            effectRatios.append(result['stdEffectRatio'])
            effectIds.append(realIds[i])
            #interestings.append(result['isInteresting'])
            if result['r1']!=None and result['r2']!=None:
                pass
#                if result['isInteresting']:
#                    print("found interesting user")
#                    interestingIds.append(i)
#                    outputPath = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific), "Bootstrap-" + str(args.bootstrap))#, "user-"+str(i))
#                    if args.user_specific:
#                        outputPath = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"_User-"+str(args.user), "Bootstrap-" + str(args.bootstrap))
#                    plotUserDay(result, bs_result[i], i, outputPath, F_KEYS[indexFS], args.user_specific)
            else:
                nNone=nNone+1

        print(nNone)
        print("reses")
        print(r3s)
        print(r4s)
 
        a_file_name=output_dir+"/results.csv"
        b_file_name=output_dir+"/statistics.csv"
        print("wrote "+a_file_name)
#        if args.user_specific:
#            output_dir = os.path.join(args.output, "Baseline-"+ str(args.baseline)+"_UserSpecific-"+str(args.user_specific)+"_User-"+str(args.user))
#            a_file_name=output_dir+"/results-"+ str(args.bootstrap) +".csv"
#            b_file_name=output_dir+"/statistics-"+ str(args.bootstrap) +".csv"

        # save a file in each dir:
        a_file = open(a_file_name, "w")
        header="bootstrapIndex,trueId,effectRatio,r1,r2,rawR2,r3,r4,rawR4\n"
        a_file.write(header)
        for user in range(len(r2s)):
            statisticsLine=str(user)+","+str(effectIds[user])+","+str(effectRatios[user])+","+str(r1s[user])+ ","+str(r2s[user])+ ","+ str(rawR2s[user])+","+str(r3s[user])+","+str(r4s[user])+","+str(rawR4s[user])
            a_file.write(statisticsLine+"\n")
        a_file.close()

        a_file = open(b_file_name, "w")
        header="nNone \n"
        a_file.write(header)
        statisticsLine=str(nNone)
        a_file.write(statisticsLine+"\n")
        a_file.close()

    else:
        for j in range(2,5):
            indexFS=j
            r1s=[]
            r2s=[]
            rawR2s=[]
            effectRatios=[]
            effectIds=[]
            interestingR2s=[]
            interestingR1s=[]
            interestingIds=[]
            interestings=[]
            for i in range(len(bs_result)):
                result=computeMetricSlidingDay(bs_result[i], indexFS,delta1=delta1, delta2=delta2)
                r1s.append(result['r1'])
                r2s.append(result['r2'])
                rawR2s.append(result['rawR2'])
                effectRatios.append(result['stdEffectRatio'])
                effectIds.append(realIds[i])
                if result['r1']!=None and result['r2']!=None:
                    pass
                else:
                    nNone=nNone+1

            print(nNone)
 
            a_file_name=output_dir+"/results_"+F_KEYS[indexFS]+".csv"
            b_file_name=output_dir+"/statistics_"+F_KEYS[indexFS]+".csv"
            print("wrote "+a_file_name)

            # save a file in each dir:
            a_file = open(a_file_name, "w")
            header="bootstrapIndex, trueId, effectRatio, r1, r2, rawR2\n"
            a_file.write(header)
            for user in range(len(r2s)):
                statisticsLine=str(user)+","+str(effectIds[user])+","+str(effectRatios[user])+","+str(r1s[user])+ ","+str(r2s[user])+ ","+ str(rawR2s[user])
                a_file.write(statisticsLine+"\n")
            a_file.close()

            a_file = open(b_file_name, "w")
            header="nNone \n"
            a_file.write(header)
            statisticsLine=str(nNone)
            a_file.write(statisticsLine+"\n")
            a_file.close()
    print("Finished "+str(args.baseline))#+ " , "+str(user))

if __name__=="__main__":
    main()
