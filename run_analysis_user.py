import subprocess
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product

NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    user=int(sys.argv[1])
    experiment = int(sys.argv[2])
    baseline="Zero"
    if experiment!=-1:
        baseline=F_KEYS[experiment]
    BRUN=500

    #idx does not matter here
    idx=user
    subprocess.run(f'python slidingWindow_bootstrap.py -b {idx} -bi {baseline} -pec True -u {user}', shell=True) 
    print("User "+str(user)+" baseline is "+baseline)

if __name__ == "__main__":
    main()
