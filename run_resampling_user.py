### feeder script for user specific bootstrap ###

import subprocess
import numpy as np
import sys
import os
from datetime import date, datetime
from itertools import product

NUSERS = 91
F_KEYS=["intercept", "dosage", "engagement", "other_location", "variation"]

def main():
    offset=0
    offset=10000
    offset=20000
    offset=30000
    offset=40000

    idx = int(sys.argv[1])+offset
    bruns=500

    # Define the parameters
    boot_run = idx % bruns
    user = idx // bruns #useridx itself does not matter for user-spec run.

    experiment = 4 
    baseline = F_KEYS[experiment]
    #baseline="Zero"

    # When we run, we keep the boot_run as seed, since boot_run goes from 0-500/whatever index. This is consistent with the users in pop bootstrap. also note that userBIdx does not matter in user spec run
    if user <= 90:
        subprocess.run(f'python main.py -pec True -u {user} -b {boot_run} -s {boot_run} -userBIdx {boot_run} -bi {baseline}', shell=True) 
    else:
        print("Went through all users (no need to run - we are done)")

if __name__ == "__main__":
    main()

