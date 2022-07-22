"""
Random code lines
"""

# Code to copy-paste in Ubuntu 20 terminal to sync data with a server
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/

# Code for copy-paste in Ubuntu 20 terminal to upgrade all python packages at once
# pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U

# To display where the currently active matplotlibrc file was loaded from, one can do the following:
# import matplotlib
# matplotlib.matplotlib_fname()

# How to pickle
# save:
with open('bootstrap_samples_100.pickle', 'wb') as handle:
    pickle.dump(bootstrap_samples_100, handle, protocol=pickle.HIGHEST_PROTOCOL)

# load:
with open('../Data/new/df_corrected.pickle', 'rb') as handle:
    df_dat = pickle.load(handle)

# How to 'listen' specific sounds with the DAC mic in the setup:
# 1. Change the calibration of 'on_play_Sound1' from '001.wav' to the desired sound ('xxx.wav')
# 2. Open the console, go to the directory 'cd pluginsr-for-pybpod/sound\ test'
# 3. Run 'python sount_test.py "/devttyACMX" "left/right/both"
# 4. Modify in sount_test the2nd sleep (line 57) to adjust the window in which dB are calculated