"""
Random code lines
"""

# Code to copy-paste in Ubuntu 20 terminal to sync data with a server
# rsync -avzP -e 'ssh -p 4022' mouse@neurocomp.fcrb.es:/archive/mouse/pv_nmdar_eranet* ~/

# rcl code for copy-paste in Ubuntu 20 terminal to upgrade all python packages at once
# pip3 list --outdated --format=freeze | grep -v '^\-e' | cut -d = -f 1 | xargs -n1 pip3 install -U

# To display where the currently active matplotlibrc file was loaded from, one can do the following:
# import matplotlib
# matplotlib.matplotlib_fname()