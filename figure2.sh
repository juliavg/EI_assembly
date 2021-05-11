#!/bin/bash

# Generates data for static connections
python3 scripts/2_single_readout.py 'static'

# Generates data for plastic connections
python3 scripts/2_single_readout.py 'stp'

# Plots figure
python3 plot_scripts/2_plot_single_readout.py
