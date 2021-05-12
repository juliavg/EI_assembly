#!/bin/bash

WHERE='local'
SCRIPTS="$(pwd)/scripts/"

# Generates data for static connections
python3${SCRIPTS}2_single_readout.py $WHERE 'static'

# Generates data for plastic connections
python3 ${SCRIPTS}2_single_readout.py $WHERE 'stp'

# Plots figure
python3 ${SCRIPTS}2_plot_single_readout.py $WHERE
