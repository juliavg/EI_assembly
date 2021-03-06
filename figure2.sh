#!/bin/bash

WHERE="local"
SCRIPTS="$(pwd)/scripts/"
REF_CV="0.4"

# Generates data for plastic connections
python3 ${SCRIPTS}2_single_readout.py $WHERE 'stp'

# Generates data for static connections
python3 ${SCRIPTS}2_single_readout.py $WHERE 'static' $REF_CV

# Plots figure
python3 ${SCRIPTS}2_plot_single_readout.py $WHERE
