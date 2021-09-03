#!/bin/bash

WHERE="local"
SCRIPTS="$(pwd)/scripts/"
REF_CV="1.4"

# Generates data for static connections
python3 ${SCRIPTS}2_single_readout.py $WHERE 'static' $REF_CV

# Plots figure
python3 ${SCRIPTS}s5_plot_single_readout_weights.py $WHERE
