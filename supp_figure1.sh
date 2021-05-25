#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="local"
MODE="static"
MASTER_SEED=15
STIM_IDX=4

# Runs simulation
python3 ${SCRIPTS}3_simulation_assembly.py $WHERE $MODE $MASTER_SEED $STIM_IDX

# Plots
python3 ${SCRIPTS}3_plot_assembly.py $WHERE $MODE $STIM_IDX
