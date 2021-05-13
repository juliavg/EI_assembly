#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="local"
MODE="speedup"
MASTER_SEED=7
STIM_IDX=2

# Runs simulation
python3 ${SCRIPTS}3_simulation_assembly.py $WHERE $MODE $MASTER_SEED $STIM_IDX

# Plots
python3 ${SCRIPTS}3_plot_assembly.py $MODE
