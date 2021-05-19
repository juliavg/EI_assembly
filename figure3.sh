#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="local"
MODE="plastic"
MASTER_SEED=0
STIM_IDX=2
SPEED="normal"

# Runs simulation
python3 ${SCRIPTS}3_simulation_assembly.py $WHERE $MODE $MASTER_SEED $STIM_IDX

# Plots
python3 ${SCRIPTS}3_plot_assembly.py $WHERE $MODE $STIM_IDX
