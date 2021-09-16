#!/bin/bash

WHERE="local"
MODE="plastic"
STIM_IDX=2
SCRIPTS="$(pwd)/scripts/"

# Plots figure
python3 ${SCRIPTS}s6_plot_time_series.py $WHERE $MODE $STIM_IDX
