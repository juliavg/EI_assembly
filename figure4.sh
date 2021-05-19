#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="local"

# Runs simulation
python3 ${SCRIPTS}4_shift_spike_trains.py $WHERE

# Plots
python3 ${SCRIPTS}4_plot_decay.py $WHERE
