#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="local"

# Plots
python3 ${SCRIPTS}s4_plot_full_raster.py $WHERE
