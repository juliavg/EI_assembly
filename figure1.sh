#!/bin/bash

# Generates rate and cv matrix from theory
python3 scripts/1_rate_cv_theo.py

# Generates firing rate data from simulation
python3 scripts/1_single_neuron_rate.py 'local'

# Generates subthreshold membrane potential data from simulation
# has to be run after 1_single_neuron_rate.py, because it uses inhibitory weight generated during that simulation
python3 scripts/1_single_neuron_vm.py 'local'

# Plots figure
python3 plot_scripts/1_plot_single_neuron.py
