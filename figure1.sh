#!/bin/bash

WHERE="local"
SCRIPTS="$(pwd)/scripts/"

# Generates rate and cv matrix from theory
python3 ${SCRIPTS}1_rate_cv_theo.py $WHERE

# Generates firing rate data from simulation
python3 ${SCRIPTS}1_single_neuron_rate.py $WHERE

# Generates subthreshold membrane potential data from simulation
# has to be executed after 1_single_neuron_rate.py, because it uses inhibitory weight generated during that simulation
python3 ${SCRIPTS}1_single_neuron_vm.py $WHERE

# Plots figure
python3 ${SCRIPTS}1_plot_single_neuron.py $WHERE
