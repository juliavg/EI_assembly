# -*- coding: utf-8 -*-
#
# brunel2000_interactive.py
#
# This file is part of NEST.
#
# Copyright (C) 2004 The NEST Initiative
#
# NEST is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 2 of the License, or
# (at your option) any later version.
#
# NEST is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with NEST.  If not, see <http://www.gnu.org/licenses/>.


import nest
import numpy as np
import matplotlib.pyplot as plt

direc = '../data/single_neuron/data_vm/'

stim_strength_all = np.array([1,2,3,4,5])


for ss,strength in enumerate(stim_strength_all):
    mean_weight = np.load("../data/single_neuron/data_rate/mean_weight_"+str(ss)+".npy")
    inh_weight  = np.mean(mean_weight[-int(mean_weight.shape[0]/4):])

    nest.ResetKernel()

    # Network parameters. These are given in Brunel (2000) J.Comp.Neuro.
    delay    = 1.5    # synaptic delay in ms
    tau_m    = 20.0      # Membrane time constant (mV)
    V_th     = 20.0      # Spike threshold (mV)
    C_m      = 250.0     # Membrane capacitance (pF)
    t_ref    = 2.0       # Refractory period (ms)
    E_L      = 0.0       # Resting membrane potential (mV)
    V_reset  = 10.0      # Reset potential after spike (mV)
    tau_psc  = 1.5

    N_E = 160          # Number of excitatory input neurons
    N_I = 40           # Number of inhibitory input neurons
    N_neuron = N_E+N_I

    # PSP to PSC
    sub             = 1. / (tau_psc - tau_m)
    pre             = tau_m * tau_psc / C_m * sub
    frac            = (tau_m / tau_psc) ** sub
    PSC_over_PSP    = 1. / (pre * (frac**tau_m - frac**tau_psc))

    J_ext = 0.05*PSC_over_PSP                 # Excitatory synaptic strength (mV PSP max amplitude)
    J_E = 0.15*PSC_over_PSP                 # Excitatory synaptic strength (mV PSP max amplitude)

    input_rate = 9.     # rate of input neurons (Hz)

    ext_rate  = 18000.               # 15000 Hz for rho=15,25; 8000 Hz for rho=8; 4000 Hz for rho=3

    noise_E_rate = ext_rate+N_E*input_rate

    # iSTDP parameters
    tau_stdp = 20.                      # Time constant for iSTDP window (ms)
    WmaxI    = 100*J_E                      # Maximum weight of iSTDP connections (mV)
    eta      = 0.01*J_E                     # Learning rate for iSTDP
    rho      = 9                       # Target rate for iSTDP (Hz)
    alpha    = 2*rho*tau_stdp/1000.     # alpha parameter for iSTDP

    simtime = 200000.

    cv_interval = 50000.

    # Set parameters of the NEST simulation kernel
    nest.SetKernelStatus({'print_time': True,'local_num_threads':1})


    nest.SetDefaults('iaf_psc_exp', 
                      {"tau_m"      : tau_m,
                       "t_ref"      : t_ref,
                       "tau_syn_ex" : tau_psc,
                       "tau_syn_in" : tau_psc,
                       "C_m"        : C_m,
                       "V_reset"    : V_reset,
                       "E_L"        : E_L,
                       "V_m"        : E_L,
                       "V_th"       : V_th})

    # Create nodes -------------------------------------------------

    node   = nest.Create('iaf_psc_exp')
    parrot = nest.Create('parrot_neuron')
    nest.SetStatus(node,params={'V_th':1000.})

    noise_E = nest.Create('poisson_generator', 2)
    noise_I = nest.Create('poisson_generator', 1, {'rate': N_I*input_rate})
    
    nest.SetStatus(noise_E,'rate', [ext_rate,N_E*input_rate])

    multimeter = nest.Create('multimeter')
    nest.SetStatus(multimeter,params={'record_from':['V_m']})

    # Connect nodes ------------------------------------------------

    nest.CopyModel('static_synapse',
                   'excitatory',
                   {'weight':strength*J_E, 
                    'delay':delay})

    nest.CopyModel('static_synapse',
                   'external',
                   {'weight':J_ext, 
                    'delay':delay})

    nest.Connect([noise_E[0]], node,'all_to_all','external')
    nest.Connect([noise_E[1]], node,'all_to_all','excitatory')

    nest.CopyModel('static_synapse',
                   'inhibitory',
                   {'weight':inh_weight, 
                    'delay':delay})

    nest.Connect(noise_I,node,'all_to_all','inhibitory')

    nest.Connect(multimeter, node)

    nest.Simulate(simtime)


    # Analysis ------------------------------------------------------

    # Vm
    events  = nest.GetStatus(multimeter,'events')[0]
    time_vm = events['times']
    vm      = events['V_m']

    extension = "_"+str(ss)+".npy"
    np.save(direc+"vm"+extension,vm)
    np.save(direc+"time_vm.npy",time_vm)
