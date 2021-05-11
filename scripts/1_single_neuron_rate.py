import nest
import numpy as np
import matplotlib.pyplot as plt

direc = '../data/single_neuron/data_rate/'

# Network parameters. These are given in Brunel (2000) J.Comp.Neuro.
delay    = 1.5    # synaptic delay in ms
tau_m    = 20.0      # Membrane time constant (mV)
V_th     = 20.0      # Spike threshold (mV)
C_m      = 250.0     # Membrane capacitance (pF)
t_ref    = 2.0       # Refractory period (ms)
E_L      = 0.0       # Resting membrane potential (mV)
V_reset  = 10.0      # Reset potential after spike (mV)
tau_psc  = 1.5

N_E = 160           # Number of excitatory input neurons
N_I = 40           # Number of inhibitory input neurons
N_neuron = N_E+N_I

# PSP to PSC
sub             = 1. / (tau_psc - tau_m)
pre             = tau_m * tau_psc / C_m * sub
frac            = (tau_m / tau_psc) ** sub
PSC_over_PSP    = 1. / (pre * (frac**tau_m - frac**tau_psc))

J_ext = 0.05*PSC_over_PSP
J_E = 0.15*PSC_over_PSP                 # Excitatory synaptic strength (mV PSP max amplitude)

input_rate = 9.     # rate of input neurons (Hz)

ext_rate  = 18000.               # 15000 Hz for rho=15,25; 8000 Hz for rho=8; 4000 Hz for rho=3

# iSTDP parameters
tau_stdp = 20.                      # Time constant for iSTDP window (ms)
WmaxI    = 100*J_E                      # Maximum weight of iSTDP connections (mV)
eta      = 0.01*J_E                     # Learning rate for iSTDP
rho      = 9                       # Target rate for iSTDP (Hz)
alpha    = 2*rho*tau_stdp/1000.     # alpha parameter for iSTDP

n_neurons = 10

simtime = 200000.

binsize = 5000.

time_bins    = np.arange(0,2*simtime+binsize,binsize)
record_final = 50000.

stim_strength_all = np.array([1,2,3,4,5])

np.save(direc+"stim_strength_all.npy",stim_strength_all)
for ss,strength in enumerate(stim_strength_all):
    nest.ResetKernel()

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

    node   = nest.Create('iaf_psc_exp',n_neurons)
    parrot = nest.Create('parrot_neuron',n_neurons)

    noise_E = nest.Create('poisson_generator', 2)
    noise_I = nest.Create('poisson_generator', 1, params={'rate': N_I*input_rate})

    nest.SetStatus(noise_E,'rate', [ext_rate,N_E*input_rate])

    spikes  = nest.Create('spike_detector')

    weight_recorder = nest.Create('weight_recorder',params={'targets':[node[0]]})


    # Connect nodes ------------------------------------------------

    nest.CopyModel('static_synapse',
                   'excitatory',
                   {'weight':J_E, 
                    'delay':delay})

    nest.CopyModel('static_synapse',
                   'external',
                   {'weight':J_ext, 
                    'delay':delay})

    nest.Connect([noise_E[0]], node,'all_to_all','external')
    nest.Connect([noise_E[1]], node,'all_to_all','excitatory')

    nest.CopyModel('vogels_sprekeler_synapse',
                   'plastic_inhibitory',
                   {'tau': tau_stdp, 
                    'Wmax': -WmaxI,
                    'eta': eta,
                    'alpha': alpha,
                    'weight': -.1,
                    'weight_recorder': weight_recorder[0]})

    nest.Connect(parrot,node,'one_to_one','plastic_inhibitory')

    nest.Connect(noise_I, parrot)

    nest.Connect(node, spikes)

    nest.Simulate(simtime)

    connections = nest.GetConnections([noise_E[1]],node)

    nest.SetStatus(connections,params={'weight':strength*J_E})
    nest.Simulate(simtime)

    # Analysis ------------------------------------------------------

    # Rate and CV
    events     = nest.GetStatus(spikes,'events')[0]
    sd_senders = events['senders']
    sd_times   = events['times']
    sd_senders = sd_senders[np.argsort(sd_times)]
    sd_times   = sd_times[np.argsort(sd_times)]

    for nn in node:
        times_neuron = sd_times[sd_senders==nn]
        rate_series  = np.histogram(times_neuron,bins=time_bins)[0]/binsize*1000.
        rate_final   = len(times_neuron[times_neuron>(2*simtime-record_final)])/record_final*1000.

        isi = np.diff(times_neuron[times_neuron>(2*simtime-record_final)])
        cv  = np.std(isi)/np.mean(isi)
        
        extension = "_"+str(nn)+"_"+str(ss)+".npy"
        np.save(direc+"rate_series"+extension,rate_series)
        np.save(direc+"rate_final"+extension,rate_final)
        np.save(direc+"cv"+extension,cv)
        

    # Weights
    wr_status  = nest.GetStatus(weight_recorder,'events')[0]
    wr_times   = wr_status['times']
    wr_weights = wr_status['weights']

    mean_weight = np.histogram(wr_times,bins=time_bins,weights=wr_weights)[0]/np.histogram(wr_times,bins=time_bins)[0]

    np.save(direc+"mean_weight_"+str(ss)+".npy",mean_weight)
    np.save(direc+"time_bins_"+str(ss)+".npy",time_bins)
