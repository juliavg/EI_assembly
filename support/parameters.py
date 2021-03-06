import numpy as np

# Paths
path_to_figure = {}
path_to_figure['local'] = ''
path_to_figure['cluster'] = ''

path_to_data = {}
path_to_data['local'] = ''
path_to_data['cluster'] = ''


# Simulation
n_threads  = {}
n_threads['local']   = 2
n_threads['cluster'] = 32
print_time = True


# Neuron 
neuron_model = 'iaf_psc_exp'
delay        = 1.5              # Synaptic delay (ms)
tau_m        = 20.0             # Membrane time constant (ms)
V_th         = 20.0             # Spike threshold (mV)
C_m          = 250.0            # Membrane capacitance (pF)
t_ref        = 2.0              # Refractory period (ms)
E_L          = 0.0              # Resting membrane potential (mV)
V_reset      = 10.0             # Reset potential after spike (mV)
tau_psc      = 1.5              # Synaptic time constant (ms)


# Synaptic weights
sub             = 1. / (tau_psc - tau_m)
pre             = tau_m * tau_psc / C_m * sub
frac            = (tau_m / tau_psc) ** sub
PSC_over_PSP    = 1. / (pre * (frac**tau_m - frac**tau_psc))        # Convert PSP to PSC

IE_ratio = 10                           # Ratio of IPSP to EPSP amplitude: J_I/J_E
J_ext    = 0.05*PSC_over_PSP            # Syaptic strength of external input (mV PSP max amplitude)
J_E      = 0.15*PSC_over_PSP            # Excitatory synaptic strength (mV PSP max amplitude)
J_I      = -IE_ratio * J_E              # Inhibitory synpatic strength (mV PSP max amplitude)

WmaxE    = [1.5,3.,5.,5.5,6.]                       # Maximum weight for excitatory plasticity
WminE    = J_E                          # Minimum weight for excitatory plasticity


# Network
N_E = 1600               # Number of excitatory neurons
N_I = 400                # Number of inhibitory neurons
N_neurons = N_E + N_I    # Total number of neurons in the network

C_EE = int(N_E * 0.1)    # E to E
C_EI = int(N_I * 0.1)    # I to E
C_IE = int(N_E * 0.1)    # E to I
C_II = int(N_I * 0.1)    # I to I

p_rate = 18000.          # Rate of the external population (s^-1)

autapses  = False        # Whether or not connections from a neuron to itself are allowed
multapses = False        # Whether or not multiple connections between the same pair of neurons are allowed

# Plasticity
learning_rate = {}            # Speedup term for plasticity
learning_rate['normal'] = 1
learning_rate['fast']   = 10

# Triplets
tau_minus   = 33.7
tau_plus    = 16.8
tau_x       = 101.
tau_y       = 125.
A2_plus     = 7.5e-10
A3_plus     = 9.3e-3
A2_minus    = 7e-3
A3_minus    = 2.3e-4

# iSTDP
tau_stdp = 20.                      # Time constant for iSTDP window (ms)
WmaxI    = 100*J_E                  # Maximum weight of iSTDP connections (mV)
eta      = 0.01*J_E        # Learning rate for iSTDP
rho      = 9                        # Target rate for iSTDP (Hz)
alpha    = 2*rho*tau_stdp/1000.     # alpha parameter for iSTDP

# Short term plasticity
tau_rec = 100.                      # Recovery time
tau_fac = 100.                      # Facilitation time
U       = 0.02                      # Facilitation parameter U
A       = 1000.                     # PSC weight in pA


# Dictionary
neuron_param_dict = {'tau_m'              : tau_m,
                     't_ref'              : t_ref,
                     'tau_syn_ex'         : tau_psc,
                     'tau_syn_in'         : tau_psc,
                     'C_m'                : C_m,
                     'V_reset'            : V_reset,
                     'E_L'                : E_L,
                     'V_m'                : E_L,
                     'V_th'               : V_th,
                     'tau_minus'          : tau_minus,
                     'tau_minus_triplet'  : tau_y}


# Stimulation parameters
stim_strength = 5.              # Strength of stimulation
assembly_size = int(N_E*0.1)    # Number of neurons to be stimulated


# Excitatory input to readout
n_input_readout = assembly_size         # Number of excitatory neurons that project to readout
n_readout       = 2                     # Number of readout neurons
noise_mean_std  = (150.,0.)             # Mean and standard deviation of input current to readout (other than spiking from E neurons)


# Assembly simulation parameters
warmup_time             = 2000000.      # Time for network warmup (ms)
stimulation_time        = 1000.         # Stimulation time (ms)
post_stimulation_time   = 500000.       # Post stimulation time (ms)
decay_time              = {}            # Decay time (ms)
decay_time['normal']    = 1500000.
decay_time['fast']      = 2500000. 

rec_spk_from    = 10                # Number of excitatory neurons to record spikes from
rec_weight_from = 5                 # Number of pairs to record weight from
save_for        = 50000.            # Period for saving spike data from all neurons (ms)

labels = ['grow','stim','post','decay']

# Single neuron simulation
mu_range      = (1.,20.,.1)                                 # Minimum mu, maximum mu, mu step
std_range     = (1.,30.,.1)                                 # Minimum std, maximum std, std step
rates_contour = np.array([0.001,0.008,0.05])                 # Rates to extract contour lines from (/ms)

n_single_neurons    = 10                                        # Number of single neurons to simulate
single_sim_time     = 400000.                                   # Total simulation time (ms)
single_binsize      = 5000.                                     # Binsize for calculating rate time series (ms)
single_bins         = np.arange(0,single_sim_time+single_binsize,single_binsize)      # Rate bins (ms)
single_rec_final    = 50000.                                    # Period for recording final rate (ms)

stim_strength_all = np.array([1,2,3,4,5])               # Strength of stimulation . W_EE weights will be increased as *stim_strength_all


# Single readout simulation
readout_n_spikes = 1000
readout_sim_time = 55000.
readout_warmup   = 5000.
readout_binsize  = 10.
readout_bins     = np.arange(0,readout_sim_time+readout_binsize,readout_binsize)
CV_all           = np.arange(.4,1.5,0.1)

# Weight decay analysis
shift_all = np.array([0,3000])
binsize_cc = 10.
cv_interval = 50000.
cc_interval = 1000000.
rate_interval = 50000.
min_t_decay = 3000000.
max_t_decay = 4000000.
WmaxE_shift = 20.
WminE_shift = 0.

triplets_shift = {'tau_minus' : tau_minus,
                  'tau_plus'  : tau_plus,
                  'tau_x'     : tau_x,
                  'tau_y'     : tau_y,
                  'A2_plus'   : A2_plus,
                  'A3_plus'   : A3_plus,
                  'A2_minus'  : A2_minus,
                  'A3_minus'  : A3_minus,
                  'delay'     : delay,
                  'WmaxE'     : WmaxE_shift,
                  'WminE'     : WminE_shift,
                  'weight'    : WmaxE_shift}
                  
# Plot
fontsize = 7
