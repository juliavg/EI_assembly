import numpy as np
import matplotlib.pyplot as plt
import nest
import scipy.special as sp
from scipy.integrate import quad as INT
import sys

mode = sys.argv[1]

direc = '../data/single_readout/data_'+mode+'/'

n_input  = 160
n_spikes = 1000

dt = 0.1

delay   = 1.5

C_m     = 250.
tau_m   = 20.
tau_psc = 1.5
t_ref   = 2.
E_L     = 0.
V_th    = 20.
V_reset = 10.

# PSP to PSC
sub             = 1. / (tau_psc - tau_m)
pre             = tau_m * tau_psc / C_m * sub
frac            = (tau_m / tau_psc) ** sub
PSC_over_PSP    = 1. / (pre * (frac**tau_m - frac**tau_psc))

J_E = 0.15*PSC_over_PSP

rate = 9.

simtime = 50000.
warmup  = 5000.

tau_rec = 100.   # recovery time
tau_fac = 100.   # facilitation time
U       = 0.02   # facilitation parameter U
A       = 1000.   # PSC weight in pA

mean_noise = 150.
std_noise = 0.

CV_all  = np.arange(.4,1.5,0.1)

final_weight = np.zeros(CV_all.shape[0])
rate_out     = np.zeros(CV_all.shape[0])
rate_in      = np.zeros((CV_all.shape[0],min(n_input,1000)))
cv_in        = np.zeros((CV_all.shape[0],min(n_input,1000)))
cv_out       = np.zeros(CV_all.shape[0])
mean_cc      = np.zeros(CV_all.shape[0])

times_vm = {}
vm       = {}

for cc,CV in enumerate(CV_all):
    shape = 1./(CV**2)
    scale = CV**2/rate

    intervals = np.random.gamma(shape=shape,scale=scale,size=(n_input,n_spikes))          # ISI distribution

    spk_times = np.floor(np.cumsum(intervals,axis=1)*10000)/10.                             # spike times in ms
    spk_times += 1                                         # shifts spike trains by 1ms (spike_generator does not take spikes at 0)


    nest.ResetKernel()
    
    weight_recorder = nest.Create('weight_recorder')

    # Set defaults
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

    nest.CopyModel('static_synapse',
                   'static',
                   {'weight':J_E, 
                    'delay':delay})
                    
    nest.CopyModel("tsodyks_synapse", 
               "stp",
               {"tau_psc": tau_psc,
                "tau_rec": tau_rec,
                "tau_fac": tau_fac,
                "U"      : U,
                "delay"  : 0.1,
                "weight" : A,
                "u"      : 0.0,
                "x"      : 1.0,
                'weight_recorder': weight_recorder[0]})


    # Create nodes
    spike_generator = nest.Create("spike_generator",n_input)
    parrot_neurons  = nest.Create("parrot_neuron",n_input)
    output_neurons  = nest.Create("iaf_psc_exp",2)
    spike_detector  = nest.Create("spike_detector",2)
    multimeter      = nest.Create("multimeter")
    noise           = nest.Create("noise_generator")

    # Set Status
    nest.SetStatus(multimeter,{'record_from':['V_m']})
    nest.SetStatus([output_neurons[1]],'V_th',1000.)
    for ii,sg in enumerate(spike_generator):
        nest.SetStatus([sg],{'spike_times':spk_times[ii,:]})
    nest.SetStatus(noise,{'mean':mean_noise,'std':std_noise})

    # Connect nodes
    nest.Connect(spike_generator,parrot_neurons,'one_to_one')
    nest.Connect(parrot_neurons,[output_neurons[0]],'all_to_all',syn_spec=mode)
    nest.Connect([output_neurons[0]],[spike_detector[0]])
    nest.Connect(parrot_neurons,[spike_detector[1]])
    nest.Connect(noise,[output_neurons[0]])
    
    nest.Connect(parrot_neurons,[output_neurons[1]],'all_to_all',syn_spec=mode)
    nest.Connect(multimeter,[output_neurons[1]])
    nest.Connect(noise,[output_neurons[1]])
    

    # Simulate
    nest.Simulate(simtime)

    # Read data
    events = nest.GetStatus(spike_detector,'events')[0]
    times  = events['times']
    
    events = nest.GetStatus(spike_detector,'events')[1]
    times_in = events['times']
    senders_in = events['senders']
    
    # Output cv and cc
    binsize       = 10.
    bins          = np.arange(0,simtime+binsize,binsize)
    spk_train_out = np.histogram(times,bins=bins)[0]
    cc_all        = np.zeros(min(n_input,1000))
    cv            = []
    rate_temp     = []
    for ii in np.arange(min(n_input,1000)):
        spk_times = times_in[senders_in==np.unique(senders_in)[ii]]
        spk_train_in = np.histogram(spk_times,bins=bins)[0]
        cc_all[ii]   = np.corrcoef(spk_train_out,spk_train_in)[0,1]
        
        isi_in = np.diff(spk_times[spk_times>warmup])
        cv_in[cc,ii] = np.std(isi_in)/np.mean(isi_in)
        rate_in[cc,ii] = len(spk_times)/simtime

    isi_out = np.diff(times[times>warmup])

    wr_status  = nest.GetStatus(weight_recorder,'events')[0]
    wr_times   = wr_status['times']
    wr_weights = wr_status['weights']
    
    mean_weight = np.histogram(wr_times,bins=bins,weights=wr_weights)[0]/np.histogram(wr_times,bins=bins)[0]/n_input
    plt.subplot(2,3,1)
    plt.plot(bins[:-1],mean_weight)
    
    final_weight[cc] = mean_weight[-1] 
    rate_out[cc]     = len(times[times>warmup])/(simtime-warmup)*1000.
    cv_out[cc]       = np.std(isi_out)/np.mean(isi_out)
    mean_cc[cc]      = np.mean(cc_all)
    CV_all[cc]       = CV
    
    events = nest.GetStatus(multimeter,'events')[0]
    times_vm[cc] = events['times']
    vm[cc]       = events['V_m']
    
np.save(direc+"times_vm.npy",times_vm)
np.save(direc+"vm.npy",vm)

np.save(direc+"final_weight.npy",final_weight)
np.save(direc+"rate_out.npy",rate_out)
np.save(direc+"rate_in.npy",rate_in)
np.save(direc+"cv_out.npy",cv_out)
np.save(direc+"cv_in.npy",cv_in)
np.save(direc+"mean_cc.npy",mean_cc)
np.save(direc+"CV_all.npy",CV_all)
