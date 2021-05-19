import numpy as np
import matplotlib.pyplot as plt
import scipy.special as sp
import h5py as h5
from scipy.integrate import quad as INT
from importlib import reload 
import sys

direc = sys.argv[0].split('scripts')[0]
where = sys.argv[1]
synapse_type = sys.argv[2]

sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f

sys.path.insert(2,par.path_to_nest[where])
import nest

# Create data file
data      = h5.File(par.path_to_data[where]+'data_single_readout.hdf5','a')
data_mode = data.require_group(synapse_type)

# Initialize arrays
rate_out = np.zeros(par.CV_all.shape[0])
cv_in    = np.zeros((par.CV_all.shape[0],par.assembly_size))
mean_vm  = np.zeros(par.CV_all.shape[0])
std_vm   = np.zeros(par.CV_all.shape[0])

for cc,CV in enumerate(par.CV_all):
    # Generate spike trains
    spk_times = f.generate_spk_train(CV,par.rho,par.assembly_size,par.readout_n_spikes)
    spk_times += 1                                         # shifts spike trains by 1ms (spike_generator does not take spikes at 0)

    # Initialize NEST kernel
    nest.ResetKernel()
    weight_recorder = nest.Create('weight_recorder')

    # Set defaults
    nest.SetDefaults(par.neuron_model,par.neuron_param_dict)

    nest.CopyModel('static_synapse',
                   'static',
                   {'weight':par.J_E, 
                    'delay':par.delay})
                    
    nest.CopyModel("tsodyks_synapse", 
               "stp",
               {"tau_psc": par.tau_psc,
                "tau_rec": par.tau_rec,
                "tau_fac": par.tau_fac,
                "U"      : par.U,
                "delay"  : par.delay,
                "weight" : par.A,
                "u"      : 0.0,
                "x"      : 1.0,
                'weight_recorder': weight_recorder[0]})

    # Create nodes
    spike_generator = nest.Create("spike_generator",par.assembly_size)
    parrot_neurons  = nest.Create("parrot_neuron",par.assembly_size)
    output_neurons  = nest.Create("iaf_psc_exp",2)
    spike_detector  = nest.Create("spike_detector",2)
    multimeter      = nest.Create("multimeter")
    noise           = nest.Create("noise_generator")

    # Set Status
    nest.SetStatus(multimeter,{'record_from':['V_m']})
    nest.SetStatus([output_neurons[1]],'V_th',1000.)
    for ii,sg in enumerate(spike_generator):
        nest.SetStatus([sg],{'spike_times':spk_times[ii,:]})
    nest.SetStatus(noise,{'mean':par.noise_mean_std[0],'std':par.noise_mean_std[1]})

    # Connect nodes
    nest.Connect(spike_generator,parrot_neurons,'one_to_one')
    nest.Connect(parrot_neurons,[output_neurons[0]],'all_to_all',syn_spec=synapse_type)
    nest.Connect([output_neurons[0]],[spike_detector[0]])
    nest.Connect(parrot_neurons,[spike_detector[1]])
    nest.Connect(noise,[output_neurons[0]])
    nest.Connect(parrot_neurons,[output_neurons[1]],'all_to_all',syn_spec=synapse_type)
    nest.Connect(multimeter,[output_neurons[1]])
    nest.Connect(noise,[output_neurons[1]])

    # Simulate
    nest.Simulate(par.readout_sim_time)

    # Read data
    events     = nest.GetStatus(spike_detector,'events')[0]
    times      = events['times']
    events     = nest.GetStatus(spike_detector,'events')[1]
    times_in   = events['times']
    senders_in = events['senders']
    
    # Output rate
    for ii in np.arange(par.assembly_size):
        spk_times    = times_in[senders_in==np.unique(senders_in)[ii]]
        spk_times    = spk_times[spk_times>par.readout_warmup]
        cv_in[cc,ii] = f.cv(spk_times)

    wr_status  = nest.GetStatus(weight_recorder,'events')[0]
    wr_times   = wr_status['times']
    wr_weights = wr_status['weights']
    
    mean_weight = np.histogram(wr_times,bins=par.readout_bins,weights=wr_weights)[0]/np.histogram(wr_times,bins=par.readout_bins)[0]/par.assembly_size

    rate_out[cc] = f.rate_mean(times[times>par.readout_warmup],(par.readout_sim_time-par.readout_warmup),1)
    
    events = nest.GetStatus(multimeter,'events')[0]
    vm     = events['V_m']
    tvm    = events['times']
    mean_vm[cc] = np.mean(vm[tvm>par.readout_warmup])
    std_vm[cc]  = np.std(vm[tvm>par.readout_warmup])

f.save_to_group(data_mode,rate_out,'rate_out')
f.save_to_group(data_mode,cv_in,'cv_in')
f.save_to_group(data_mode,mean_vm,'mean_vm')
f.save_to_group(data_mode,std_vm,'std_vm')

data.close()
