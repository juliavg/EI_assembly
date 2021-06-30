from importlib import reload 
import numpy as np
import h5py as h5
import sys
direc = sys.argv[0].split('scripts')[0]
where = sys.argv[1]
sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f
import nest

# Create data file
data     = h5.File(par.path_to_data[where]+'data_single_neuron.hdf5','a')
data_sim = data.require_group('simulation')

for ss,strength in enumerate(par.stim_strength_all):
    strength_group = data_sim.require_group(str(strength))

    # Initialize NEST kernel
    nest.ResetKernel()
    nest.SetDefaults(par.neuron_model,par.neuron_param_dict)

    # Create nodes
    node    = nest.Create(par.neuron_model,par.n_single_neurons)
    parrot  = nest.Create('parrot_neuron',par.n_single_neurons)
    noise_E = nest.Create('poisson_generator', 2)
    noise_I = nest.Create('poisson_generator', 1, params={'rate': float(par.C_EI*par.rho)})
    spikes  = nest.Create('spike_detector')
    weight_recorder = nest.Create('weight_recorder',params={'targets':[node[0]]})
    nest.SetStatus(noise_E,'rate', [par.p_rate,float(par.C_EE*par.rho)])

    # Connect nodes
    nest.CopyModel('static_synapse',
                   'excitatory',
                   {'weight' : par.J_E, 
                    'delay'  : par.delay})

    nest.CopyModel('static_synapse',
                   'external',
                   {'weight' : par.J_ext, 
                    'delay'  : par.delay})

    nest.CopyModel('vogels_sprekeler_synapse',
                   'plastic_inhibitory',
                   {'tau'               : par.tau_stdp, 
                    'Wmax'              : -par.WmaxI,
                    'eta'               : par.eta,
                    'alpha'             : par.alpha,
                    'weight'            : -.1,
                    'weight_recorder'   : weight_recorder[0]})

    nest.Connect([noise_E[0]], node,'all_to_all','external')
    nest.Connect([noise_E[1]], node,'all_to_all','excitatory')
    nest.Connect(parrot,node,'one_to_one','plastic_inhibitory')
    nest.Connect(noise_I, parrot)
    nest.Connect(node, spikes)
    
    # Simulate
    nest.Simulate(par.single_sim_time/2)
    connections = nest.GetConnections([noise_E[1]],node)
    nest.SetStatus(connections,params={'weight':strength*par.J_E})
    nest.Simulate(par.single_sim_time/2)

    # Analysis
    # Rate and CV
    events     = nest.GetStatus(spikes,'events')[0]
    sd_senders = events['senders']
    sd_times   = events['times']
    sd_senders = sd_senders[np.argsort(sd_times)]
    sd_times   = sd_times[np.argsort(sd_times)]

    rate_series = f.rate_time_series(sd_times,par.single_bins,par.n_single_neurons)
    f.save_to_group(strength_group,rate_series,'rate_series')

    rate_final = np.zeros(len(node))
    cv_all     = np.zeros(len(node))
    for idx,nn in enumerate(node):
        times_neuron    = sd_times[sd_senders==nn]
        times_neuron    = times_neuron[times_neuron>(par.single_sim_time-par.single_rec_final)]
        rate_final[idx] = f.rate_mean(times_neuron,par.single_rec_final,1)
        cv_all[idx]     = f.cv(times_neuron)
    f.save_to_group(strength_group,rate_final,'rate_final')
    f.save_to_group(strength_group,cv_all,'cv_all')

    # Weights
    wr_status  = nest.GetStatus(weight_recorder,'events')[0]
    wr_times   = wr_status['times']
    wr_weights = wr_status['weights']
    mean_weight = np.histogram(wr_times,bins=par.single_bins,weights=wr_weights)[0]/np.histogram(wr_times,bins=par.single_bins)[0]
    f.save_to_group(strength_group,mean_weight,'mean_weight')

data.close()
