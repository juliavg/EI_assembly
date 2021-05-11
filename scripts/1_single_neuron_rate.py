from importlib import reload 
import numpy as np
import sys
sys.path.append('/home/julia/Documents/iSTDP/paper/main/parameters')

import common
reload(common)
import common as par

mode = sys.argv[1]

sys.path.insert(2,par.path_to_nest[mode])
import nest

direc = par.path_to_data+'single_neuron/data_rate/'

for ss,strength in enumerate(par.stim_strength_all):
    nest.ResetKernel()
    nest.SetDefaults(par.neuron_model,par.neuron_param_dict)

    # Create nodes -------------------------------------------------

    node    = nest.Create(par.neuron_model,par.n_single_neurons)
    parrot  = nest.Create('parrot_neuron',par.n_single_neurons)
    noise_E = nest.Create('poisson_generator', 2)
    noise_I = nest.Create('poisson_generator', 1, params={'rate': float(par.C_EI*par.rho)})
    spikes  = nest.Create('spike_detector')
    weight_recorder = nest.Create('weight_recorder',params={'targets':[node[0]]})

    nest.SetStatus(noise_E,'rate', [par.p_rate,float(par.C_EE*par.rho)])

    # Connect nodes ------------------------------------------------

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
    
    # Simulate -----------------------------------------------------

    nest.Simulate(par.single_sim_time/2)
    connections = nest.GetConnections([noise_E[1]],node)
    nest.SetStatus(connections,params={'weight':strength*par.J_E})
    nest.Simulate(par.single_sim_time/2)

    # Analysis ------------------------------------------------------

    # Rate and CV
    events     = nest.GetStatus(spikes,'events')[0]
    sd_senders = events['senders']
    sd_times   = events['times']
    sd_senders = sd_senders[np.argsort(sd_times)]
    sd_times   = sd_times[np.argsort(sd_times)]

    for nn in node:
        times_neuron = sd_times[sd_senders==nn]
        rate_series  = np.histogram(times_neuron,bins=par.single_bins)[0]/par.single_binsize*1000.
        rate_final   = len(times_neuron[times_neuron>(par.single_sim_time-par.single_rec_final)])/par.single_rec_final*1000.

        isi = np.diff(times_neuron[times_neuron>(par.single_sim_time-par.single_rec_final)])
        cv  = np.std(isi)/np.mean(isi)
        
        extension = "_"+str(nn)+"_"+str(ss)+".npy"
        np.save(direc+"rate_series"+extension,rate_series)
        np.save(direc+"rate_final"+extension,rate_final)
        np.save(direc+"cv"+extension,cv)
        

    # Weights
    wr_status  = nest.GetStatus(weight_recorder,'events')[0]
    wr_times   = wr_status['times']
    wr_weights = wr_status['weights']

    mean_weight = np.histogram(wr_times,bins=par.single_bins,weights=wr_weights)[0]/np.histogram(wr_times,bins=par.single_bins)[0]

    np.save(direc+"mean_weight_"+str(ss)+".npy",mean_weight)
