import numpy as np
from importlib import reload  

import parameters
reload(parameters)
import parameters as par

import sys
sys.path.insert(2,par.path_to_nest) 
import nest

master_seed = int(sys.argv[1])
direc       = '/home/jgallina/homedir/iSTDP/static/'+str(master_seed)+'/'

#####################################################################################3

# Random seeds
pyrngs    = [np.random.RandomState(s) for s in range(master_seed, master_seed+par.n_threads)]
grng_seed = master_seed+par.n_threads
rng_seeds = range(master_seed+par.n_threads+1, master_seed+2*par.n_threads+1)

# Set parameters of the NEST simulation kernel
nest.ResetKernel()
nest.SetKernelStatus({'print_time'          : par.print_time,
                      'local_num_threads'   : par.n_threads,
                      'grng_seed'           : grng_seed,
                      'rng_seeds'           : rng_seeds})


# Set models default -------------------------------------------

weight_E = nest.Create('weight_recorder')
weight_I = nest.Create('weight_recorder')

nest.SetDefaults(par.neuron_model,par.neuron_param_dict)

nest.CopyModel('static_synapse',
               'excitatory_noise',
               {'weight': par.J_ext, 
                'delay' : par.delay})

nest.CopyModel('static_synapse',
               'excitatory',
               {'weight': par.J_E, 
                'delay' : par.delay})

nest.CopyModel('static_synapse',
               'inhibitory',
               {'weight': par.J_I, 
                'delay' : par.delay})

nest.CopyModel('vogels_sprekeler_synapse',
               'iSTDP',
               {'tau'               : par.tau_stdp, 
                'Wmax'              : -par.WmaxI,
                'eta'               : par.eta,
                'alpha'             : par.alpha,
                'weight'            : par.J_I,
                'weight_recorder'   : weight_I[0]})

nest.CopyModel('stdp_triplet_synapse', 
               'triplets',
               {'delay'             : par.delay,
                'Aminus'            : par.A2_minus,
                'Aminus_triplet'    : par.A3_minus,
                'Aplus'             : par.A2_plus,
                'Aplus_triplet'     : par.A3_plus,
                'tau_plus'          : par.tau_plus,
                'tau_plus_triplet'  : par.tau_x,
                'Wmax'              : par.WmaxE,
                'Wmin'              : par.WminE,
                'weight'            : par.WminE,
                'weight_recorder'   : weight_E[0]})


nest.CopyModel('tsodyks_synapse', 
               'stp',
               {'delay'     : par.delay,
                'tau_psc'   : par.tau_psc,
                'tau_rec'   : par.tau_rec,
                'tau_fac'   : par.tau_fac,
                'U'         : par.U,
                'delay'     : 0.1,
                'weight'    : par.A,
                'u'         : 0.0,
                'x'         : 1.0})

# Create nodes -------------------------------------------------

neurons                 = nest.Create('iaf_psc_exp', par.N_neurons)
neurons_E               = neurons[:par.N_E]
neurons_I               = neurons[par.N_E:]
neurons_E_assembly      = neurons_E[:par.assembly_size]
neurons_E_non_assembly  = neurons_E[par.assembly_size:]
readout                 = nest.Create(par.neuron_model,4)

spk_all_neuron  = nest.Create('spike_detector')
spk_all_sim     = nest.Create('spike_detector')
spk_readout     = nest.Create('spike_detector')
external_input  = nest.Create('poisson_generator', 2)
nest.SetStatus(external_input,'rate',par.p_rate)
noise           = nest.Create('noise_generator')
nest.SetStatus(noise,{'mean':par.noise_mean,'std':par.noise_std})



# Connect nodes ------------------------------------------------

def random_connect_readout(readout_pop,input_pop,n_input,exc_synapse_model):
    node_info   = nest.GetStatus(readout_pop)
    local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
    for gid,vp in local_nodes:
        input_pool = list(pyrngs[vp].choice(input_pop,n_input,replace=False))
        nest.Connect(input_pool,[gid],'all_to_all', syn_spec=exc_synapse_model)

nest.Connect(neurons_E, neurons_E,
             {'rule'        : 'fixed_indegree', 
              'indegree'    : par.C_EE,
              'autapses'    : par.autapses,
              'multapses'   : par.multapses},
              'excitatory')

nest.Connect(neurons_E, neurons_I,
             {'rule'        : 'fixed_indegree', 
              'indegree'    : par.C_IE,
              'multapses'   : par.multapses},
              'excitatory')

nest.Connect(neurons_I, neurons_I,
             {'rule'        : 'fixed_indegree', 
              'indegree'    : par.C_II,
              'autapses'    : par.autapses,
              'multapses'   : par.multapses},
              'inhibitory')
             
nest.Connect(neurons_I, neurons_E,
             {'rule'        : 'fixed_indegree', 
              'indegree'    : par.C_EI,
              'multapses'   : par.multapses},
              'iSTDP')

random_connect_readout([readout[0]],neurons_E_assembly,par.n_input_readout,'stp')
random_connect_readout([readout[1]],neurons_E_non_assembly,par.n_input_readout,'stp')
random_connect_readout([readout[2]],neurons_E_assembly,par.n_input_readout,'excitatory')
random_connect_readout([readout[3]],neurons_E_non_assembly,par.n_input_readout,'excitatory')

nest.Connect([external_input[0]],neurons_E_assembly,syn_spec='excitatory_noise')
nest.Connect([external_input[1]],neurons_E_non_assembly+neurons_I,syn_spec='excitatory_noise')

# Connect weight recorder
def connect_weight_recorder(weight_recorder,pop_pre,pop_post,record_from):
    node_info   = nest.GetStatus(weight_recorder)
    local_nodes = [(ni['global_id'], ni['vp']) for ni in node_info if ni['local']]
    connections = nest.GetConnections(pop_pre,pop_post)
    for gid,vp in local_nodes:
        index   = pyrngs[vp].choice(np.arange(len(connections)),record_from,replace=False)
        sources = np.array(nest.GetStatus(connections,'source'))[index]
        targets = np.array(nest.GetStatus(connections,'target'))[index]
        nest.SetStatus(weight_recorder,{'senders':sources,'targets':targets})
        return sources,targets

connect_weight_recorder(weight_I,neurons_I,neurons_E_non_assembly,par.rec_weight_from)
connect_weight_recorder(weight_I,neurons_I,neurons_E_assembly,par.rec_weight_from)
connect_weight_recorder(weight_E,neurons_E_non_assembly,neurons_E_non_assembly,par.rec_weight_from)
sources,targets = connect_weight_recorder(weight_E,neurons_E_assembly,neurons_E_assembly,par.rec_weight_from)
np.save(direc+"sources.npy",sources)
np.save(direc+"targets.npy",targets)

nest.Connect(noise,readout)

# Connect spike detector
nest.Connect(tuple(np.unique(list(sources)+list(targets)))+neurons_E_non_assembly[:par.rec_spk_from]+neurons_I[:par.rec_spk_from],spk_all_sim)
nest.Connect(neurons,spk_all_neuron)
nest.Connect(readout,spk_readout)

# Simulate -----------------------------------------------------

def simulation_cycle(direc,global_time,simulation_time,label):

    nest.Simulate(simulation_time)

    # Save ---------------------------------------------------------

    extension = "_"+label+".npy"

    events = nest.GetStatus(spk_all_sim,'events')[0]
    np.save(direc+"spk_all_sim"+extension,events)
    
    events = nest.GetStatus(spk_all_neuron,'events')[0]
    np.save(direc+"spk_all_neuron"+extension,events)
    
    events = nest.GetStatus(spk_readout,'events')[0]
    np.save(direc+"spk_readout"+extension,events)
    
    # Weights
    events = nest.GetStatus(weight_E,'events')[0]
    np.save(direc+"weight_E"+extension,events)
    
    events = nest.GetStatus(weight_I,'events')[0]
    np.save(direc+"weight_I"+extension,events)
    
    connections = nest.GetConnections(neurons,neurons)
    np.save(direc+"sources"+extension,np.array(nest.GetStatus(connections,'source')))
    np.save(direc+"targets"+extension,np.array(nest.GetStatus(connections,'target')))
    np.save(direc+"weights"+extension,np.array(nest.GetStatus(connections,'weight')))


    # Clean ---------------------------------------------------------
    nest.SetStatus(spk_all_sim,'n_events',0)
    nest.SetStatus(spk_all_neuron,'n_events',0)
    nest.SetStatus(spk_readout,'n_events',0)
    nest.SetStatus(weight_E,'n_events',0)
    nest.SetStatus(weight_I,'n_events',0)
    
    return global_time+simulation_time


nest.SetStatus(spk_all_neuron,{'start':par.grow_time-par.save_for,'stop':par.grow_time+par.stimulation_time})

global_time = 0.
global_time = simulation_cycle(direc,global_time,par.grow_time,'grow')

connections = nest.GetConnections(neurons_E[:par.assembly_size],neurons_E[:par.assembly_size])
nest.SetStatus(connections,'weight',par.jee_strength*par.J_E)
global_time = simulation_cycle(direc,global_time,par.stimulation_time,'stim')

stop_time = par.grow_time+par.stimulation_time+par.post_stimulation_time
nest.SetStatus(spk_all_neuron,{'start':stop_time-par.save_for,'stop':stop_time})

nest.SetStatus([external_input[0]],'rate',par.p_rate)
global_time = simulation_cycle(direc,global_time,par.post_stimulation_time,'post')

simulation_cycle(direc,global_time,par.decay_time,'decay')
