from importlib import reload 
import numpy as np
import h5py as h5
import sys
sys.path.append('/home/julia/Documents/iSTDP/paper/main/parameters')

import common
reload(common)
import common as par

mode = sys.argv[1]

sys.path.insert(2,par.path_to_nest[mode])
import nest

data     = h5.File(par.path_to_data+'data_single_neuron.hdf5','r+')
data_sim = data['simulation']

for ss,strength in enumerate(par.stim_strength_all):
    data_strength = data_sim[str(strength)]

    mean_weight = data_strength['mean_weight']
    inh_weight  = np.mean(mean_weight[-int(mean_weight.shape[0]/4):])

    # Set parameters of the NEST simulation kernel
    nest.ResetKernel()
    nest.SetDefaults(par.neuron_model,par.neuron_param_dict)

    # Create nodes -------------------------------------------------

    node       = nest.Create(par.neuron_model)
    parrot     = nest.Create('parrot_neuron')
    noise_E    = nest.Create('poisson_generator', 2)
    noise_I    = nest.Create('poisson_generator', 1, params={'rate': float(par.C_EI*par.rho)})
    multimeter = nest.Create('multimeter')
    
    nest.SetStatus(noise_E,'rate', [par.p_rate,float(par.C_EE*par.rho)])
    nest.SetStatus(node,params={'V_th':1000.})
    nest.SetStatus(multimeter,params={'record_from':['V_m']})

    # Connect nodes ------------------------------------------------

    nest.CopyModel('static_synapse',
                   'excitatory',
                   {'weight' : strength*par.J_E, 
                    'delay'  : par.delay})

    nest.CopyModel('static_synapse',
                   'external',
                   {'weight' : par.J_ext, 
                    'delay'  : par.delay})

    nest.CopyModel('static_synapse',
                   'inhibitory',
                   {'weight' : inh_weight, 
                    'delay'  : par.delay})

    nest.Connect([noise_E[0]], node,'all_to_all','external')
    nest.Connect([noise_E[1]], node,'all_to_all','excitatory')
    nest.Connect(noise_I,node,'all_to_all','inhibitory')
    nest.Connect(multimeter, node)

    # Simulate -----------------------------------------------------

    nest.Simulate(par.single_sim_time)

    # Analysis ------------------------------------------------------
    events  = nest.GetStatus(multimeter,'events')[0]
    time_vm = events['times']
    vm      = events['V_m']

    data_strength.require_dataset('time_vm',time_vm.shape,dtype=time_vm.dtype)
    data_strength.require_dataset('vm',vm.shape,dtype=vm.dtype)
    data_strength['time_vm'][...] = time_vm
    data_strength['vm'][...] = vm
    
data.close()
