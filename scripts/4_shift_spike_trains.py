import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from importlib import reload
import sys

direc = sys.argv[0].split('scripts')[0]

sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f

data_file = h5.File(direc+"data_assembly.hdf5","r+")
data = data_file['plastic']

#####################################################################################

for jj,J in enumerate(list(data.keys())):

    data_J = data[J]
    seeds  = list(data_J.keys())
    
    # 0: rate_pre, 1: rate_post, 2: cv_pre, 3: cv_post, 4: cc, 5: slope
    data_all = np.zeros((par.rec_weight_from*len(seeds),len(par.shift_all),6))
    
    weight_offline_save = {}
    time_offline_save = {}
    
    ii = 0
    for seed in seeds:
    
        data_seed = data_J[seed]
        sources   = np.array(data_seed['sources'])
        targets   = np.array(data_seed['targets'])
        data_step = data_seed['steps']
    
        for source,target in list(zip(sources,targets)):
            
            for ss,shift in enumerate(par.shift_all):
            
                times_spikes = np.array([])
                senders      = np.array([])
                
                for label in par.labels[2:]:
                    times_spikes = np.concatenate((times_spikes,np.array(data_step[label+'/all_sim/times'])))
                    senders      = np.concatenate((senders,np.array(data_step[label+'/all_sim/senders'])))

                times_spikes_pre = times_spikes[senders==source]
                times_spikes_post = times_spikes[senders==target]
                times_spikes_post += shift

                times_spikes_pre = times_spikes_pre[(times_spikes_pre>par.min_t_decay) & (times_spikes_pre<par.max_t_decay)]
                times_spikes_post = times_spikes_post[(times_spikes_post>par.min_t_decay) & (times_spikes_post<par.max_t_decay)]

                reference_time = min([np.min(times_spikes_pre),np.min(times_spikes_post)])-1.
                times_spikes_pre     -= reference_time
                times_spikes_post    -= reference_time

                weight_offline = f.calculate_weight_triplets(times_spikes_pre,times_spikes_post,par.triplets_shift)
                
                # 0: rate_pre, 1: rate_post, 2: cv_pre, 3: cv_post, 4: cc, 5: slope
                data_all[ii,ss,0] = f.rate_mean(times_spikes_pre[times_spikes_pre>(max(times_spikes_pre)-par.rate_interval)],par.rate_interval,1)
                data_all[ii,ss,1] = f.rate_mean(times_spikes_post[times_spikes_post>(max(times_spikes_post)-par.rate_interval)],par.rate_interval,1)
                data_all[ii,ss,2] = f.cv(times_spikes_pre[times_spikes_pre>(max(times_spikes_pre)-par.cv_interval)])
                data_all[ii,ss,3] = f.cv(times_spikes_post[times_spikes_post>(max(times_spikes_post)-par.cv_interval)])
                data_all[ii,ss,4] = f.cc(times_spikes_pre,times_spikes_post,par.binsize_cc)
                data_all[ii,ss,5] = np.polyfit(list(weight_offline.keys()),list(weight_offline.values()),1)[0]
            
                
            weight_offline_save[ii] = weight_offline

            ii += 1
            
    np.save('../data/assembly/plastic/J'+J+'/weight_offline_'+str(J)+'.npy',weight_offline_save) 
    np.save('../data/assembly/plastic/J'+J+'/data_triplets_'+str(J)+'.npy',data_all)
data_file.close()
