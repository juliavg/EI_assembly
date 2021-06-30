import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
from importlib import reload
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

data_file = h5.File(par.path_to_data[where]+"data_assembly.hdf5","r+")
data      = data_file['plastic']

#####################################################################################

for jj,J in enumerate(list(data.keys())):

    data_decay    = data[J].require_group('weight_decay')
    data_original = data_decay.require_group('original')
    data_shifted  = data_decay.require_group('shifted')
    data_offline  = data_shifted.require_group('offline_weight')
    data_J        = data[J+'/seeds']
    seeds         = list(data_J.keys())
    
    cv_all    = np.zeros((par.rec_weight_from*len(seeds),len(par.shift_all)))
    cc_all    = np.zeros((par.rec_weight_from*len(seeds),len(par.shift_all)))
    rate_all  = np.zeros((par.rec_weight_from*len(seeds),len(par.shift_all)))
    slope_all = np.zeros((par.rec_weight_from*len(seeds),len(par.shift_all)))
    
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
                    
                idx          = np.argsort(times_spikes)
                senders      = senders[idx]
                times_spikes = times_spikes[idx]

                times_spikes_pre   = times_spikes[senders==source]
                times_spikes_post  = times_spikes[senders==target]
                times_spikes_post += shift

                times_spikes_pre  = times_spikes_pre[(times_spikes_pre>par.min_t_decay) & (times_spikes_pre<par.max_t_decay)]
                times_spikes_post = times_spikes_post[(times_spikes_post>par.min_t_decay) & (times_spikes_post<par.max_t_decay)]

                reference_time     = min([np.min(times_spikes_pre),np.min(times_spikes_post)])-1.
                times_spikes_pre  -= reference_time
                times_spikes_post -= reference_time

                weight_offline,time_offline = f.calculate_weight_triplets(times_spikes_pre,times_spikes_post,par.triplets_shift)
                
                idx = np.argsort(time_offline)
                weight_offline = weight_offline[idx]
                time_offline = time_offline[idx]
                
                r_pre  = f.rate_mean(times_spikes_pre[times_spikes_pre>(max(times_spikes_pre)-par.rate_interval)],par.rate_interval,1)
                r_post = f.rate_mean(times_spikes_post[times_spikes_post>(max(times_spikes_post)-par.rate_interval)],par.rate_interval,1)
                rate_all[ii,ss] = np.mean([r_pre,r_post])
                cv_pre  = f.cv(times_spikes_pre[times_spikes_pre>(max(times_spikes_pre)-par.cv_interval)])
                cv_post = f.cv(times_spikes_post[times_spikes_post>(max(times_spikes_post)-par.cv_interval)])
                cv_all[ii,ss] = np.mean([cv_pre,cv_post])
                cc_all[ii,ss] = f.cc(times_spikes_pre,times_spikes_post,par.binsize_cc)
                slope_all[ii,ss] = np.polyfit(time_offline,weight_offline,1)[0]

            data_pair = data_offline.require_group(str(ii))
            f.save_to_group(data_pair,time_offline,'times')
            f.save_to_group(data_pair,weight_offline,'weights')

            ii += 1

    f.save_to_group(data_original,cv_all[:,0],'cv')
    f.save_to_group(data_original,cc_all[:,0],'cc')
    f.save_to_group(data_original,slope_all[:,0],'slope')
    f.save_to_group(data_shifted,cv_all[:,1],'cv')
    f.save_to_group(data_shifted,cc_all[:,1],'cc')
    f.save_to_group(data_shifted,slope_all[:,1],'slope')
    
data_file.close()
