import numpy as np
import matplotlib.pyplot as plt

speed_up  = 1

shift_all = np.arange(0,10000,10)
j_all     = np.array(['1.5','5'])
seeds     = np.array([[500,600,700,800,900],[0,100,200,300,400]])

colors = ['b','yellow','r','g','m','k']

simtime = 1000000.

#####################################################################################3

cc_bin = 100
time_bins = np.arange(0,simtime,cc_bin)
for jj,J in enumerate(j_all):
    direc = 'J'+J+'/'+str(seeds[jj][4])+'/'

    sources      = np.load(direc+"sources.npy")
    targets      = np.load(direc+"targets.npy")

    for source,target in list(zip(sources,targets)):
        
        for ss,shift in enumerate(shift_all):

            if ss == 0:
                alpha = 1.
            else:
                alpha = 0.3


            times_spikes = np.array([])
            senders      = np.array([])
            
            for label in ['_post','_decay']:
                events       = np.load(direc+"spk_all_sim"+label+".npy",allow_pickle=True)
                events       = events[()]
                times_spikes = np.concatenate((times_spikes,events['times']))
                senders      = np.concatenate((senders,events['senders']))
            
            times_spikes_pre = times_spikes[senders==source]
            times_spikes_post = times_spikes[senders==target]
            times_spikes_post += shift

            times_spikes_pre = times_spikes_pre[times_spikes_pre>3000000.]
            times_spikes_post = times_spikes_post[times_spikes_post>3000000.]

            reference_time = min([np.min(times_spikes_pre),np.min(times_spikes_post)])-1.
            times_spikes_pre     -= reference_time
            times_spikes_post    -= reference_time
                
            spk_train_source = np.histogram(times_spikes_pre,bins=time_bins)[0]
            spk_train_target = np.histogram(times_spikes_post,bins=time_bins)[0]
            cc               = np.corrcoef(spk_train_source,spk_train_target)[0,1]
                        
            plt.subplot(2,2,jj+1)
            plt.plot(shift,cc,'.',color=colors[jj])
            
plt.subplot(2,2,1)
plt.axhline(0,color='k')
plt.subplot(2,2,2)
plt.axhline(0,color='k')

            

plt.show()     
