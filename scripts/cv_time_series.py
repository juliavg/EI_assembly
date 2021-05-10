import numpy as np
import matplotlib.pyplot as plt

rate_bin = 5000
cv_bin  = 40000

#j_all     = np.array(['1.25','1.5','1.75','2','3','4'])
j_all = np.array(['1.5','5'])
seeds = np.array([[500,600,700,800,900],[0,100,200,300,400]])
colors = ['b','yellow','r','g','m','k']

for jj,J in enumerate(j_all):
    direc = 'J'+J+'/'+str(seeds[jj][0])+'/'

    times_spikes = np.array([])
    senders      = np.array([])
    
    for label in ['_post','_decay']:
        events       = np.load(direc+"spk_all_sim"+label+".npy",allow_pickle=True)
        events       = events[()]
        times_spikes = np.concatenate((times_spikes,events['times']))
        senders      = np.concatenate((senders,events['senders']))
    source       = np.load(direc+"sources.npy")[0]
    target       = np.load(direc+"targets.npy")[0]
    
    #times_spikes = times_spikes[times_spikes>100000.]
    times_spikes -= min(times_spikes)
    
    rate_bins = np.arange(min(times_spikes),max(times_spikes),rate_bin)
    rate_source = np.histogram(times_spikes[senders==source],bins=rate_bins)[0]/rate_bin*1000.
    rate_target = np.histogram(times_spikes[senders==target],bins=rate_bins)[0]/rate_bin*1000.
    
    alpha = (jj+1)/len(j_all)
    plt.figure(1)
    plt.subplot(2,3,jj+1)
    plt.plot(rate_bins[:-1],rate_source,color='b',alpha=alpha)
    plt.plot(rate_bins[:-1],rate_target,color='r',alpha=alpha)
    
    cv_bins = np.arange(min(times_spikes),max(times_spikes),cv_bin)
    cv_source = np.zeros(len(cv_bins))
    cv_target = np.zeros(len(cv_bins))
    for tt,time in enumerate(cv_bins):
        times_temp = times_spikes[senders==source]
        times_temp = times_temp[(times_temp>time)&(times_temp<(time+cv_bin))]
        isi        = np.diff(times_temp)
        cv_source[tt] = np.std(isi)/np.mean(isi)

        times_temp = times_spikes[senders==target]
        times_temp = times_temp[(times_temp>time)&(times_temp<(time+cv_bin))]
        isi        = np.diff(times_temp)
        cv_target[tt] = np.std(isi)/np.mean(isi)
        
    plt.figure(2)
    plt.plot(cv_bins,cv_source,color=colors[jj],label=str(J))
    plt.plot(cv_bins,cv_target,color=colors[jj])
    
        
    plt.figure(3)
    plt.plot(jj,np.mean(cv_source),'.')
    plt.plot(jj,np.mean(cv_target),'.')
    
plt.legend()
plt.show()
