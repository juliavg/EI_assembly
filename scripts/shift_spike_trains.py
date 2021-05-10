import numpy as np
import matplotlib.pyplot as plt

shift_all = np.array([0,3000,4000,5000])
j_all     = np.array(['1.5','3','5'])
seeds     = np.array([[500,600,700,800,900],[1000,1100,1200,1300,1400],[0,100,200,300,400]])

# Triplets parameters
tau_minus   = 33.7
tau_plus    = 16.8
tau_x       = 101.
tau_y       = 125.
A2_plus     = 7.5e-10
A3_plus     = 9.3e-3
A2_minus    = 7e-3
A3_minus    = 2.3e-4
WmaxE       = 20.
WminE       = 0.
delay       = 1.5       # Synaptic delay (ms)

binsize_cc = 10.
cv_interval = 50000.
rate_interval = 50000.
record_pairs = 5

#####################################################################################

def calculate_weight_triplets(times_spikes_pre,times_spikes_post):
    r1 = 0
    r2 = 0
    o1 = 0
    o2 = 0

    last_spk_pre = 0
    last_spk_post = 0

    weight_offline = {}

    weight = WmaxE
    for ee,spk_time_pre in enumerate(times_spikes_pre):

        # Facilitation due to post spikes from (last spike pre - delay) to (current spike pre - delay)
        mask        = np.where(np.logical_and(times_spikes_post>(last_spk_pre-delay),times_spikes_post<=(spk_time_pre-delay)))
        spikes_post = times_spikes_post[mask]
        
        for oo,spk_time_post in enumerate(spikes_post):
            o1          = o1*np.exp(-(spk_time_post-last_spk_post)/tau_minus) + 1
            o2          = o2*np.exp(-(spk_time_post-last_spk_post)/tau_y)
            r1_at_post  = r1*np.exp(-((spk_time_post+delay)-last_spk_pre)/tau_plus)
            weight      = np.clip(weight + r1_at_post*(A2_plus+A3_plus*o2),a_min=WminE,a_max=WmaxE)

            last_spk_post = spk_time_post*1        
            o2 += 1
            
            weight_offline[spk_time_post] = weight*1
            
        # Depression due to pre spike
        r1            = r1*np.exp(-(spk_time_pre-last_spk_pre)/tau_plus) + 1
        r2            = r2*np.exp(-(spk_time_pre-last_spk_pre)/tau_x)
            
        o1_at_pre_spk = o1*np.exp(-((spk_time_pre-delay)-last_spk_post)/tau_minus)-1*(last_spk_post==(spk_time_pre-delay))

        weight        = np.clip(weight - o1_at_pre_spk*(A2_minus+A3_minus*r2),a_min=WminE,a_max=WmaxE)

        last_spk_pre = spk_time_pre*1
        r2 += 1

        weight_offline[spk_time_pre] = weight*1
        
    return weight_offline

def calculate_cc(times_spikes_pre,times_spikes_post,binsize):
    time_bins        = np.arange(0,min([max(times_spikes_pre),max(times_spikes_post)]),binsize)
    spk_train_source = np.histogram(times_spikes_pre,bins=time_bins)[0]
    spk_train_target = np.histogram(times_spikes_post,bins=time_bins)[0]
    spk_train_source[spk_train_source>1] = 1
    spk_train_target[spk_train_target>1] = 1
    cc = np.corrcoef(spk_train_source,spk_train_target)[0,1]
    return cc

def calculate_cv(times_spikes,cv_interval):
    isi = np.diff(times_spikes[times_spikes>(max(times_spikes)-cv_interval)])
    cv  = np.std(isi)/np.mean(isi)
    return cv
    
def calculate_rate(times_spike,rate_interval):
    rate = len(times_spikes[times_spikes>(max(times_spikes)-rate_interval)])/rate_interval*1000.
    return rate

for jj,J in enumerate(j_all):
    # 0: rate_pre, 1: rate_post, 2: cv_pre, 3: cv_post, 4: cc, 5: slope
    data_all = np.zeros((record_pairs*len(seeds[jj]),len(shift_all),6))
    weight_offline_save = {}
    
    ii = 0
    for seed in seeds[jj]:
    
        direc = '../data/assembly/plastic/J'+J+'/'+str(seed)+'/'

        sources      = np.load(direc+"sources.npy")
        targets      = np.load(direc+"targets.npy")
    
        for source,target in list(zip(sources,targets)):
            
            for ss,shift in enumerate(shift_all):
            
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

                times_spikes_pre = times_spikes_pre[(times_spikes_pre>3000000.) & (times_spikes_pre<4000000.)]
                times_spikes_post = times_spikes_post[(times_spikes_post>3000000.) & (times_spikes_post<4000000.)]

                reference_time = min([np.min(times_spikes_pre),np.min(times_spikes_post)])-1.
                times_spikes_pre     -= reference_time
                times_spikes_post    -= reference_time

                weight_offline = calculate_weight_triplets(times_spikes_pre,times_spikes_post)
                
                # 0: rate_pre, 1: rate_post, 2: cv_pre, 3: cv_post, 4: cc, 5: slope
                data_all[ii,ss,0] = calculate_rate(times_spikes_pre,rate_interval)
                data_all[ii,ss,1] = calculate_rate(times_spikes_post,rate_interval)
                data_all[ii,ss,2] = calculate_cv(times_spikes_pre,cv_interval)
                data_all[ii,ss,3] = calculate_cv(times_spikes_post,cv_interval)
                data_all[ii,ss,4] = calculate_cc(times_spikes_pre,times_spikes_post,binsize_cc)
                data_all[ii,ss,5] = np.polyfit(list(weight_offline.keys()),list(weight_offline.values()),1)[0]
                
            weight_offline_save[ii] = weight_offline

            ii += 1
            
    np.save('../data/assembly/plastic/J'+J+'/weight_offline_'+str(J)+'.npy',weight_offline_save)            
    np.save('../data/assembly/plastic/J'+J+'/data_triplets_'+str(J)+'.npy',data_all)
