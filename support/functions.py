import numpy as np
import h5py as h5
from scipy.integrate import quad as INT
from scipy.integrate import quad
import scipy.special as sp
import matplotlib.patches as patches

# General
def save_to_group(group,dataset,label):
    group.require_dataset(label,dataset.shape,dtype=dataset.dtype)
    group[label][...] = dataset
    
# Theoretical CV - Brunel, 2000
def integrand(u):
    return sp.erfcx(-u)

def integrand1(u):
    return np.exp(-u**2)*sp.erfcx(-u)**2

def integral1(x):
    return quad(integrand1,-np.inf,x)[0]

def integrand2(x):
    return np.exp(x**2)*integral1(x)

def integral2(yr,yth):
    return quad(integrand2,yr,yth)[0]

def calculate_cv(rate,mean_vm,std_vm,tau,V_reset,V_th):
    return np.sqrt(2*np.pi*(rate*tau)**2*integral2((V_reset-mean_vm)/std_vm,(V_th-mean_vm)/std_vm))

# Theoretical Rate - Brunel, 2000
def integrand(u):
    return sp.erfcx(-u)

def calculate_rate(mean_vm,std_vm,t_ref,tau,V_reset,V_th):
    return (t_ref+tau*np.sqrt(np.pi)*INT(integrand,(V_reset-mean_vm)/std_vm,(V_th-mean_vm)/std_vm)[0])**-1
    
    
# Simulation rate, cv and cc
def rate_mean(spk_times,period,n_neurons):
    return len(spk_times)/n_neurons/period*1000.

def rate_time_series(spk_times,bins,n_neurons):
    binsize = np.diff(bins)[0]
    return np.histogram(spk_times,bins=bins)[0]/n_neurons/binsize*1000.

def cv(spk_times):
    isi = np.diff(spk_times)
    return np.std(isi)/np.mean(isi)

def cc(times_spikes_pre,times_spikes_post,binsize):
    time_bins = np.arange(min([min(times_spikes_pre),min(times_spikes_post)]),
                          max([max(times_spikes_pre),max(times_spikes_post)])+binsize,
                          binsize)
    spk_train_source = np.histogram(times_spikes_pre,bins=time_bins)[0]
    spk_train_target = np.histogram(times_spikes_post,bins=time_bins)[0]
    spk_train_source[spk_train_source>1] = 1
    spk_train_target[spk_train_target>1] = 1
    cc = np.corrcoef(spk_train_source,spk_train_target)[0,1]
    return cc

# Synthetic spike trains and offline weight decay
def calculate_weight_triplets(times_spikes_pre,times_spikes_post,triplets):
    r1 = 0
    r2 = 0
    o1 = 0
    o2 = 0

    last_spk_pre = 0
    last_spk_post = 0

    weight_offline = {}
    
    ii = 0
    weight = triplets['weight']*1
    for ee,spk_time_pre in enumerate(times_spikes_pre):

        # Facilitation due to post spikes from (last spike pre - delay) to (current spike pre - delay)
        mask = np.where(np.logical_and(times_spikes_post>(last_spk_pre-triplets['delay']),times_spikes_post<=(spk_time_pre-triplets['delay'])))
        spikes_post = times_spikes_post[mask]
        
        for oo,spk_time_post in enumerate(spikes_post):
            o1          = o1*np.exp(-(spk_time_post-last_spk_post)/triplets['tau_minus']) + 1
            o2          = o2*np.exp(-(spk_time_post-last_spk_post)/triplets['tau_y'])
            r1_at_post  = r1*np.exp(-((spk_time_post+triplets['delay'])-last_spk_pre)/triplets['tau_plus'])
            weight = np.clip(weight + r1_at_post*(triplets['A2_plus']+triplets['A3_plus']*o2),a_min=triplets['WminE'],a_max=triplets['WmaxE'])

            last_spk_post = spk_time_post*1        
            o2 += 1
            
            weight_offline[spk_time_post] = weight*1
            
            ii += 1
            
        # Depression due to pre spike
        r1 = r1*np.exp(-(spk_time_pre-last_spk_pre)/triplets['tau_plus']) + 1
        r2 = r2*np.exp(-(spk_time_pre-last_spk_pre)/triplets['tau_x'])
            
        o1_at_pre_spk = o1*np.exp(-((spk_time_pre-triplets['delay'])-last_spk_post)/triplets['tau_minus'])-1*(last_spk_post==(spk_time_pre-triplets['delay']))

        weight = np.clip(weight - o1_at_pre_spk*(triplets['A2_minus']+triplets['A3_minus']*r2),a_min=triplets['WminE'],a_max=triplets['WmaxE'])

        last_spk_pre = spk_time_pre*1
        r2 += 1

        weight_offline[spk_time_pre] = weight*1
        
        ii += 1
    return np.array(list(weight_offline.values())),np.array(list(weight_offline.keys()))

def generate_spk_train(cv,rate,n_neurons,n_spikes):
    shape = 1./(cv**2)
    scale = cv**2/rate
    intervals = np.random.gamma(shape=shape,scale=scale,size=(n_neurons,n_spikes))          # ISI distribution
    spk_times = np.floor(np.cumsum(intervals,axis=1)*10000)/10.                             # spike times in ms
    
    return spk_times


# Plot
def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

def rate_and_cv(times,period,n_neurons):
    rate = rate_mean(times,period,n_neurons)
    cv_neuron = cv(times)
    return rate,cv_neuron

def grid_plot(array):
    a = 16
    b = 10
    matrix = np.zeros((2*a,5*b))
    for ii in np.arange(10):
        matrix[(ii//5)*a:(ii//5+1)*a,(ii%5)*b:(ii%5+1)*b] = array[ii*a*b:(ii+1)*a*b].reshape(a,b)
    return matrix

def mean_connectivity_matrix(matrix_original,group_size):
    matrix_reduced = np.zeros((int(matrix_original.shape[0]/group_size),int(matrix_original.shape[1]/group_size)))
    for ii in np.arange(matrix_reduced.shape[0]):
        for jj in np.arange(matrix_reduced.shape[1]):
            matrix_patch = matrix_original[ii*group_size:(ii+1)*group_size,jj*group_size:(jj+1)*group_size]
            matrix_reduced[ii,jj] = np.mean(matrix_patch[np.nonzero(matrix_patch)])
    return matrix_reduced

def plot_assembly_rectangle(ax):
    rectangle = patches.Rectangle((-0.5,-0.5),10,16,edgecolor='k',facecolor='none',clip_on=False)
    ax.add_patch(rectangle)
