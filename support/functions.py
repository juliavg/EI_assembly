import numpy as np
import h5py as h5
from scipy.integrate import quad as INT
from scipy.integrate import quad
import scipy.special as sp

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
    
    
# Simulation rate and cv
def rate_mean(spk_times,period,n_neurons):
    return len(spk_times)/n_neurons/period*1000.

def rate_time_series(spk_times,bins,n_neurons):
    binsize = np.diff(bins)[0]
    return np.histogram(spk_times,bins=bins)[0]/n_neurons/binsize*1000.

def cv(spk_times):
    isi = np.diff(spk_times)
    return np.std(isi)/np.mean(isi)
    
    
# 
def generate_spk_train(cv,rate,n_neurons,n_spikes):
    shape = 1./(cv**2)
    scale = cv**2/rate
    intervals = np.random.gamma(shape=shape,scale=scale,size=(n_neurons,n_spikes))          # ISI distribution
    spk_times = np.floor(np.cumsum(intervals,axis=1)*10000)/10.                             # spike times in ms
    
    return spk_times
