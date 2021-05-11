from scipy.integrate import quad as INT
from scipy.integrate import quad
from importlib import reload 
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import h5py as h5
import sys
sys.path.append('/home/julia/Documents/iSTDP/paper/main/parameters')

import common
reload(common)
import common as par

data   = h5.File(par.path_to_data+'data_single_neuron.hdf5','a')
data_theory = data.require_group('theory')

#################################
# Theoretical CV - Brunel, 2000
#################################

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

#################################
# Theoretical Rate - Brunel, 2000
#################################

def integrand(u):
    return sp.erfcx(-u)


mu_all  = np.arange(par.mu_range[0],par.mu_range[1],par.mu_range[2])
std_all = np.arange(par.std_range[0],par.std_range[1],par.std_range[2])
rate    = np.zeros((mu_all.shape[0],std_all.shape[0]))
CV_all  = np.zeros((mu_all.shape[0],std_all.shape[0]))

for ii,mu in enumerate(mu_all):
    for jj,std in enumerate(std_all):
        rate[ii,jj] = (par.t_ref+par.tau_m*np.sqrt(np.pi)*INT(integrand,(par.V_reset-mu)/std,(par.V_th-mu)/std)[0])**-1
        CV_all[ii,jj] = np.sqrt(2*np.pi*(rate[ii,jj]*par.tau_m)**2*integral2((par.V_reset-mu)/std,(par.V_th-mu)/std))

data_theory.require_dataset('mu_all',mu_all.shape,dtype=mu_all.dtype)
data_theory.require_dataset('std_all',std_all.shape,dtype=std_all.dtype)
data_theory.require_dataset('rate',rate.shape,dtype=rate.dtype)
data_theory.require_dataset('CV_all',CV_all.shape,dtype=CV_all.dtype)
data_theory['mu_all'][...]  = mu_all
data_theory['std_all'][...] = std_all
data_theory['rate'][...]    = rate
data_theory['CV_all'][...]  = CV_all

X,Y = np.meshgrid(mu_all,std_all)
CS  = plt.contour(X,Y, rate.T,levels=par.rates_contour)

CS_rates = data_theory.require_group('CS_rates')
for idx,seg in enumerate(CS.allsegs):
    CS_rates.require_dataset(str(int(par.rates_contour[idx]*1000.)),seg[0].shape,dtype=seg[0].dtype)
    CS_rates[str(int(par.rates_contour[idx]*1000.))][...] = seg[0]

CV_contours = data_theory.require_group('CV_contours')
for rr,nu in enumerate(par.rates_contour):
    CV_temp = np.zeros(len(CS.allsegs[rr][0]))
    for ii in np.arange(len(CS.allsegs[rr][0])):
        mu  = CS.allsegs[rr][0][ii,0]
        std = CS.allsegs[rr][0][ii,1]
        CV_temp[ii] = np.sqrt(2*np.pi*(nu*par.tau_m)**2*integral2((par.V_reset-mu)/std,(par.V_th-mu)/std))
    CV_contours.require_dataset(str(int(nu*1000.)),CV_temp.shape,dtype=CV_temp.dtype)
    CV_contours[str(int(nu*1000.))][...] = CV_temp

data.close()
