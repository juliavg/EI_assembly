from scipy.integrate import quad as INT
from scipy.integrate import quad
from importlib import reload 
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt
import sys
sys.path.append('/home/julia/Documents/iSTDP/paper/main/parameters')

import common
reload(common)
import common as par

direc = par.path_to_data+'single_neuron/data_theory/'


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


mu_values  = np.arange(par.mu_range[0],par.mu_range[1],par.mu_range[2])
std_values = np.arange(par.std_range[0],par.std_range[1],par.std_range[2])
rate       = np.zeros((mu_values.shape[0],std_values.shape[0]))
CV_all     = np.zeros((mu_values.shape[0],std_values.shape[0]))

for ii,mu in enumerate(mu_values):
    for jj,std in enumerate(std_values):
        rate[ii,jj] = (par.t_ref+par.tau_m*np.sqrt(np.pi)*INT(integrand,(par.V_reset-mu)/std,(par.V_th-mu)/std)[0])**-1
        CV_all[ii,jj] = np.sqrt(2*np.pi*(rate[ii,jj]*par.tau_m)**2*integral2((par.V_reset-mu)/std,(par.V_th-mu)/std))

X,Y = np.meshgrid(mu_values,std_values)
plt.pcolor(X,Y,rate.T,cmap='Greys')
CS = plt.contour(X,Y, rate.T,levels=par.rates_contour)

CV_contours = {}
for rr,nu in enumerate(par.rates_contour):
    CV_temp = np.zeros(len(CS.allsegs[rr][0]))
    for ii in np.arange(len(CS.allsegs[rr][0])):
        mu  = CS.allsegs[rr][0][ii,0]
        std = CS.allsegs[rr][0][ii,1]
        CV_temp[ii] = np.sqrt(2*np.pi*(nu*par.tau_m)**2*integral2((par.V_reset-mu)/std,(par.V_th-mu)/std))
    CV_contours[rr] = CV_temp

#################################
# Save
#################################

np.save(direc+"mu_values.npy",mu_values)
np.save(direc+"std_values.npy",std_values)
np.save(direc+"rate.npy",rate)
np.save(direc+"CV_contours.npy",CV_contours)
np.save(direc+"CS_rates.npy",CS.allsegs)
np.save(direc+"CV_all.npy",CV_all)
