import numpy as np
from scipy.integrate import quad as INT
import scipy.special as sp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.integrate import odeint
import scipy.optimize as so

direc = "../data/single_neuron/data_theory/"

#################################
# Parameters
#################################

t_ref       = 2.                                # refractory period (ms)
tau_mem     = 20.                               # membrane time constant (ms)
V_reset     = 10.                               # reset potential (mV)
V_th        = 20.                               # firing threshold (mV)

rates_contour = np.array([0.001,0.01,0.05])

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

# x = [rateE,meanE,sigmaE,rateI,meanI,sigmaI,d_g,d_CE]

mu_values  = np.arange(0,20.,.1)
std_values = np.arange(1,30,.1)
rate       = np.zeros((mu_values.shape[0],std_values.shape[0]))
CV_all     = np.zeros((mu_values.shape[0],std_values.shape[0]))
#rate       = np.load("../data/single_neuron/data_theory/rate.npy")
#CV_all     = np.load("../data/single_neuron/data_theory/CV_all.npy")

for ii,mu in enumerate(mu_values):
    for jj,sigma in enumerate(std_values):
        rate[ii,jj] = (t_ref+tau_mem*np.sqrt(np.pi)*INT(integrand,(V_reset-mu)/sigma,(V_th-mu)/sigma)[0])**-1
        CV_all[ii,jj] = np.sqrt(2*np.pi*(rate[ii,jj]*tau_mem)**2*integral2((V_reset-mu)/sigma,(V_th-mu)/sigma))

X,Y = np.meshgrid(mu_values,std_values)
plt.pcolor(X,Y,rate.T,cmap='Greys')
CS = plt.contour(X,Y, rate.T,levels=rates_contour)

CV_contours = {}
for rr,nu in enumerate(rates_contour):
    CV_temp = np.zeros(len(CS.allsegs[rr][0]))
    for ii in np.arange(len(CS.allsegs[rr][0])):
        mu  = CS.allsegs[rr][0][ii,0]
        std = CS.allsegs[rr][0][ii,1]
        CV_temp[ii] = np.sqrt(2*np.pi*(nu*tau_mem)**2*integral2((V_reset-mu)/std,(V_th-mu)/std))
    CV_contours[rr] = CV_temp

np.save(direc+"mu_values.npy",mu_values)
np.save(direc+"std_values.npy",std_values)
np.save(direc+"rate.npy",rate)
np.save(direc+"CV_contours.npy",CV_contours)
np.save(direc+"CS_rates.npy",CS.allsegs)
np.save(direc+"CV_all.npy",CV_all)

plt.show()
