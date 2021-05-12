from importlib import reload 
import numpy as np
import matplotlib.pyplot as plt
import h5py as h5
import sys
sys.path.append('/home/julia/Documents/iSTDP/paper/main/support')
import functions
reload(functions)
import functions as f
import parameters
reload(parameters)
import parameters as par

# Create data file
data   = h5.File(par.path_to_data+'data_single_neuron.hdf5','a')
data_theory = data.require_group('theory')

# Initialize vectors
mu_all  = np.arange(par.mu_range[0],par.mu_range[1],par.mu_range[2])
std_all = np.arange(par.std_range[0],par.std_range[1],par.std_range[2])
rate    = np.zeros((mu_all.shape[0],std_all.shape[0]))
CV_all  = np.zeros((mu_all.shape[0],std_all.shape[0]))

# Calculate and save multiple rate and cv
for ii,mu in enumerate(mu_all):
    for jj,std in enumerate(std_all):
        rate[ii,jj]   = f.calculate_rate(mu,std,par.t_ref,par.tau_m,par.V_reset,par.V_th)
        CV_all[ii,jj] = f.calculate_cv(rate[ii,jj],mu,std,par.tau_m,par.V_reset,par.V_th)

f.save_to_group(data_theory,mu_all,'mu_all')
f.save_to_group(data_theory,std_all,'std_all')
f.save_to_group(data_theory,rate,'rate')
f.save_to_group(data_theory,CV_all,'CV_all')

# Extract and save V_m mean and std for rates contour
X,Y = np.meshgrid(mu_all,std_all)
CS  = plt.contour(X,Y, rate.T,levels=par.rates_contour)

CS_rates = data_theory.require_group('CS_rates')
for idx,seg in enumerate(CS.allsegs):
    f.save_to_group(CS_rates,seg[0],str(int(par.rates_contour[idx]*1000.)))

# Calculate and save cv at rate contour
CV_contours = data_theory.require_group('CV_contours')
for rr,nu in enumerate(par.rates_contour):
    CV_temp = np.zeros(len(CS.allsegs[rr][0]))
    for ii in np.arange(len(CS.allsegs[rr][0])):
        mu  = CS.allsegs[rr][0][ii,0]
        std = CS.allsegs[rr][0][ii,1]
        CV_temp[ii] = f.calculate_cv(nu,mu,std,par.tau_m,par.V_reset,par.V_th)
    f.save_to_group(CV_contours,CV_temp,str(int(nu*1000.)))

data.close()
