import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload 
import h5py as h5
import sys
sys.path.append('/home/julia/Documents/iSTDP/paper/main/parameters')

import common
reload(common)
import common as par

matplotlib.rcParams.update({'font.size': 7})

data  = h5.File(par.path_to_data+'data_single_readout.hdf5','r')

def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig = plt.figure(figsize=(7,3))

ax2 = fig.add_axes([0.375,0.65,0.125,0.3])
ax3 = fig.add_axes([0.6,0.65,0.125,0.3])
ax4 = fig.add_axes([0.825,0.65,0.125,0.3])
ax5 = fig.add_axes([0.375,0.15,0.125,0.3])
ax6 = fig.add_axes([0.6,0.15,0.125,0.3])
ax7 = fig.add_axes([0.825,0.15,0.125,0.3])

def plot_row(data_mode,axes):
    mean_vm  = np.array(data_mode['mean_vm'])
    std_vm   = np.array(data_mode['std_vm'])
    rate_out = np.array(data_mode['rate_out'])
    cv_in    = np.array(data_mode['cv_in'])
    
    axes[0].plot(np.mean(cv_in,axis=1),mean_vm)
    axes[0].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[0].set_ylabel(r"$\mu_{V_m}$")
    spines(axes[0])

    axes[1].plot(np.mean(cv_in,axis=1),std_vm)
    axes[1].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[1].set_ylabel(r"$\sigma_{V_m}$")
    spines(axes[1])

    axes[2].plot(np.mean(cv_in,axis=1),rate_out)
    axes[2].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[2].set_ylabel("Rate [Hz]")
    spines(axes[2])

plot_row(data['stp'],[ax5,ax6,ax7])
plot_row(data['static'],[ax2,ax3,ax4])

ax2.set_ylim(ax5.get_ylim())
ax3.set_ylim(ax6.get_ylim())
ax4.set_ylim(ax7.get_ylim())

fig.text(0.03,0.96,'A')
fig.text(0.305,0.96,'B')
fig.text(0.53,0.96,'C')
fig.text(0.755,0.96,'D')

fig.text(0.01,0.69,'No plasticity',rotation='vertical')
fig.text(0.01,0.11,'Short term facilitation',rotation='vertical')

data.close()

fig.set_size_inches(7,3)
plt.savefig(par.path_to_figures+"figure_single_readout.svg")
