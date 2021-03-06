import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from importlib import reload 
import h5py as h5
import sys
direc = sys.argv[0].split('scripts')[0]
where = sys.argv[1]
sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f

matplotlib.rcParams.update({'font.size': par.fontsize})
data = h5.File(par.path_to_data[where]+'data_single_readout.hdf5','r')
color = 'grey'

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
    
    axes[0].plot(np.mean(cv_in,axis=1),mean_vm,color=color)
    axes[0].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[0].set_ylabel(r"Subthres. mean, $\mu_{V_m}$")
    f.spines(axes[0])

    axes[1].plot(np.mean(cv_in,axis=1),std_vm,color=color)
    axes[1].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[1].set_ylabel(r"Subthres. std, $\sigma_{V_m}$")
    f.spines(axes[1])

    axes[2].plot(np.mean(cv_in,axis=1),rate_out,color=color)
    axes[2].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[2].set_ylabel("Rate [Hz]")
    f.spines(axes[2])

plot_row(data['stp'],[ax5,ax6,ax7])
plot_row(data['static'],[ax2,ax3,ax4])

ax2.set_ylim(ax5.get_ylim())
ax3.set_ylim(ax6.get_ylim())
ax4.set_ylim(ax7.get_ylim())

fig.text(0.03,0.96,'A')
fig.text(0.305,0.96,'B')
fig.text(0.53,0.96,'C')
fig.text(0.755,0.96,'D')
fig.text(0.03,0.5,'E')
fig.text(0.305,0.5,'F')
fig.text(0.53,0.5,'G')
fig.text(0.755,0.5,'H')


fig.text(0.01,0.69,'No plasticity',rotation='vertical')
fig.text(0.01,0.11,'Short term facilitation',rotation='vertical')

data.close()

fig.set_size_inches(7,3)
plt.savefig(par.path_to_figure[where]+"figure_single_readout.svg")
