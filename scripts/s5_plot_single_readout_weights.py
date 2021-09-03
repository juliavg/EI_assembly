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

fig = plt.figure(figsize=(6,5))

ax1 = fig.add_axes([0.15,0.75,0.35,0.2])
ax1b = fig.add_axes([0.6,0.75,0.35,0.2])
ax2 = fig.add_axes([0.15,0.4,0.2,0.2])
ax3 = fig.add_axes([0.45,0.4,0.2,0.2])
ax4 = fig.add_axes([0.75,0.4,0.2,0.2])
ax5 = fig.add_axes([0.15,0.1,0.2,0.2])
ax6 = fig.add_axes([0.45,0.1,0.2,0.2])
ax7 = fig.add_axes([0.75,0.1,0.2,0.2])

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

# Plot weight histograms
def plot_histogram(cv,ax):
    wr_weights = np.array(data['stp/weights/'+str(cv)+'/wr_weights'])
    wr_times = np.array(data['stp/weights/'+str(cv)+'/wr_times'])
    wr_senders = np.array(data['stp/weights/'+str(cv)+'/wr_senders'])
    mean_weights = np.zeros(len(np.unique(wr_senders)))
    for ss,sender in enumerate(np.unique(wr_senders)):
        weights = wr_weights[wr_senders==sender]
        times = wr_times[wr_senders==sender]
        mean_weights[ss] = np.mean(weights[times>par.readout_warmup])
    ax.hist(mean_weights,color='grey',bins=20,density=True)
    ax.axvline(np.mean(wr_weights[wr_times>par.readout_warmup]),color='r')
    ax.set_xlabel("Synaptic weight [pA]")
    ax.set_ylabel("pdf")
    ax.set_title("CV = %.1f" %cv)
    f.spines(ax)


plot_histogram(par.CV_all[0],ax1)
plot_histogram(par.CV_all[-1],ax1b)

fig.text(0.05,0.96,'A')
fig.text(0.1,0.64,'B')
fig.text(0.39,0.64,'C')
fig.text(0.72,0.64,'D')
fig.text(0.1,0.32,'E')
fig.text(0.39,0.32,'F')
fig.text(0.72,0.32,'G')


fig.text(0.02,0.45,'No plasticity',rotation='vertical')
fig.text(0.02,0.09,'Short term facilitation',rotation='vertical')

data.close()

fig.set_size_inches(6,5)
plt.savefig(par.path_to_figure[where]+"figure_single_readout_weights.svg")
