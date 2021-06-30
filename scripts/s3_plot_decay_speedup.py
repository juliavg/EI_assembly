import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import h5py as h5
from importlib import reload
import sys
direc = sys.argv[0].split('scripts')[0]
where = sys.argv[1]
sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par

matplotlib.rcParams.update({'font.size': par.fontsize})
colors = np.array([[252,187,161],[251,106,74],[165,15,21]])/255.

fig = plt.figure(figsize=(7,3))
ax = fig.add_axes([0.15,0.15,0.8,0.8])

data = h5.File(par.path_to_data[where]+"data_assembly.hdf5",'r')
data_mode = data['speedup']
j_all = list(data_mode.keys())

for jj,J in enumerate(j_all):
    seed  = list(data_mode[J+'/seeds/'].keys())[0]
    group = data_mode[J+'/seeds/'+seed]
    
    senders_all = np.array(group['sources'])
    targets_all = np.array(group['targets'])
    
    group = group['steps']
    times   = np.array([])
    senders = np.array([])
    targets = np.array([])
    weights = np.array([])
    for label in par.labels[2:]:
        weight_E = group[label+'/weight_E']
        times    = np.concatenate((times,weight_E['times']))
        senders  = np.concatenate((senders,weight_E['senders']))
        targets  = np.concatenate((targets,weight_E['targets']))
        weights  = np.concatenate((weights,weight_E['weights']))
    weights = weights[times>(par.warmup_time+par.stimulation_time)]
    targets = targets[times>(par.warmup_time+par.stimulation_time)]
    senders = senders[times>(par.warmup_time+par.stimulation_time)]
    times   = times[times>(par.warmup_time+par.stimulation_time)]
    for ii in np.arange(len(senders_all)):
        t_plot = times[(senders==senders_all[ii])&(targets==targets_all[ii])]
        w_plot = weights[(senders==senders_all[ii])&(targets==targets_all[ii])]
        ax.plot(t_plot/1000.,w_plot,color=colors[jj],label=r'$W_{E \to E}^{max}$ = %.1f J' %float(J))

handles, labels = fig.gca().get_legend_handles_labels()
labels, ids = np.unique(labels, return_index=True)
handles = [handles[i] for i in ids]
ax.legend(handles, labels, loc='best')

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$W_{E \to E}$ [pA]")

fig.set_size_inches(7,3)
plt.savefig(par.path_to_figure[where]+'figure_decay_speedup.pdf')
