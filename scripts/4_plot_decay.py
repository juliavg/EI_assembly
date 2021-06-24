import numpy as np
import matplotlib.pyplot as plt
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
import functions
reload(functions)
import functions as f

matplotlib.rcParams.update({'font.size': 7})

data_file = h5.File(par.path_to_data[where]+"data_assembly.hdf5","r")
data      = data_file['plastic']
j_all     = list(data.keys())[:3]

colors = np.array([[252,187,161],[251,106,74],[165,15,21]])/255.

def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig = plt.figure(figsize=(7,3))
ax2 = fig.add_axes([0.325,0.65,0.125,0.3])
ax3 = fig.add_axes([0.6,0.65,0.125,0.3])
ax4 = fig.add_axes([0.84,0.65,0.125,0.3])
ax6 = fig.add_axes([0.325,0.15,0.125,0.3])
ax7 = fig.add_axes([0.6,0.15,0.125,0.3])
ax8 = fig.add_axes([0.84,0.15,0.125,0.3])


cv_original = np.array([])
cc_original = np.array([])
slope_original = np.array([])
cv_shifted = np.array([])
cc_shifted = np.array([])
slope_shifted = np.array([])
for jj,J in enumerate(j_all):
    data_J = data[J]
    seeds   = list(data_J['seeds'].keys())
    
    times = np.array([])
    senders = np.array([])
    targets = np.array([])
    weights = np.array([])
    for label in ['post','decay']:
        group   = data_J['seeds/'+seeds[0]+'/steps/'+label+'/weight_E']
        times   = np.concatenate((times,group['times']))
        senders = np.concatenate((senders,group['senders']))
        targets = np.concatenate((targets,group['targets']))
        weights = np.concatenate((weights,group['weights']))

    all_sources = data_J['seeds/'+seeds[0]+'/sources']
    all_targets = data_J['seeds/'+seeds[0]+'/targets']
    
    for ii in np.arange(5):
        t_plot = times[(senders==all_sources[ii])&(targets==all_targets[ii])]
        w_plot = weights[(senders==all_sources[ii])&(targets==all_targets[ii])]
        ax2.plot(t_plot/1000.,w_plot,color=colors[jj])
    
    cv    = data_J['weight_decay/original/cv']
    slope = data_J['weight_decay/original/slope']
    cc    = data_J['weight_decay/original/cc']
    ax3.plot(cv,slope,'.',color=colors[jj])
    ax4.plot(cc,slope,'.',color=colors[jj])
    
    cv_original    = np.concatenate((cv_original,cv))
    slope_original = np.concatenate((slope_original,slope))
    cc_original    = np.concatenate((cc_original,cc))
    
    data_offline = data_J['weight_decay/shifted/offline_weight']
    for ii in list(data_offline.keys()):
        times = np.array(data_offline[ii+'/times'])
        weights = np.array(data_offline[ii+'/weights'])
        ax6.plot(times/1000,weights,color=colors[jj])
    
    cv    = data_J['weight_decay/shifted/cv']
    slope = data_J['weight_decay/shifted/slope']
    cc    = data_J['weight_decay/shifted/cc']
    
    ax7.plot(cv,slope,'.',color=colors[jj],label=r'$W_{E \to E}^{\mathrm{max}}$= %.1f J' %float(J))
    ax8.plot(cc,slope,'.',color=colors[jj])
    
    cv_shifted    = np.concatenate((cv_shifted,cv))
    slope_shifted = np.concatenate((slope_shifted,slope))
    cc_shifted    = np.concatenate((cc_shifted,cc))
    


ax3.text(0.8,-6.5e-6,'r = %.2f' %np.corrcoef(cv_original,slope_original)[0,1])
ax4.text(0.04,-6.e-6,'r = %.2f' %np.corrcoef(cc_original,slope_original)[0,1])
ax7.text(0.8,-7.e-6,'r = %.2f' %np.corrcoef(cv_shifted,slope_shifted)[0,1])
ax8.text(0.001,-7.2e-6,'r = %.2f' %np.corrcoef(cc_shifted,slope_shifted)[0,1])

ax7.legend(bbox_to_anchor=(-0.6, 1.3))

ax2.set_xlabel("Time [s]")
ax2.set_ylabel(r"$W_{E \to E}^{pre \to post}$ [pA]")
ax6.set_xlabel("Time [s]")
ax6.set_ylabel(r"$W_{E \to E}^{pre \to post}$ [pA]")

ax3.set_xlabel("CV")
ax4.set_xlabel("CC")
ax6.set_xlabel("Time [s]")
ax6.set_ylabel(r"$W_{E \to E}^{pre \to post}$ [pA]")
ax7.set_xlabel("CV")
ax8.set_xlabel("CC")

def slope_subplot(ax):
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
    ax.set_ylabel(r"Decay slope, $\alpha$")
    spines(ax)

spines(ax2)
slope_subplot(ax3)
slope_subplot(ax4)
spines(ax6)
slope_subplot(ax7)
slope_subplot(ax8)

fig.text(0.05,0.95,'Original spike trains')
fig.text(0.05,0.45,'Shifted spike trains')
fig.text(0.01,0.96,'A')
fig.text(0.26,0.96,'B')
fig.text(0.53,0.96,'C')
fig.text(0.78,0.96,'D')
fig.text(0.01,0.49,'E')
fig.text(0.26,0.49,'F')
fig.text(0.53,0.49,'G')
fig.text(0.78,0.49,'H')

fig.set_size_inches(7,3)
plt.savefig(par.path_to_figure[where]+'figure_decay.svg')

data_file.close()
