import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib
import h5py as h5
from importlib import reload
import sys
import matplotlib.patches as patches

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

time_raster = 10000.

color_ass = np.array([152,78,163])/255.

J_all = ['5.0','5.5']

fig = plt.figure(figsize=(6,7))
ax1 = fig.add_axes([0.09,0.82,0.9,0.13])
ax2 = fig.add_axes([0.09,0.64,0.9,0.13])
ax3 = fig.add_axes([0.09,0.42,0.9,0.13])
ax4 = fig.add_axes([0.09,0.24,0.9,0.13])
axes = [ax1,ax2,ax3,ax4]
ax_cc1 = fig.add_axes([0.09,0.05,0.4,0.1])
ax_cc2 = fig.add_axes([0.59,0.05,0.4,0.1])
axes_cc = [ax_cc1,ax_cc2]

labels = ['grow','post']
titles = ['Before','After']

aa = 0
for jj,J in enumerate(J_all):
    cc = []
    for ll,label in enumerate(labels):
        data = h5.File(par.path_to_data[where]+"data_assembly.hdf5",'r')
        data_mode = data['plastic']
        seed  = list(data_mode[J+'/seeds/'].keys())[0]
        group = data_mode[J+'/seeds/'+seed+'/steps/'+label+'/all_neuron']
        times = np.array(group['times'])
        senders = np.array(group['senders'])

        times = times[senders<=(par.N_E)]
        senders = senders[senders<=(par.N_E)]

        senders = senders[times>(max(times)-time_raster)]
        times = times[times>(max(times)-time_raster)]

        axes[aa].scatter(times/1000.,senders,s=0.1,color='grey',linewidth=0,rasterized=True)
        axes[aa].set_yticks([])
        #axes[aa].set_yticklabels(['ASB','EXC'])
        rectangle = patches.Rectangle((min(times/1000.)-0.1,0),0.05,par.assembly_size,color=color_ass,clip_on=False)
        axes[aa].add_patch(rectangle)
        
        axes[aa].spines['right'].set_visible(False)
        axes[aa].spines['top'].set_visible(False)
        axes[aa].spines['left'].set_visible(False)
        
        
        for ss1,sender1 in enumerate(np.unique(senders)[:par.assembly_size]):
            times_spikes1 = times[senders==sender1]
            for ss2,sender2 in enumerate(np.unique(senders)[ss1+1:par.assembly_size]):
                times_spikes2 = times[senders==sender2]
                cc.append(f.cc(times_spikes1,times_spikes2,par.binsize_cc))
        
        axes_cc[jj].hist(cc,bins=100,density=True,histtype='step',cumulative=True,label=titles[ll])
        aa+=1

fig.text(0.01,0.82,"Before stimulation",rotation='vertical')
ax1.set_title(r"$W_{E \to E}^{max}$ = "+J_all[0]+" J")
#fig.text(0.07,0.86,"Neurons",rotation='vertical')
fig.text(0.04,0.83,"assembly")

fig.text(0.01,0.64,"After stimulation",rotation='vertical')
ax2.set_xlabel("Time [s]")
#fig.text(0.07,0.68,"Neurons",rotation='vertical')
fig.text(0.04,0.65,"assembly")

fig.text(0.01,0.42,"Before stimulation",rotation='vertical')
ax3.set_title(r"$W_{E \to E}^{max}$ = "+J_all[1]+" J")
#fig.text(0.07,0.46,"Neurons",rotation='vertical')
fig.text(0.04,0.43,"assembly")

fig.text(0.01,0.24,"After stimulation",rotation='vertical')
ax4.set_xlabel("Time [s]")
#fig.text(0.07,0.28,"Neurons",rotation='vertical')
fig.text(0.04,0.25,"assembly")

ax_cc1.legend()
ax_cc1.set_xlabel("CC")
ax_cc1.set_ylabel("cdf")
ax_cc1.set_title(r"$W_{E \to E}^{max}$ = "+J_all[0]+"J")

ax_cc2.legend()
ax_cc2.set_xlabel("CC")
ax_cc2.set_ylabel("cdf")
ax_cc2.set_title(r"$W_{E \to E}^{max}$ = "+J_all[1]+"J")

fig.text(0.01,0.97,'A')
fig.text(0.01,0.57,'B')
fig.text(0.01,0.17,'C')

#plt.show()

fig.set_size_inches(6,7)
plt.savefig(par.path_to_figure[where]+'figure_full_raster.pdf',dpi=300)
