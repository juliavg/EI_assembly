import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from importlib import reload
import h5py as h5
from matplotlib import colors
import sys
import matplotlib.patches as patches
direc = sys.argv[0].split('scripts')[0]
where = sys.argv[1]
mode = sys.argv[2]
stim_idx = int(sys.argv[3])
sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f

matplotlib.rcParams.update({'font.size': par.fontsize})
data = h5.File('/home/julia/Documents/iSTDP/paper/main/data/data_assembly.hdf5','r')
group = data[mode+'/'+str(par.WmaxE[stim_idx])+'/seeds']
seeds = list(group.keys())

# Parameters
color_ass = np.array([152,78,163])/255.
color_exc = np.array([228,26,28])/255.
cmap = 'OrRd'
cmap_w = 'coolwarm'
bar_width = 0.3
raster_time = 3000.
conn_group_size = 40
binsize_rate = 100.
binsize_weight = 100.
binsize_cv = 1000.

total_time = par.warmup_time+par.stimulation_time+par.post_stimulation_time

weight_bins = np.arange(0.,total_time+par.decay_time['normal'],binsize_weight)
cv_bins = np.arange(0.,total_time+par.decay_time['normal'],binsize_cv)
readout_bins = np.arange(0.,total_time+par.decay_time['normal']+binsize_rate,binsize_rate)

weight_E = np.zeros(weight_bins.shape[0]-1)
weight_I = np.zeros(weight_bins.shape[0]-1)
cv_assembly = np.zeros(cv_bins.shape[0]-1)
cv_excitatory = np.zeros(cv_bins.shape[0]-1)
rate_assembly = np.zeros(len(readout_bins)-1)
readout_assembly_rate = np.zeros(len(readout_bins)-1)

def gather_data(group,seed,data_type):
    group_single = group[seed+'/steps']

    senders = np.array([])
    times = np.array([])
    if data_type[:6]=="weight":
        targets = np.array([])
        weights = np.array([])
        for label in par.labels:
            senders = np.concatenate((senders,np.array(group_single[label+"/"+data_type+"/senders"])))
            targets = np.concatenate((targets,np.array(group_single[label+"/"+data_type+"/targets"])))
            times = np.concatenate((times,np.array(group_single[label+"/"+data_type+"/times"])))
            weights = np.concatenate((weights,np.array(group_single[label+"/"+data_type+"/weights"])))
        senders -= 3
        return senders,targets,times,weights
        
    else:
        for label in par.labels:
            senders = np.concatenate((senders,np.array(group_single[label+"/"+data_type+"/senders"])))
            times = np.concatenate((times,np.array(group_single[label+"/"+data_type+"/times"])))
        senders -= 3
        idx = np.argsort(times)
        senders = senders[idx]
        times = times[idx]
        return senders,times


for seed in seeds:
    # Excitatory weights
    senders,targets,times,weights = gather_data(group,seed,"weight_E")
    targets = targets[senders<=par.assembly_size]
    times = times[senders<=par.assembly_size]
    weights = weights[senders<=par.assembly_size]
    senders = senders[senders<=par.assembly_size]
    times = times[targets<=par.assembly_size]
    weights = weights[targets<=par.assembly_size]
    senders = senders[targets<=par.assembly_size]
    targets = targets[targets<=par.assembly_size]    
    weight_E += np.histogram(times,bins=weight_bins,weights=weights)[0]/np.histogram(times,bins=weight_bins)[0]
    
    # Inhibitory weights
    senders,targets,times,weights = gather_data(group,seed,"weight_I")
    times = times[targets<=par.assembly_size]
    weights = weights[targets<=par.assembly_size]
    senders = senders[targets<=par.assembly_size]
    targets = targets[targets<=par.assembly_size]    
    weight_I += np.histogram(times,bins=weight_bins,weights=weights)[0]/np.histogram(times,bins=weight_bins)[0]

    # Assembly rate
    senders,times = gather_data(group,seed,"all_sim")
    times = times[senders<par.assembly_size]
    senders = senders[senders<par.assembly_size]
    rate_assembly += np.histogram(times,bins=readout_bins)[0]/len(np.unique(senders))/binsize_rate*1000.

    # CV
    cv_temp = np.zeros((len(np.unique(senders)),cv_bins.shape[0]-1))
    senders = senders[times>=cv_bins[0]]
    times = times[times>=cv_bins[0]]
    cv_asb = np.zeros((np.unique(senders).shape[0],cv_bins.shape[0]-1))
    for tt,max_time in enumerate(cv_bins[1:]):
        senders_bin = senders[times<max_time]
        times_bin = times[times<max_time]
        for nn,neuron in enumerate(np.unique(senders)):
            times_neuron = times_bin[senders_bin==neuron]
            isi = np.diff(times_neuron)
            cv_asb[nn,tt] = np.std(isi)/np.mean(isi)
        senders = senders[times>=max_time]
        times = times[times>=max_time] 
    cv_assembly = np.vstack((cv_assembly,cv_asb))

    # Readout rate
    senders,times = gather_data(group,seed,"readout") 
    readout_assembly_rate += np.histogram(times[senders==np.unique(senders)[0]],bins=readout_bins)[0]/binsize_rate*1000.


weight_E /= len(seeds)
weight_I /= len(seeds)
rate_assembly /= len(seeds)
readout_assembly_rate /= len(seeds)

xmin = 1980
xmax = 2050

fig = plt.figure(figsize=(6,6))
ax1 = fig.add_axes([0.15,0.86,0.75,0.1])
ax2 = fig.add_axes([0.15,0.67,0.75,0.1])
ax3 = fig.add_axes([0.15,0.48,0.75,0.1])
ax4 = fig.add_axes([0.15,0.29,0.75,0.1])
ax5 = fig.add_axes([0.15,0.1,0.75,0.1])

ax1.plot(weight_bins[:-1]/1000.,weight_E)
ax1.set_xlim([xmin,xmax])
ax1.axvspan(2000,2001,color='grey',alpha=0.3)
ax1.set_ylabel("E Weight [pA]")

ax2.plot(weight_bins[:-1]/1000.,weight_I)
ax2.set_ylabel("I Weight [pA]")
ax2.set_xlim([xmin,xmax])
ax2.axvspan(2000,2001,color='grey',alpha=0.3)

ax3.plot(readout_bins[:-1]/1000.,rate_assembly)
ax3.set_ylabel("Assembly\nRate [Hz]")
ax3.set_xlim([xmin,xmax])
ax3.axvspan(2000,2001,color='grey',alpha=0.3)

ax4.plot(cv_bins[:-1]/1000.,np.mean(cv_assembly,axis=0))
ax4.set_ylabel("Assembly CV")
ax4.set_xlim([xmin,xmax])
ax4.axvspan(2000,2001,color='grey',alpha=0.3)

ax5.plot(readout_bins[:-1]/1000.,readout_assembly_rate)
ax5.set_ylabel("Readout\nRate [Hz]")
ax5.set_xlim([xmin,xmax])
ax5.axvspan(2000,2001,color='grey',alpha=0.3)
ax5.set_xlabel("Time [s]")

fig.set_size_inches(6,6)
plt.savefig(par.path_to_figure[where]+'figure_time_series.pdf',dpi=300)
