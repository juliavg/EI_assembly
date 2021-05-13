import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from importlib import reload
import h5py as h5
from matplotlib import colors
import sys

direc = sys.argv[0].split('scripts')[0]
mode  = sys.argv[1]
stim_idx = int(sys.argv[2])

sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f

data  = h5.File(direc+'data_assembly.hdf5','r')
group = data[mode+'/'+str(par.WmaxE[stim_idx])]
seeds = list(group.keys())


matplotlib.rcParams.update({'font.size': 7})


color_ass = np.array([222,203,228])/255.
color_exc = np.array([254,217,166])/255.

color_ass = np.array([179,205,227])/255.
color_exc = np.array([204,235,197])/255.

cmap = 'viridis'
cmap_w = 'seismic'

senders      = np.array([])
times        = np.array([])
group_single = group[seeds[0]+'/steps']

for label in list(group_single.keys()):
    senders = np.concatenate((senders,np.array(group_single[label+"/all_neuron/senders"])))
    times   = np.concatenate((times,np.array(group_single[label+"/all_neuron/times"])))

idx     = np.argsort(times)
senders = senders[idx]
times   = times[idx]

all_senders = np.unique(senders)
ass_id = all_senders[all_senders<par.assembly_size]
exc_id = all_senders[(all_senders>par.assembly_size) & (all_senders<(par.N_E+1))]
inh_id = all_senders[all_senders>(par.N_E+1)]

total_time  = par.warmup_time+par.stimulation_time+par.post_stimulation_time

raster_time = 3000.
raster_n_neurons = par.assembly_size

time_before_upp = par.warmup_time
time_before_low = time_before_upp - par.save_for
time_stim_upp   = par.warmup_time + par.stimulation_time
time_stim_low   = par.warmup_time
time_post_upp   = par.warmup_time + par.stimulation_time + par.post_stimulation_time
time_post_low   = time_post_upp - par.save_for


# Analysis
# Rate
rate_before = np.zeros(par.N_neurons)
rate_stim   = np.zeros(par.N_neurons)
rate_post   = np.zeros(par.N_neurons)

cv_before = np.zeros(par.N_neurons)
cv_stim   = np.zeros(par.N_neurons)
cv_post   = np.zeros(par.N_neurons)

def rate_and_cv(times,period,n_neurons):
    rate = f.rate_mean(times,period,n_neurons)
    cv   = f.cv(times)
    return rate,cv

for nn,neuron in enumerate(np.unique(senders)):
    times_neuron = times[senders==neuron]
    rate_before[nn],cv_before[nn] = rate_and_cv(times_neuron[(times_neuron>time_before_low) & (times_neuron<time_before_upp)],par.save_for,1)
    rate_stim[nn],cv_stim[nn] = rate_and_cv(times_neuron[(times_neuron>time_stim_low) & (times_neuron<time_stim_upp)],par.stimulation_time,1)
    rate_post[nn],cv_post[nn] = rate_and_cv(times_neuron[(times_neuron>time_post_low) & (times_neuron<time_post_upp)],par.save_for,1)

raster_times   = times[np.isin(senders,np.concatenate((ass_id[:raster_n_neurons],exc_id[:raster_n_neurons])))]
raster_senders = senders[np.isin(senders,np.concatenate((ass_id[:raster_n_neurons],exc_id[:raster_n_neurons])))]
raster_senders_b = raster_senders[(raster_times<par.warmup_time) & (raster_times>(par.warmup_time-raster_time))]
raster_times_b = raster_times[(raster_times<par.warmup_time) & (raster_times>(par.warmup_time-raster_time))]
raster_senders_a = raster_senders[(raster_times<total_time) & (raster_times>(total_time-raster_time))]
raster_times_a = raster_times[(raster_times<total_time) & (raster_times>(total_time-raster_time))]


a = 16
b = 10
def create_matrix(array):
    matrix = np.zeros((2*a,5*b))
    for ii in np.arange(10):
        matrix[(ii//5)*a:(ii//5+1)*a,(ii%5)*b:(ii%5+1)*b] = array[ii*a*b:(ii+1)*a*b].reshape(a,b)
    return matrix
    
rate_matrix_before = create_matrix(rate_before[:par.N_E])
rate_matrix_post   = create_matrix(rate_post[:par.N_E])
cv_matrix_before   = create_matrix(cv_before[:par.N_E])
cv_matrix_post     = create_matrix(cv_post[:par.N_E])

min_rate = min([np.min(rate_matrix_before),np.min(rate_matrix_post)])
max_rate = max([np.max(rate_matrix_before),np.max(rate_matrix_post)])
min_cv   = min([np.min(cv_matrix_before),np.min(cv_matrix_post)])
max_cv   = max([np.max(cv_matrix_before),np.max(cv_matrix_post)])

# Connectivity

group_size = 40
def create_mean_matrix(matrix_original):
    matrix_reduced = np.zeros((int(matrix_original.shape[0]/group_size),int(matrix_original.shape[1]/group_size)))
    for ii in np.arange(matrix_reduced.shape[0]):
        for jj in np.arange(matrix_reduced.shape[1]):
            matrix_patch = matrix_original[ii*group_size:(ii+1)*group_size,jj*group_size:(jj+1)*group_size]
            matrix_reduced[ii,jj] = np.mean(matrix_patch[np.nonzero(matrix_patch)])
    return matrix_reduced



sources = np.array(group_single['grow/connections/sources'])-3
targets = np.array(group_single['grow/connections/targets'])-3
weights = np.array(group_single['grow/connections/weights'])

matrix_before = np.zeros((par.N_neurons,par.N_neurons))
matrix_before[targets,sources] = weights

sources = np.array(group_single['post/connections/sources'])-3
targets = np.array(group_single['post/connections/targets'])-3
weights = np.array(group_single['post/connections/weights'])
matrix_after = np.zeros((par.N_neurons,par.N_neurons))
matrix_after[targets,sources] = weights

weight_before = create_mean_matrix(matrix_before[:par.N_E,:])
weight_after = create_mean_matrix(matrix_after[:par.N_E,:])

min_w = min([np.min(weight_before),np.min(weight_after)])
max_w = max([np.max(weight_before),np.max(weight_after)])
norm = colors.DivergingNorm(vmin=min_w, vcenter=0., vmax=max_w)

fig = plt.figure(figsize=(7,3.5))

ax2a = fig.add_axes([0.3,0.42,0.1,0.2])  
ax2b = fig.add_axes([0.3,0.12,0.1,0.2])  
ax2c = fig.add_axes([0.41,0.12,0.01,0.5])  

ax3a = fig.add_axes([0.54,0.42,0.1,0.2])
ax3b = fig.add_axes([0.54,0.12,0.1,0.2])
ax3c = fig.add_axes([0.65,0.12,0.01,0.5])

ax4a = fig.add_axes([0.75,0.42,0.16,0.2])
ax4b = fig.add_axes([0.75,0.12,0.16,0.2])
ax4c = fig.add_axes([0.9,0.12,0.01,0.5])

ax6  = fig.add_axes([0.3,0.72,0.1,0.2])
ax7  = fig.add_axes([0.54,0.72,0.1,0.2])
ax8  = fig.add_axes([0.77,0.72,0.1,0.2])

ax9  = fig.add_axes([0.07,0.42,0.18,0.2])
ax9a = fig.add_axes([0.07,0.12,0.18,0.2])


ax2a.imshow(rate_matrix_before,vmin=min_rate,vmax=max_rate,cmap=cmap,rasterized=True)
im   = ax2b.imshow(rate_matrix_post,vmin=min_rate,vmax=max_rate,cmap=cmap,rasterized=True)
cbar = fig.colorbar(im,cax=ax2c)
cbar.set_label("Firing Rate [Hz]")

ax3a.imshow(cv_matrix_before,vmin=min_cv,vmax=max_cv,cmap=cmap,rasterized=True)
im   = ax3b.imshow(cv_matrix_post,vmin=min_cv,vmax=max_cv,cmap=cmap,rasterized=True)
cbar = fig.colorbar(im,cax=ax3c)
cbar.set_label("CV")    

ax4a.imshow(weight_before,vmin=min_w,vmax=max_w,cmap=cmap_w,norm=norm,rasterized=True)
im = ax4b.imshow(weight_after,vmin=min_w,vmax=max_w,cmap=cmap_w,norm=norm,rasterized=True)
cbar = fig.colorbar(im,cax=ax4c)
cbar.set_label("Weight [pA]")

a = par.assembly_size
b = par.N_E
width=0.3

ax6.bar(0-width/2,height=np.mean(rate_before[:a]),width=width,yerr=np.std(rate_before[:a]),color=color_ass)
ax6.bar(0+width/2,height=np.mean(rate_before[a:b]),width=width,yerr=np.std(rate_before[a:b]),color=color_exc)
ax6.bar(1-width/2,height=np.mean(rate_post[:a]),width=width,yerr=np.std(rate_post[:a]),color=color_ass)
ax6.bar(1+width/2,height=np.mean(rate_post[a:b]),width=width,yerr=np.std(rate_post[a:b]),color=color_exc)
ax6.set_ylabel("Firing rate [Hz]")
ax6.set_xticks([0,1])
ax6.set_xticklabels(['Before','After'])

ax7.bar(0-width/2,height=np.mean(cv_before[:a]),width=width,yerr=np.std(cv_before[:a]),color=color_ass)
ax7.bar(0+width/2,height=np.mean(cv_before[a:b]),width=width,yerr=np.std(cv_before[a:b]),color=color_exc)
ax7.bar(1-width/2,height=np.mean(cv_post[:a]),width=width,yerr=np.std(cv_post[:a]),color=color_ass)
ax7.bar(1+width/2,height=np.mean(cv_post[a:b]),width=width,yerr=np.std(cv_post[a:b]),color=color_exc)
ax7.set_ylabel("CV")
ax7.set_xticks([0,1])
ax7.set_xticklabels(['Before','After'])

for ii,neuron in enumerate(np.unique(raster_senders_b)):
    times_temp = raster_times_b[raster_senders_b==neuron]/1000.
    ax9.scatter(times_temp,ii*np.ones(len(times_temp)),color='grey',s=0.1,linewidth=0,rasterized=True)
ax9.set_ylabel("Before")

for ii,neuron in enumerate(np.unique(raster_senders_a)):
    times_temp = raster_times_a[raster_senders_a==neuron]/1000.
    ax9a.scatter(times_temp,ii*np.ones(len(times_temp)),color='grey',s=0.1,linewidth=0,rasterized=True)
ax9a.set_ylabel("After")
ax9a.set_xlabel("Time [s]")

# Readout rate
readout_ass_before_plastic = []
readout_ass_after_plastic  = []
readout_exc_before_plastic = []
readout_exc_after_plastic  = []
readout_ass_before_static = []
readout_ass_after_static  = []
readout_exc_before_static = []
readout_exc_after_static  = []
for seed in seeds:
    readout_senders   = np.array(group[str(seed)+'/steps/post/readout/senders'])
    readout_times     = np.array(group[str(seed)+'/steps/post/readout/times'])
    times_ass_plastic = readout_times[(readout_senders==np.unique(readout_senders)[0])&(readout_times>(max(readout_times)-par.save_for))]
    times_exc_plastic = readout_times[(readout_senders==np.unique(readout_senders)[1])&(readout_times>(max(readout_times)-par.save_for))]
    readout_ass_after_plastic.append(f.rate_mean(times_ass_plastic,par.save_for,1))
    readout_exc_after_plastic.append(f.rate_mean(times_exc_plastic,par.save_for,1))
    times_ass_plastic = readout_times[(readout_senders==np.unique(readout_senders)[0])&(readout_times>(par.warmup_time-par.save_for))&(readout_times<par.warmup_time)]
    times_exc_plastic = readout_times[(readout_senders==np.unique(readout_senders)[1])&(readout_times>(par.warmup_time-par.save_for))&(readout_times<par.warmup_time)]
    readout_ass_before_plastic.append(f.rate_mean(times_ass_plastic,par.save_for,1))
    readout_exc_before_plastic.append(f.rate_mean(times_exc_plastic,par.save_for,1))
    
ax8.bar(0-width/2,height=np.mean(readout_ass_before_plastic),width=width,yerr=np.std(readout_ass_before_plastic),color=color_ass)
ax8.bar(0+width/2,height=np.mean(readout_exc_before_plastic),width=width,yerr=np.std(readout_exc_before_plastic),color=color_exc)   
ax8.bar(1-width/2,height=np.mean(readout_ass_after_plastic),width=width,yerr=np.std(readout_ass_after_plastic),color=color_ass)
ax8.bar(1+width/2,height=np.mean(readout_exc_after_plastic),width=width,yerr=np.std(readout_exc_after_plastic),color=color_exc)

ax8.set_ylabel("Readout rate [Hz]")
ax8.set_xticks([0,1])
ax8.set_xticklabels(['Before','After'])

fig.set_size_inches(7,3.5)

plt.savefig(direc+"figure_assembly_"+mode+".svg",dpi=300)
