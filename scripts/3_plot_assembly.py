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
mode  = sys.argv[2]
stim_idx = int(sys.argv[3])
sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par
import functions
reload(functions)
import functions as f

matplotlib.rcParams.update({'font.size': par.fontsize})
data = h5.File(par.path_to_data[where]+'data_assembly.hdf5','r')
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

# Indices and time bounds for raster plots
senders      = np.array([])
times        = np.array([])
group_single = group[seeds[0]+'/steps']
for label in par.labels:
    #senders = np.concatenate((senders,np.array(group_single[label+"/all_neuron/senders"])))
    senders = np.concatenate((senders,np.array(group_single[label+"/all_sim/senders"])))
    #times   = np.concatenate((times,np.array(group_single[label+"/all_neuron/times"])))
    times   = np.concatenate((times,np.array(group_single[label+"/all_sim/times"])))

senders -= min(senders)
idx = np.argsort(times)
senders = senders[idx]
times = times[idx]

all_senders = np.unique(senders)
ass_id = all_senders[all_senders<par.assembly_size]
exc_id = all_senders[(all_senders>=par.assembly_size) & (all_senders<par.N_E)]
inh_id = all_senders[all_senders>=par.N_E]

total_time = par.warmup_time+par.stimulation_time+par.post_stimulation_time

raster_n_neurons = par.assembly_size

time_before_upp = par.warmup_time
time_before_low = time_before_upp - par.save_for
time_stim_upp = par.warmup_time + par.stimulation_time
time_stim_low = par.warmup_time
time_post_upp = par.warmup_time + par.stimulation_time + par.post_stimulation_time
time_post_low = time_post_upp - par.save_for

# Analysis
# Rates and CVs
rate_before = np.zeros(par.N_neurons)
rate_post = np.zeros(par.N_neurons)

cv_before = np.zeros(par.N_neurons)
cv_post = np.zeros(par.N_neurons)

for nn,neuron in enumerate(np.unique(senders)):
    times_neuron = times[senders==neuron]
    rate_before[nn],cv_before[nn] = f.rate_and_cv(times_neuron[(times_neuron>time_before_low) &
                                                               (times_neuron<time_before_upp)],par.save_for,1)
    rate_post[nn],cv_post[nn] = f.rate_and_cv(times_neuron[(times_neuron>time_post_low) & 
                                                           (times_neuron<time_post_upp)],par.save_for,1)

# Select times and neurons for raster
raster_times = times[np.isin(senders,np.concatenate((ass_id[:raster_n_neurons],exc_id[:raster_n_neurons])))]
raster_senders = senders[np.isin(senders,np.concatenate((ass_id[:raster_n_neurons],exc_id[:raster_n_neurons])))]
raster_senders_b = raster_senders[(raster_times<par.warmup_time) & (raster_times>(par.warmup_time-raster_time))]
raster_times_b = raster_times[(raster_times<par.warmup_time) & (raster_times>(par.warmup_time-raster_time))]
raster_senders_a = raster_senders[(raster_times<total_time) & (raster_times>(total_time-raster_time))]
raster_times_a = raster_times[(raster_times<total_time) & (raster_times>(total_time-raster_time))]

# Creates matrix for displaying firing rates and cv in a grid
rate_matrix_before = f.grid_plot(rate_before[:par.N_E])
rate_matrix_post = f.grid_plot(rate_post[:par.N_E])
cv_matrix_before = f.grid_plot(cv_before[:par.N_E])
cv_matrix_post = f.grid_plot(cv_post[:par.N_E])

min_rate = min([np.min(rate_matrix_before),np.min(rate_matrix_post)])
max_rate = max([np.max(rate_matrix_before),np.max(rate_matrix_post)])
min_cv = min([np.min(cv_matrix_before),np.min(cv_matrix_post)])
max_cv = max([np.max(cv_matrix_before),np.max(cv_matrix_post)])

# Creates connectivity matrix
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

weight_before = f.mean_connectivity_matrix(matrix_before[:par.N_E,:],conn_group_size)
weight_after = f.mean_connectivity_matrix(matrix_after[:par.N_E,:],conn_group_size)

min_w = min([np.min(weight_before),np.min(weight_after)])
max_w = max([np.max(weight_before),np.max(weight_after)])
norm = colors.DivergingNorm(vmin=min_w, vcenter=0., vmax=max_w)

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
    readout_senders = np.array(group[str(seed)+'/steps/post/readout/senders'])
    readout_times = np.array(group[str(seed)+'/steps/post/readout/times'])
    times_ass_plastic = readout_times[(readout_senders==np.unique(readout_senders)[0])&
                                      (readout_times>(max(readout_times)-par.save_for))]
    times_exc_plastic = readout_times[(readout_senders==np.unique(readout_senders)[1])&
                                      (readout_times>(max(readout_times)-par.save_for))]
    readout_ass_after_plastic.append(f.rate_mean(times_ass_plastic,par.save_for,1))
    readout_exc_after_plastic.append(f.rate_mean(times_exc_plastic,par.save_for,1))
    times_ass_plastic = readout_times[(readout_senders==np.unique(readout_senders)[0])&
                                      (readout_times>(par.warmup_time-par.save_for))&
                                      (readout_times<par.warmup_time)]
    times_exc_plastic = readout_times[(readout_senders==np.unique(readout_senders)[1])&
                                      (readout_times>(par.warmup_time-par.save_for))&
                                      (readout_times<par.warmup_time)]
    readout_ass_before_plastic.append(f.rate_mean(times_ass_plastic,par.save_for,1))
    readout_exc_before_plastic.append(f.rate_mean(times_exc_plastic,par.save_for,1))

# Figure
fig = plt.figure(figsize=(7,3.5))

ax2a = fig.add_axes([0.33,0.42,0.1,0.2])  
ax2b = fig.add_axes([0.33,0.12,0.1,0.2])  
ax2c = fig.add_axes([0.44,0.12,0.01,0.5])  

ax3a = fig.add_axes([0.55,0.42,0.1,0.2])
ax3b = fig.add_axes([0.55,0.12,0.1,0.2])
ax3c = fig.add_axes([0.66,0.12,0.01,0.5])

ax4a = fig.add_axes([0.05,0.38,0.16,0.2])
ax4b = fig.add_axes([0.05,0.12,0.16,0.2])
ax4c = fig.add_axes([0.2,0.12,0.01,0.46])

ax6  = fig.add_axes([0.33,0.72,0.1,0.2])
ax7  = fig.add_axes([0.55,0.72,0.1,0.2])
ax8  = fig.add_axes([0.77,0.72,0.1,0.2])

ax9  = fig.add_axes([0.78,0.42,0.18,0.2])
ax9a = fig.add_axes([0.78,0.12,0.18,0.2])

# Panel D
ax2a.imshow(rate_matrix_before,vmin=min_rate,vmax=max_rate,cmap=cmap,rasterized=True)
im = ax2b.imshow(rate_matrix_post,vmin=min_rate,vmax=max_rate,cmap=cmap,rasterized=True)
cbar = fig.colorbar(im,cax=ax2c)
cbar.set_label("Firing Rate [Hz]")
ax2a.set_xticks([])
ax2a.set_yticks([])
ax2b.set_xticks([])
ax2b.set_yticks([])
f.plot_assembly_rectangle(ax2a)
f.plot_assembly_rectangle(ax2b)

# Panel F
ax3a.imshow(cv_matrix_before,vmin=min_cv,vmax=max_cv,cmap=cmap,rasterized=True)
im = ax3b.imshow(cv_matrix_post,vmin=min_cv,vmax=max_cv,cmap=cmap,rasterized=True)
cbar = fig.colorbar(im,cax=ax3c)
cbar.set_label("CV")    
ax3a.set_xticks([])
ax3a.set_yticks([])
ax3b.set_xticks([])
ax3b.set_yticks([])
f.plot_assembly_rectangle(ax3a)
f.plot_assembly_rectangle(ax3b)

# Panel G
ax4a.imshow(weight_before,vmin=min_w,vmax=max_w,cmap=cmap_w,norm=norm,rasterized=True)
im = ax4b.imshow(weight_after,vmin=min_w,vmax=max_w,cmap=cmap_w,norm=norm,rasterized=True)
cbar = fig.colorbar(im,cax=ax4c)
cbar.set_label("Mean weight between groups [pA]")
ax4a.set_xticks([])
ax4a.set_yticks([])
ax4a.set_xlabel("Pre group")
ax4a.set_ylabel("Post group")
ax4b.set_xticks([])
ax4b.set_yticks([])
ax4b.set_xlabel("Pre group")
ax4b.set_ylabel("Post group")
ax4a.set_title("Synaptic weights",fontsize=par.fontsize)
n_groups = par.N_E/conn_group_size
rectangle = patches.Rectangle((-3,0),1,par.assembly_size/conn_group_size,color=color_ass,clip_on=False)
ax4b.add_patch(rectangle)
rectangle = patches.Rectangle((-3,0),1,par.assembly_size/conn_group_size,color=color_ass,clip_on=False)
ax4a.add_patch(rectangle)
rectangle = patches.Rectangle((0,n_groups+1),par.assembly_size/conn_group_size,1,color=color_ass,clip_on=False)
ax4a.add_patch(rectangle)
rectangle = patches.Rectangle((0,n_groups+1),par.assembly_size/conn_group_size,1,color=color_ass,clip_on=False)
ax4b.add_patch(rectangle)

# Panel C
ax6.bar(0-bar_width/2,height=np.mean(rate_before[:par.assembly_size]),
                  width=bar_width,
                  yerr=np.std(rate_before[:par.assembly_size]),
                  color=color_ass)
ax6.bar(0+bar_width/2,height=np.mean(rate_before[par.assembly_size:par.N_E]),
                  width=bar_width,
                  yerr=np.std(rate_before[par.assembly_size:par.N_E]),
                  color=color_exc)
ax6.bar(1-bar_width/2,height=np.mean(rate_post[:par.assembly_size]),
                  width=bar_width,
                  yerr=np.std(rate_post[:par.assembly_size]),
                  color=color_ass,
                  label='within assembly')
ax6.bar(1+bar_width/2,height=np.mean(rate_post[par.assembly_size:par.N_E]),
                  width=bar_width,
                  yerr=np.std(rate_post[par.assembly_size:par.N_E]),
                  color=color_exc,
                  label='outside assembly')
ax6.legend(bbox_to_anchor=(3.8,1.4),ncol=2)
ax6.set_ylabel("Firing rate [Hz]")
ax6.set_xticks([0,1])
ax6.set_xticklabels(['Before','After'])
f.spines(ax6)

# Panel E
ax7.bar(0-bar_width/2,height=np.mean(cv_before[:par.assembly_size]),
                  width=bar_width,
                  yerr=np.std(cv_before[:par.assembly_size]),
                  color=color_ass)
ax7.bar(0+bar_width/2,height=np.mean(cv_before[par.assembly_size:par.N_E]),
                  width=bar_width,
                  yerr=np.std(cv_before[par.assembly_size:par.N_E]),
                  color=color_exc)
ax7.bar(1-bar_width/2,height=np.mean(cv_post[:par.assembly_size]),
                  width=bar_width,
                  yerr=np.std(cv_post[:par.assembly_size]),
                  color=color_ass)
ax7.bar(1+bar_width/2,height=np.mean(cv_post[par.assembly_size:par.N_E]),
                  width=bar_width,
                  yerr=np.std(cv_post[par.assembly_size:par.N_E]),
                  color=color_exc)
ax7.set_ylabel("CV")
ax7.set_xticks([0,1])
ax7.set_xticklabels(['Before','After'])
f.spines(ax7)

# Panel B
for ii,neuron in enumerate(np.unique(raster_senders_b)):
    times_temp = raster_times_b[raster_senders_b==neuron]/1000.
    ax9.scatter(times_temp,ii*np.ones(len(times_temp)),color='grey',s=0.3,linewidth=0,rasterized=False)
ax9.set_yticks([])
f.spines(ax9)
ax9.spines['left'].set_visible(False)
fig.text(0.01,0.45,"Before",rotation='vertical')
ax9.text(1996.5,-15,"assembly",rotation='vertical')
rectangle = patches.Rectangle((1996.8,0),0.1,par.assembly_size,color=color_ass,clip_on=False)
ax9.add_patch(rectangle)

for ii,neuron in enumerate(np.unique(raster_senders_a)):
    times_temp = raster_times_a[raster_senders_a==neuron]/1000.
    ax9a.scatter(times_temp,ii*np.ones(len(times_temp)),color='grey',s=0.3,linewidth=0,rasterized=False)
ax9a.set_xlabel("Time [s]")
ax9a.set_yticks([])
f.spines(ax9a)
ax9a.spines['left'].set_visible(False)
fig.text(0.01,0.195,"After",rotation='vertical')
ax9a.text(2497.5,-15,"assembly",rotation='vertical')
rectangle = patches.Rectangle((2497.8,0),0.1,par.assembly_size,color=color_ass,clip_on=False)
ax9a.add_patch(rectangle)


# Panel H
ax8.bar(0-bar_width/2,height=np.mean(readout_ass_before_plastic),
                      width=bar_width,
                      yerr=np.std(readout_ass_before_plastic),
                      edgecolor=color_ass,
                      color='white')
ax8.bar(0+bar_width/2,height=np.mean(readout_exc_before_plastic),
                      width=bar_width,
                      yerr=np.std(readout_exc_before_plastic),
                      edgecolor=color_exc,
                      color='white')   
ax8.bar(1-bar_width/2,height=np.mean(readout_ass_after_plastic),
                      width=bar_width,
                      yerr=np.std(readout_ass_after_plastic),
                      edgecolor=color_ass,
                      color='white',
                      label='within\nassembly')
ax8.bar(1+bar_width/2,height=np.mean(readout_exc_after_plastic),
                      width=bar_width,
                      yerr=np.std(readout_exc_after_plastic),
                      edgecolor=color_exc,
                      color='white',
                      label='outside\nassembly')
ax8.legend(bbox_to_anchor=(2.2, 1.3))
ax8.set_ylabel("Readout rate [Hz]")
ax8.set_xticks([0,1])
ax8.set_xticklabels(['Before','After'])
f.spines(ax8)

fig.text(0.01,0.96,'A')
fig.text(0.01,0.63,'B')
fig.text(0.31,0.96,'C')
fig.text(0.31,0.6,'D')
fig.text(0.52,0.9,'E')
fig.text(0.53,0.6,'F')
fig.text(0.75,0.96,'H')
fig.text(0.76,0.61,'G')

fig.set_size_inches(7,3.5)

plt.savefig(par.path_to_figure[where]+"figure_assembly_"+mode+"_"+str(stim_idx)+".svg",dpi=600)
