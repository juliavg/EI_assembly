import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import colors
import sys

mode    = sys.argv[1]
which_j = str(sys.argv[2])

seeds_all = {'static'  : {
             'J6'      : [1500]},
             'plastic' : {
             'J1.5'    : [500,600,700,800,900],
             'J3'      : [1000,1100,1200,1300,1400],
             'J5'      : [0,100,200,300,400]},
             'speedup' : {
             'J1.5'    : [1600],
             'J3'      : [1700],
             'J5'      : [1800]}}

seeds = seeds_all[mode][which_j]
direc_all    = '../data/assembly/'+mode+'/'+which_j+'/'
direc_single = direc_all+str(seeds[0])+'/'


matplotlib.rcParams.update({'font.size': 7})


color_ass = np.array([222,203,228])/255.
color_exc = np.array([254,217,166])/255.

color_ass = np.array([179,205,227])/255.
color_exc = np.array([204,235,197])/255.

cmap = 'viridis'
cmap_w = 'seismic'

NE = 1600
NI = 400               # Number of inhibitory neurons
N_neurons = NE + NI   # Total number of neurons in the network
assembly_size = int(NE*0.1)    # Number of neurons to be stimulated

senders         = np.array([])
times           = np.array([])

for label in ['_grow','_stim','_post']:
    events  = np.load(direc_single+"spk_all_neuron"+label+".npy",allow_pickle=True)
    events  = events[()]
    senders = np.concatenate((senders,events['senders']))
    times   = np.concatenate((times,events['times']))

all_senders = np.unique(senders)
ass_id = all_senders[all_senders<assembly_size]
exc_id = all_senders[(all_senders>assembly_size) & (all_senders<(NE+1))]
inh_id = all_senders[all_senders>(NE+1)]

growth_time = 2000000.
stim_time   = 1000.
post_time   = 500000.
total_time  = growth_time+stim_time+post_time

interval = 50000.

raster_time = 3000.
raster_n_neurons = assembly_size

time_before_upp = growth_time
time_before_low = time_before_upp - interval
time_stim_upp   = growth_time + stim_time
time_stim_low   = growth_time
time_post_upp   = growth_time + stim_time + post_time
time_post_low   = time_post_upp - interval


# Analysis
# Rate
rate_before = np.zeros(N_neurons)
rate_stim   = np.zeros(N_neurons)
rate_post   = np.zeros(N_neurons)

cv_before = np.zeros(N_neurons)
cv_stim   = np.zeros(N_neurons)
cv_post   = np.zeros(N_neurons)

def rate_and_cv(times,interval):
    rate = len(times)/interval*1000.
    isi  = np.diff(times)
    cv   = np.std(isi)/np.mean(isi)
    return (rate,cv)

for nn,neuron in enumerate(np.unique(senders)):
    times_neuron = times[senders==neuron]
    rate_before[nn],cv_before[nn] = rate_and_cv(times_neuron[(times_neuron>time_before_low) & (times_neuron<time_before_upp)],interval)
    rate_stim[nn],cv_stim[nn]     = rate_and_cv(times_neuron[(times_neuron>time_stim_low) & (times_neuron<time_stim_upp)],stim_time)
    rate_post[nn],cv_post[nn]     = rate_and_cv(times_neuron[(times_neuron>time_post_low) & (times_neuron<time_post_upp)],interval)

raster_times   = times[np.isin(senders,np.concatenate((ass_id[:raster_n_neurons],exc_id[:raster_n_neurons])))]
raster_senders = senders[np.isin(senders,np.concatenate((ass_id[:raster_n_neurons],exc_id[:raster_n_neurons])))]
raster_senders_b = raster_senders[(raster_times<growth_time) & (raster_times>(growth_time-raster_time))]
raster_times_b = raster_times[(raster_times<growth_time) & (raster_times>(growth_time-raster_time))]
raster_senders_a = raster_senders[(raster_times<total_time) & (raster_times>(total_time-raster_time))]
raster_times_a = raster_times[(raster_times<total_time) & (raster_times>(total_time-raster_time))]


a = 16
b = 10
def create_matrix(array):
    matrix = np.zeros((2*a,5*b))
    for ii in np.arange(10):
        matrix[(ii//5)*a:(ii//5+1)*a,(ii%5)*b:(ii%5+1)*b] = array[ii*a*b:(ii+1)*a*b].reshape(a,b)
    return matrix
    
rate_matrix_before = create_matrix(rate_before[:NE])
rate_matrix_post   = create_matrix(rate_post[:NE])
cv_matrix_before   = create_matrix(cv_before[:NE])
cv_matrix_post     = create_matrix(cv_post[:NE])

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

sources = np.load(direc_single+"sources_grow.npy")-3
targets = np.load(direc_single+"targets_grow.npy")-3
weights = np.load(direc_single+"weights_grow.npy")
matrix_before = np.zeros((N_neurons,N_neurons))
matrix_before[targets,sources] = weights

sources = np.load(direc_single+"sources_post.npy")-3
targets = np.load(direc_single+"targets_post.npy")-3
weights = np.load(direc_single+"weights_post.npy")
matrix_after = np.zeros((N_neurons,N_neurons))
matrix_after[targets,sources] = weights

weight_before = create_mean_matrix(matrix_before[:NE,:])
weight_after = create_mean_matrix(matrix_after[:NE,:])

min_w = min([np.min(weight_before),np.min(weight_after)])
max_w = max([np.max(weight_before),np.max(weight_after)])
norm = colors.DivergingNorm(vmin=min_w, vcenter=0., vmax=max_w)

fig = plt.figure(figsize=(7,3.5))

#ax1  = fig.add_axes([0.1,0.6,0.1,0.35])

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

a = assembly_size
b = NE
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
    #ax9.plot(times_temp,ii*np.ones(len(times_temp)),'o',color='grey',markersize=0.5,linewidth=0)
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
    events           = np.load(direc_all+str(seed)+'/spk_readout_post.npy',allow_pickle=True)
    events           = events[()]
    readout_senders  = events['senders']
    readout_times    = events['times']
    times_ass_plastic = readout_times[(readout_senders==2003)&(readout_times>(max(readout_times)-50000.))]
    times_exc_plastic = readout_times[(readout_senders==2004)&(readout_times>(max(readout_times)-50000.))]
    times_ass_static  = readout_times[(readout_senders==2005)&(readout_times>(max(readout_times)-50000.))]
    times_exc_static  = readout_times[(readout_senders==2006)&(readout_times>(max(readout_times)-50000.))]
    readout_ass_after_plastic.append(len(times_ass_plastic)/50.)
    readout_exc_after_plastic.append(len(times_exc_plastic)/50.)
    readout_ass_after_static.append(len(times_ass_static)/50.)
    readout_exc_after_static.append(len(times_exc_static)/50.)
    times_ass_plastic = readout_times[(readout_senders==2003)&(readout_times>(growth_time-50000.))&(readout_times<growth_time)]
    times_exc_plastic = readout_times[(readout_senders==2004)&(readout_times>(growth_time-50000.))&(readout_times<growth_time)]
    times_ass_static  = readout_times[(readout_senders==2005)&(readout_times>(growth_time-50000.))&(readout_times<growth_time)]
    times_exc_static  = readout_times[(readout_senders==2006)&(readout_times>(growth_time-50000.))&(readout_times<growth_time)]
    readout_ass_before_plastic.append(len(times_ass_plastic)/50.)
    readout_exc_before_plastic.append(len(times_exc_plastic)/50.)
    readout_ass_before_static.append(len(times_ass_static)/50.)
    readout_exc_before_static.append(len(times_exc_static)/50.)
    
ax8.bar(0-width/2,height=np.mean(readout_ass_before_plastic),width=width,yerr=np.std(readout_ass_before_plastic),color=color_ass)
ax8.bar(0+width/2,height=np.mean(readout_exc_before_plastic),width=width,yerr=np.std(readout_exc_before_plastic),color=color_exc)   
ax8.bar(1-width/2,height=np.mean(readout_ass_after_plastic),width=width,yerr=np.std(readout_ass_after_plastic),color=color_ass)
ax8.bar(1+width/2,height=np.mean(readout_exc_after_plastic),width=width,yerr=np.std(readout_exc_after_plastic),color=color_exc)

ax8.bar(2-width/2,height=np.mean(readout_ass_before_static),width=width,yerr=np.std(readout_ass_before_static),color='g')
ax8.bar(2+width/2,height=np.mean(readout_exc_before_static),width=width,yerr=np.std(readout_exc_before_static),color='k')   
ax8.bar(3-width/2,height=np.mean(readout_ass_after_static),width=width,yerr=np.std(readout_ass_after_static),color='g')
ax8.bar(3+width/2,height=np.mean(readout_exc_after_static),width=width,yerr=np.std(readout_exc_after_static),color='k')

ax8.set_ylabel("Readout rate [Hz]")
ax8.set_xticks([0,1])
ax8.set_xticklabels(['Before','After'])

fig.set_size_inches(7,3.5)

plt.savefig('../figures/figure_assembly_'+mode+'_'+which_j+'.svg',dpi=300)

#plt.show()

