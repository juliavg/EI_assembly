from importlib import reload 
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.colors as colors 
import matplotlib.cm as cmx
import h5py as h5
import sys

direc = sys.argv[0].split('scripts')[0]
where = sys.argv[1]

sys.path.append(direc+'/support')
import parameters
reload(parameters)
import parameters as par


matplotlib.rcParams.update({'font.size': 7})

data = h5.File(par.path_to_data[where]+'data_single_neuron.hdf5','r')

# Plot parameters
bar_width = 0.3

# Colors
color_shade  = (0.9,0.9,0.9)
colors_J     = np.array([[252,187,161],[252,146,114],[251,106,74],[222,45,38],[165,15,21]])/255.
colors_rates = np.array([[228,26,28],[55,126,184],[65,171,93],[152,78,163],[255,127,0]])/255.

# Axes
fig  = plt.figure(figsize=(7,3))
ax2  = fig.add_axes([0.325,0.65,0.125,0.3])
ax3  = fig.add_axes([0.55,0.65,0.125,0.3])
ax4  = fig.add_axes([0.775,0.65,0.125,0.3])
ax4b = ax4.twinx()
ax5  = fig.add_axes([0.1,0.15,0.125,0.3])
ax5b = fig.add_axes([0.25,0.15,0.01,0.3])
ax6  = fig.add_axes([0.4,0.15,0.125,0.3])
ax6b = fig.add_axes([0.55,0.15,0.01,0.3])
ax7  = fig.add_axes([0.72,0.15,0.125,0.3])

# Panel B
ii = 3                                                                      # Selects which stim_strengh to plot weight and rate from
weight = np.array(data['simulation/'+str(par.stim_strength_all[ii])+'/mean_weight'])
time   = par.single_bins[:-1]*1000.
ax2.plot(time,weight,color='grey')
ax2.set_xlabel("Time [s]")
ax2.set_ylabel(r"$W_{I \to E}$ [pA]")
ax2.axvspan(time[int(time.shape[0]/2)],time[-1],color=color_shade)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)

# Panel C
rate = np.array(data['simulation/'+str(par.stim_strength_all[ii])+'/rate_series'])
ax3.plot(time,rate,color='grey')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Firing rate [Hz]")
ax3.axvspan(time[int(time.shape[0]/2)],time[-1],color=color_shade)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

# Panel D
for ss,strength in enumerate(par.stim_strength_all):
    rate = np.array(data['simulation/'+str(strength)+'/rate_final'])
    cv   = np.array(data['simulation/'+str(strength)+'/cv_all'])
    ax4.bar(ss-bar_width/2,height=np.mean(rate),width=bar_width,edgecolor=colors_J[ss],yerr=np.std(rate),facecolor='white')
    ax4b.bar(ss+bar_width/2,height=np.mean(cv),width=bar_width,color=colors_J[ss],yerr=np.std(cv))

ax4.set_xticks(np.arange(par.stim_strength_all.shape[0]))
ax4.set_xticklabels(['J','2J','3J','4J','5J','6J','7J'])
ax4.set_ylabel("Firing rate [Hz]")
ax4b.set_ylabel("CV")
ax4.set_xlabel(r"$W_{E \to E}$")

# Panel E
mu_values   = np.array(data['theory/mu_all'])
std_values  = np.array(data['theory/std_all'])
rate        = np.array(data['theory/rate'])*1000.
CV_all      = np.array(data['theory/CV_all'])
CV_contours = data['theory/CV_contours']
CS_rates    = data['theory/CS_rates']

rates_contour = (par.rates_contour*1000.).astype(int)

rmin = 0
rmax = np.max(rate)
X,Y  = np.meshgrid(mu_values,std_values)
im   = ax5.pcolor(X,Y,rate.T,cmap='viridis',vmin=rmin,vmax=rmax,rasterized=True)
cbar = fig.colorbar(im,cax=ax5b)
ax5.contour(X,Y, rate.T,levels=rates_contour,colors='k',linewidths=1)

for ss,strength in enumerate(par.stim_strength_all):
    vm = np.array(data['simulation/'+str(strength)+'/vm'])
    mean_vm = np.mean(vm[int(vm.shape[0]/2):])
    std_vm = np.std(vm[int(vm.shape[0]/2):])
    ax5.plot(mean_vm,std_vm,'x',color=colors_J[ss]) 
    ax6.plot(mean_vm,std_vm,'x',color=colors_J[ss])

cbar.set_label("Firing rate [Hz]")
ax5.set_xlabel(r"$\mu$ [mV]")
ax5.set_ylabel(r"$\sigma$ [mV]")

# Panel F
X,Y  = np.meshgrid(mu_values,std_values)
im   = ax6.pcolor(X,Y,CV_all.T,cmap='viridis',rasterized=True)
cbar = fig.colorbar(im,cax=ax6b)
CS   = ax6.contour(X,Y, CV_all.T,colors='k',linewidths=1)
cbar.set_label("CV")
ax6.set_xlabel(r"$\mu$ [mV]")
ax6.set_ylabel(r"$\sigma$ [mV]")


# Panel G
cNorm  = colors.Normalize(vmin=0, vmax=rmax)
cm=plt.get_cmap('viridis')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

for ii in np.arange(rates_contour.shape[0]): 
    label = rates_contour[ii]
    ax7.plot(np.array(CS_rates[str(label)][:,0]),np.array(CV_contours[str(label)]),color=scalarMap.to_rgba(rates_contour[ii]),label='%.f Hz' %label) 
ax7.legend(loc=(1.1,0.2))                
ax7.set_xlabel(r'$\mu$ (mV)')
ax7.set_ylabel('CV')

fig.text(0.03,0.96,'A')
fig.text(0.25,0.96,'B')
fig.text(0.48,0.96,'C')
fig.text(0.705,0.96,'D')
fig.text(0.03,0.46,'E')
fig.text(0.33,0.46,'F')
fig.text(0.65,0.46,'G')

data.close()

fig.set_size_inches(7,3)
plt.savefig(par.path_to_figures[where]+"figure_single_neuron.svg",dpi=300)
