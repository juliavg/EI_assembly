import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.colors as colors 
import matplotlib.cm as cmx

matplotlib.rcParams.update({'font.size': 7})

direc = 'data/single_neuron/'

color_shade = (0.9,0.9,0.9)
#colors_J    = np.array([[116,196,118],[65,171,93],[35,139,69],[0,109,44],[0,68,27]])/255.
#colors_J = np.array([[217,240,163],[173,221,142],[120,198,121],[65,171,93],[35,132,67]])/255.
colors_J = np.array([[252,187,161],[252,146,114],[251,106,74],[222,45,38],[165,15,21]])/255.
#colors_J = np.array([[255,237,160],[254,178,76],[252,78,42]])/255.
colors_rates = np.array([[228,26,28],[55,126,184],[65,171,93],[152,78,163],[255,127,0]])/255.


stim_strength_all = np.array([1,2,3,4,5])

fig = plt.figure(figsize=(7,3))
#ax1 = fig.add_axes([0.1,0.6,0.125,0.3])
ax2 = fig.add_axes([0.325,0.65,0.125,0.3])
ax3 = fig.add_axes([0.55,0.65,0.125,0.3])
ax4 = fig.add_axes([0.775,0.65,0.125,0.3])
ax4b = ax4.twinx()
ax5 = fig.add_axes([0.1,0.15,0.125,0.3])
ax5b = fig.add_axes([0.25,0.15,0.01,0.3])
ax6 = fig.add_axes([0.4,0.15,0.125,0.3])
ax6b = fig.add_axes([0.55,0.15,0.01,0.3])
ax7 = fig.add_axes([0.72,0.15,0.125,0.3])

# Panel 1
ii = 3
weight = np.load(direc+"data_rate/mean_weight_"+str(ii)+".npy")
time = np.load(direc+"data_rate/time_bins_"+str(ii)+".npy")[:-1]/1000.
ax2.plot(time,weight,color='grey')
ax2.set_xlabel("Time [s]")
ax2.set_ylabel(r"$W_{I \to E}$ [pA]")
ax2.axvspan(time[int(time.shape[0]/2)],time[-1],color=color_shade)
ax2.spines['right'].set_visible(False)
ax2.spines['top'].set_visible(False)


# Panel 2
n_neurons = 10
rate = np.load(direc+"data_rate/rate_series_1_"+str(ii)+".npy")
for nn in np.arange(n_neurons-1):
    rate += np.load(direc+"data_rate/rate_series_"+str(nn+2)+"_"+str(ii)+".npy")
ax3.plot(time,rate/n_neurons,color='grey')
ax3.set_xlabel("Time [s]")
ax3.set_ylabel("Firing rate [Hz]")
ax3.axvspan(time[int(time.shape[0]/2)],time[-1],color=color_shade)
ax3.spines['right'].set_visible(False)
ax3.spines['top'].set_visible(False)

mean_rate_panel2 = np.mean(rate[int(rate.shape[0]/2):])/n_neurons

# Panel 3
width = 0.3
for ss in np.arange(stim_strength_all.shape[0]):
    rate = []
    cv   = []
    for nn in np.arange(n_neurons):
        extension = "_"+str(nn+1)+"_"+str(ss)+".npy"
        rate.append(np.load(direc+"data_rate/rate_final"+extension))
        cv.append(np.load(direc+"data_rate/cv"+extension))
    ax4.bar(ss-width/2,height=np.mean(rate),width=width,edgecolor=colors_J[ss],yerr=np.std(rate),facecolor='white')
    ax4b.bar(ss+width/2,height=np.mean(cv),width=width,color=colors_J[ss],yerr=np.std(cv))

ax4.set_xticks(np.arange(stim_strength_all.shape[0]))
ax4.set_xticklabels(['J','2J','3J','4J','5J','6J','7J'])
ax4.set_ylabel("Firing rate [Hz]")
ax4b.set_ylabel("CV")
ax4.set_xlabel(r"$W_{E \to E}$")

# Panel 4
mu_values = np.load(direc+"data_theory/mu_values.npy")
std_values = np.load(direc+"data_theory/std_values.npy")
rate = np.load(direc+"data_theory/rate.npy")*1000.
CV_all = np.load(direc+"data_theory/CV_all.npy")
CV_contours = np.load(direc+"data_theory/CV_contours.npy",allow_pickle=True)
CV_contours = CV_contours[()]
CS_rates = np.load(direc+"data_theory/CS_rates.npy",allow_pickle=True)
#CS_rates = CS_rates[()]

rates_contour = np.array([1,10,50])


rmin = 0
rmax = np.max(rate)
X,Y  = np.meshgrid(mu_values,std_values)
im   = ax5.pcolor(X,Y,rate.T,cmap='viridis',vmin=rmin,vmax=rmax,rasterized=True)
cbar = fig.colorbar(im,cax=ax5b)
ax5.contour(X,Y, rate.T,levels=rates_contour,colors='k',linewidths=1)

for ss in np.arange(stim_strength_all.shape[0]):
    vm = np.load(direc+"data_vm/vm_"+str(ss)+".npy")
    mean_vm = np.mean(vm[int(vm.shape[0]/2):])
    std_vm = np.std(vm[int(vm.shape[0]/2):])
    ax5.plot(mean_vm,std_vm,'x',color=colors_J[ss])  

cbar.set_label("Firing rate [Hz]")
ax5.set_xlabel(r"$\mu$ [mV]")
ax5.set_ylabel(r"$\sigma$ [mV]")

# Panel 5
X,Y  = np.meshgrid(mu_values,std_values)
im   = ax6.pcolor(X,Y,CV_all.T,cmap='viridis',rasterized=True)
cbar = fig.colorbar(im,cax=ax6b)
CS   = ax6.contour(X,Y, CV_all.T,colors='k',linewidths=1)

for ss in np.arange(stim_strength_all.shape[0]):
    vm = np.load(direc+"data_vm/vm_"+str(ss)+".npy")
    mean_vm = np.mean(vm[int(vm.shape[0]/2):])
    std_vm = np.std(vm[int(vm.shape[0]/2):])
    ax6.plot(mean_vm,std_vm,'x',color=colors_J[ss])

cbar.set_label("CV")
ax6.set_xlabel(r"$\mu$ [mV]")
ax6.set_ylabel(r"$\sigma$ [mV]")


# Panel 6
cNorm  = colors.Normalize(vmin=0, vmax=rmax)
cm=plt.get_cmap('viridis')
scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cm)

for ii in np.arange(rates_contour.shape[0]): 
    label = rates_contour[ii]
    ax7.plot(CS_rates[ii][0][:,0],CV_contours[ii],color=scalarMap.to_rgba(rates_contour[ii]),label='%.f Hz' %label) 
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

fig.set_size_inches(7,3)
plt.savefig("figures/figure_single_neuron.svg",dpi=300)
plt.show()


