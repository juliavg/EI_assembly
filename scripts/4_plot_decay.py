import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 7})

j_all  = np.array(['1.5','3','5'])
seeds  = np.array([[500,600,700,800,900],[1000,1100,1200,1300,1400],[0,100,200,300,400]])
colors = np.array([[77,175,74],[152,78,163],[255,127,0]])/255.


def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig = plt.figure(figsize=(7,3))


ax2 = fig.add_axes([0.325,0.65,0.125,0.3])
ax3 = fig.add_axes([0.55,0.65,0.125,0.3])
ax4 = fig.add_axes([0.775,0.65,0.125,0.3])
ax6 = fig.add_axes([0.325,0.15,0.125,0.3])
ax7 = fig.add_axes([0.55,0.15,0.125,0.3])
ax8 = fig.add_axes([0.775,0.15,0.125,0.3])


cv_no_shift = np.array([])
cc_no_shift = np.array([])
slope_no_shift = np.array([])
cv_shift = np.array([])
cc_shift = np.array([])
slope_shift = np.array([])
for jj,J in enumerate(j_all):
    direc_single = '../data/assembly/plastic/J'+J+'/'+str(seeds[jj][0])+'/'
    direc_all    = '../data/assembly/plastic/J'+J+'/'
    times = np.array([])
    senders = np.array([])
    targets = np.array([])
    weights = np.array([])
    for label in ['_post','_decay']:
        events  = np.load(direc_single+"weight_E"+label+".npy",allow_pickle=True)
        events  = events[()]
        times   = np.concatenate((times,events['times']))
        senders = np.concatenate((senders,events['senders']))
        targets = np.concatenate((targets,events['targets']))
        weights = np.concatenate((weights,events['weights']))

    all_sources = np.load(direc_single+'sources.npy')
    all_targets = np.load(direc_single+'targets.npy')
    
    for ii in np.arange(5):
        t_plot = times[(senders==all_sources[ii])&(targets==all_targets[ii])]
        w_plot = weights[(senders==all_sources[ii])&(targets==all_targets[ii])]
        ax2.plot(t_plot/1000.,w_plot,color=colors[jj])
    
    data  = np.load(direc_all+"data_triplets_"+J+".npy")
    cv    = np.mean(data[:,:,2:4],axis=2)
    ax3.plot(cv[:,0],data[:,0,5],'x',color=colors[jj])
    ax4.plot(data[:,0,4],data[:,0,5],'x',color=colors[jj])
    
    cv_no_shift = np.concatenate((cv_no_shift,cv[:,0]))
    slope_no_shift = np.concatenate((slope_no_shift,data[:,0,5]))
    cc_no_shift    = np.concatenate((cc_no_shift,data[:,0,4]))
    
    weight_offline_all = np.load(direc_all+"weight_offline_"+J+".npy",allow_pickle=True)
    weight_offline_all = weight_offline_all[()]
    
    for ii in np.arange(data.shape[0]):
        weight_offline = weight_offline_all[ii]
        ax6.plot(np.array(list(weight_offline.keys()))/1000.,list(weight_offline.values()),color=colors[jj],alpha=0.5)
    
    cv_all_shift    = np.array([])
    slope_all_shift = np.array([])
    cc_all_shift    = np.array([])
    for ss in np.arange(1,data.shape[1],1):
        cv_all_shift = np.concatenate((cv_all_shift,cv[:,ss]))
        slope_all_shift = np.concatenate((slope_all_shift,data[:,ss,5]))
        cc_all_shift = np.concatenate((cc_all_shift,data[:,ss,4]))
    
    ax7.plot(cv_all_shift,slope_all_shift,'.',color=colors[jj],alpha=0.5)
    ax8.plot(cc_all_shift,slope_all_shift,'.',color=colors[jj],alpha=0.5)
    
    cv_shift = np.concatenate((cv_shift,cv_all_shift))
    slope_shift = np.concatenate((slope_shift,slope_all_shift))
    cc_shift = np.concatenate((cc_shift,cc_all_shift))
    


ax3.text(0.8,-6.5e-6,'r = %.2f' %np.corrcoef(cv_no_shift,slope_no_shift)[0,1])
ax4.text(0.04,-6.e-6,'r = %.2f' %np.corrcoef(cc_no_shift,slope_no_shift)[0,1])
ax7.text(0.8,-7.e-6,'r = %.2f' %np.corrcoef(cv_shift,slope_shift)[0,1])
ax8.text(0.0055,-7.e-6,'r = %.2f' %np.corrcoef(cc_shift,slope_shift)[0,1])

ax2.set_xlabel("Time [s]")
ax2.set_ylabel(r"$W_{E \to E}$ [pA]")
ax6.set_xlabel("Time [s]")
ax6.set_ylabel(r"$W_{E \to E}$ [pA]")

ax3.set_xlabel("CV")
ax4.set_xlabel("CC")
ax6.set_xlabel("Time [s]")
ax6.set_ylabel(r"$W_{E \to E}$ [pA]")
ax7.set_xlabel("CV")
ax8.set_xlabel("CC")

def slope_subplot(ax):
    ax.ticklabel_format(axis='y',style='sci',scilimits=(-3,4))
    ax.set_ylabel("Slope")
    spines(ax)

spines(ax2)
slope_subplot(ax3)
slope_subplot(ax4)
spines(ax6)
slope_subplot(ax7)
slope_subplot(ax8)

fig.text(0.05,0.95,'Original spike trains')
fig.text(0.05,0.45,'Shifted spike trains')

plt.savefig('../figures/figure_decay.pdf')

plt.show()
