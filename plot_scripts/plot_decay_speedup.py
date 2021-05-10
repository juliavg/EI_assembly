import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams.update({'font.size': 7})

j_all  = np.array(['1.5','3','5'])
colors = np.array([[77,175,74],[152,78,163],[255,127,0]])/255.

seeds_all = {'static'  : {
             'J5'      : [1500]},
             'plastic' : {
             'J1.5'    : [500,600,700,800,900],
             'J3'      : [1000,1100,1200,1300,1400],
             'J5'      : [0,100,200,300,400]},
             'speedup' : {
             'J1.5'    : [1600],
             'J3'      : [1700],
             'J5'      : [1800]}}

def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig = plt.figure(figsize=(7,3))


ax = fig.add_axes([0.15,0.15,0.8,0.8])

for jj,J in enumerate(j_all):
    direc   = '../data/assembly/plastic/J'+J+'/speed_up/'+str(seeds_all['speedup']['J'+J][0])+'/'
    senders_all = np.load(direc+"sources.npy")
    targets_all = np.load(direc+"targets.npy")
    times   = np.array([])
    senders = np.array([])
    targets = np.array([])
    weights = np.array([])
    for label in ['_post','_decay']:
        events  = np.load(direc+"weight_E"+label+".npy",allow_pickle=True)
        events  = events[()]
        times   = np.concatenate((times,events['times']))
        senders = np.concatenate((senders,events['senders']))
        targets = np.concatenate((targets,events['targets']))
        weights = np.concatenate((weights,events['weights']))
    weights = weights[times>2001000]
    targets = targets[times>2001000]
    senders = senders[times>2001000]
    times   = times[times>2001000]
    for ii in np.arange(5):
        t_plot = times[(senders==senders_all[ii])&(targets==targets_all[ii])]
        w_plot = weights[(senders==senders_all[ii])&(targets==targets_all[ii])]
        ax.plot(t_plot/1000.,w_plot,color=colors[jj])

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$W_{E \to E}$ [pA]")

plt.savefig('../figures/figure_decay_speedup.pdf')

plt.show()
