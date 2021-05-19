import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import h5py as h5
import sys
sys.path.append(direc+'support')
import parameters
reload(parameters)
import parameters as par

direc = sys.argv[0].split('scripts')[0]

matplotlib.rcParams.update({'font.size': 7})

#j_all  = np.array(['5'])#np.array(['1.5','3','5'])
colors = np.array([[77,175,74],[152,78,163],[255,127,0]])/255.
'''
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
'''
def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig = plt.figure(figsize=(7,3))


ax = fig.add_axes([0.15,0.15,0.8,0.8])

data = h5.File("data_assembly.hdf5",'r')
data_mode = data['speedup']
j_all = list(data_mode.keys())

for jj,J in enumerate(j_all):
    #direc   = '../data/assembly/plastic/J'+J+'/speed_up/'+str(seeds_all['speedup']['J'+J][0])+'/'
    seed  = list(data_mode[J].keys())
    group = data_mode[J+'/'+seed]
    
    senders_all = np.array(group['sources'])#np.load(direc+"sources.npy")
    targets_all = np.array(group['targets'])#np.load(direc+"targets.npy")
    
    group = group['steps']
    times   = np.array([])
    senders = np.array([])
    targets = np.array([])
    weights = np.array([])
    for label in list(group.keys())[2:]:
        #events  = np.load(direc+"weight_E"+label+".npy",allow_pickle=True)
        #events  = events[()]
        #times   = np.concatenate((times,events['times']))
        times   = np.concatenate((times,group[label+'/times']))
        senders = np.concatenate((senders,group[label+'/senders']))
        targets = np.concatenate((targets,group[label+'/targets']))
        weights = np.concatenate((weights,group[label+'/weights']))
    weights = weights[times>(par.warmup_time+par.stimulation_time)]
    targets = targets[times>(par.warmup_time+par.stimulation_time)]
    senders = senders[times>(par.warmup_time+par.stimulation_time)]
    times   = times[times>(par.warmup_time+par.stimulation_time)]
    for ii in np.arange(len(senders_all)):
        t_plot = times[(senders==senders_all[ii])&(targets==targets_all[ii])]
        w_plot = weights[(senders==senders_all[ii])&(targets==targets_all[ii])]
        ax.plot(t_plot/1000.,w_plot,color=colors[jj])

ax.set_xlabel("Time [s]")
ax.set_ylabel(r"$W_{E \to E}$ [pA]")

plt.savefig(direc+'figure_decay_speedup.pdf')

plt.show()
