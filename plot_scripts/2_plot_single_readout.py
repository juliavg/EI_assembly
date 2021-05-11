import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.rcParams.update({'font.size': 7})

direc = 'data/single_readout/'

def spines(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

fig = plt.figure(figsize=(7,3))

ax2 = fig.add_axes([0.375,0.65,0.125,0.3])
ax3 = fig.add_axes([0.6,0.65,0.125,0.3])
ax4 = fig.add_axes([0.825,0.65,0.125,0.3])
ax5 = fig.add_axes([0.375,0.15,0.125,0.3])
ax6 = fig.add_axes([0.6,0.15,0.125,0.3])
ax7 = fig.add_axes([0.825,0.15,0.125,0.3])

def plot_row(direc,axes):
    times_vm     = np.load(direc+"times_vm.npy",allow_pickle=True)
    vm           = np.load(direc+"vm.npy",allow_pickle=True)
    final_weight = np.load(direc+"final_weight.npy")
    rate_out     = np.load(direc+"rate_out.npy")
    cv_out       = np.load(direc+"cv_out.npy")
    cv_in        = np.load(direc+"cv_in.npy")
    mean_cc      = np.load(direc+"mean_cc.npy")
    CV_all       = np.load(direc+"CV_all.npy")

    times_vm     = times_vm[()]
    vm           = vm[()]

    mean_vm = np.zeros(len(CV_all))
    std_vm  = np.zeros(len(CV_all))
    for ii in np.arange(len(CV_all)):
        mean_vm[ii] = np.mean(vm[ii][int(len(vm[ii])/2):])
        std_vm[ii]  = np.std(vm[ii][int(len(vm[ii])/2):])

    axes[0].plot(np.mean(cv_in,axis=1),mean_vm)
    axes[0].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[0].set_ylabel(r"$\mu_{V_m}$")
    spines(axes[0])

    axes[1].plot(np.mean(cv_in,axis=1),std_vm)
    axes[1].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[1].set_ylabel(r"$\sigma_{V_m}$")
    spines(axes[1])

    axes[2].plot(np.mean(cv_in,axis=1),rate_out)
    axes[2].set_xlabel(r"$\mathregular{CV_{in}}$")
    axes[2].set_ylabel("Rate [Hz]")
    spines(axes[2])

plot_row(direc+'data_stp/',[ax5,ax6,ax7])
plot_row(direc+'data_static/',[ax2,ax3,ax4])

ax2.set_ylim(ax5.get_ylim())
ax3.set_ylim(ax6.get_ylim())
ax4.set_ylim(ax7.get_ylim())

fig.text(0.03,0.96,'A')
fig.text(0.305,0.96,'B')
fig.text(0.53,0.96,'C')
fig.text(0.755,0.96,'D')

fig.text(0.01,0.69,'No plasticity',rotation='vertical')
fig.text(0.01,0.11,'Short term facilitation',rotation='vertical')


plt.savefig("figures/figure_single_readout.svg")
