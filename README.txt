This is the code used for generating the figures on the paper

###############################################

In order to run it, you need the following:

1) Python modules: numpy, matplotlib, h5py, scipy

2) Network simulator NEST. The simulations were performed on NEST 2.20.0, but with a small alteration on the rule for the tripled-based STDP model. Essentially, a minimum bound for the weights were introduced, as compared to the standard NEST distribution. In order to install this version of NEST, follow the steps:

- Download the source code:
git clone https://github.com/juliavg/nest-simulator.git
cd nest-simulator/
git checkout wmin_triplet

- Compile the downloaded source code by following "Advanced installation" instructions on:
https://nest-simulator.readthedocs.io/en/v3.0/installation/index.html

###############################################

Before running the code:

1) Open support/parameters.py and include the path to where you would like to save simulation data and the figures

###############################################
