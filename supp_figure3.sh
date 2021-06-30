#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="local"
MODE="speedup"

for i in {0..2}
do
   STIM_IDX=$i
   MASTER_SEED=$(expr 17 + $i)
   
   source /home/jgallina/nest_custom_env/bin/activate

    # Runs simulation
    python3 ${SCRIPTS}3_simulation_assembly.py $WHERE $MODE $MASTER_SEED $STIM_IDX
done

# Plots
python3 ${SCRIPTS}s3_plot_decay_speedup.py $MODE
