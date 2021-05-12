#!/bin/bash

for i in {0..2}
do
   for j in {0..4}
   do

    SCRIPTS="$(pwd)/scripts/"
    WHERE="brain5"
    MODE="plastic"
    MASTER_SEED=$j
    STIM_IDX=$i
    SPEED="normal"
    
    source /home/jgallina/nest_custom_env/bin/activate

    # Runs simulation
    python3 ${SCRIPTS}3_simulation_assembly.py $WHERE $MODE $MASTER_SEED $STIM_IDX $SPEED


    done
done

# Plots
python3 ${SCRIPTS}3_plot_assembly.py $WHERE $MODE
