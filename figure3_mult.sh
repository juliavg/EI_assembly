#!/bin/bash

SCRIPTS="$(pwd)/scripts/"
WHERE="brain5"
MODE="plastic"

for i in {0..2}
do
   STIM_IDX=$i
   for j in {0..4}
   do
    MASTER_SEED=$(expr 5 \* $i + $j)
    
    source /home/jgallina/nest_custom_env/bin/activate

    # Runs simulation
    python3 ${SCRIPTS}3_simulation_assembly.py $WHERE $MODE $MASTER_SEED $STIM_IDX


    done
done

# Plots
python3 ${SCRIPTS}3_plot_assembly.py $MODE $STIM_IDX
