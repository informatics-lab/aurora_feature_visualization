#!/bin/bash

# Path to your Python script (update if necessary)
SCRIPT="src/feature_visualisation.py"

for i in {0..14}; do
    echo "Processing neuron index $i..."
    python "$SCRIPT" --neuron_idx "$i" --save_output --no-plot_output --n_epochs 300
done

echo "Processing complete."
