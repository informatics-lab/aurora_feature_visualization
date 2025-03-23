#!/bin/bash

# Path to your Python script (update if necessary)
SCRIPT="scripts/experiments/gcs/feature_visualisation_gcs.py"

for i in {0..14}; do
    echo "Processing neuron index $i..."
    python "$SCRIPT" --neuron_idx "$i" --save_output --no-plot_output
done

echo "Processing complete."
