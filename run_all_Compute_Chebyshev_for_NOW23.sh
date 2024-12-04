#!/bin/bash

# Define the range for i
for i in {0..20}; do
    python Compute_Chebyshev_for_NOW23.py $i
done

wait
echo "All done"
