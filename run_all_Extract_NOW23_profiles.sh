#!/bin/bash

# Define the range for i
for i in {0..20}; do
    python Extract_NOW23_profiles_zeroth_method.py $i &
done

wait
echo "All done"