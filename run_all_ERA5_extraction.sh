#!/bin/bash

# Create a bash for loop for different stations
MAX_CONCURRENT_PROCESSES=48  # Limit to 10 concurrent processes
process_count=0  # Counter to track running processes

# Surface parameters array in bash
par_names=('u10' 'v10' 'u100' 'v100' 'zust' 'i10fg' 
           't2m' 'skt' 'stl1' 'd2m' 'msl' 'blh' 'cbh'
           'ishf' 'ie' 'tcc' 'lcc' 'cape' 'cin' 'bld')

# Example loop to iterate through surface parameters
for par_name in "${par_names[@]}"; do
    echo "Processing surface parameter: $par_name"
    # Call your python script here, passing the current surface parameter
    for year in $(seq 2018 2023); do
        python Extracting_ERA5_variables.py $year $par_name &

        # Increment process count
        ((process_count++))
        # Wait if we've reached the max number of concurrent processes
        if ((process_count >= MAX_CONCURRENT_PROCESSES)); then
            wait
            process_count=0  # Reset counter after processes complete
        fi
    done
done

# Example loop to iterate through pressure parameters and levels
# Pressure levels array in bash
lvls=(1000 975 950)
# Parameters for pressure levels
pars=('u' 'v' 't')
for par in "${pars[@]}"; do
    for level in "${lvls[@]}"; do
        echo "Processing pressure parameter: $par at level $level"
        # Call your python script here, passing the current pressure parameter and level
        for year in $(seq 2018 2023); do
            python Extracting_ERA5_variables.py $year $par $level &
            # Increment process count
            ((process_count++))
            # Wait if we've reached the max number of concurrent processes
            if ((process_count >= MAX_CONCURRENT_PROCESSES)); then
                wait
                process_count=0  # Reset counter after processes complete
            fi
        done
    done
done
wait

# Combine all the data
python combining_yearly_ERA5.py

# Wait for all processes to complete
wait
echo "All processes have completed"