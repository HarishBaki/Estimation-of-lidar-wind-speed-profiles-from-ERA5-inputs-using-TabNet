# Create a bash for loop for different stations
MAX_CONCURRENT_PROCESSES=30  # Limit to 10 concurrent processes
gpu_devices=(0 1 2)          # Array of available GPU devices
gpu_count=${#gpu_devices[@]} # Count the number of GPUs available

process_count=0  # Counter to track running processes

station="PROF_QUEE"
hourly_data_method="Averaged_over_55th_to_5th_min"
segregated="segregated"
transformed="not_transformed"
loss="Kho_loss_on_profile"
for Ens in $(seq 0 9); do
    data_seed=${RANDOM:0:100}
    for trial in $(seq 0 99); do
        hp_seed=${RANDOM:0:100}
        # Select the GPU device in a round-robin manner
        gpu_device=${gpu_devices[$((process_count % gpu_count))]}
        echo "Running Ens $Ens trial $trial process $process_count on GPU $gpu_device"
        python TabNet_multioutput_profilers_best_parameter_search.py "$station" "$hourly_data_method" "$segregated" "$transformed" "$loss" "$Ens" "$data_seed" "$trial" "$hp_seed" "$gpu_device" & 
        # Increment process count
        ((process_count++))

        # Wait if we've reached the max number of concurrent processes
        if ((process_count >= MAX_CONCURRENT_PROCESSES)); then
            wait
            process_count=0  # Reset counter after processes complete
        fi
    done
done

# Wait for all processes to complete
wait
echo "All processes have completed"