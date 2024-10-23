# === multi-output training obs targets === #

# Create a bash for loop for different stations
MAX_CONCURRENT_PROCESSES=30  # Limit to 10 concurrent processes
gpu_devices=(0 1 2)          # Array of available GPU devices
gpu_count=${#gpu_devices[@]} # Count the number of GPUs available

process_count=0  # Counter to track running processes

for station in "PROF_QUEE" "PROF_BRON" "PROF_STAT" "PROF_OWEG" "PROF_REDH"; do
    for hourly_data_method in "Averaged_over_55th_to_5th_min"; do
        for segregated in "segregated"; do
            for transformed in "not_transformed"; do
                for loss in "Kho_loss_on_profile"; do
                    for Ens in $(seq 0 9); do
                        # Select the GPU device in a round-robin manner
                        gpu_device=${gpu_devices[$((process_count % gpu_count))]}
                        echo "Running station $station hourly_data_method $hourly_data_method segregated $segregated transformed $transformed loss $loss Ens $Ens on GPU $gpu_device"
                        # Run the Python script
                        python TabNet_multioutput_profilers.py "$station" "$hourly_data_method" "$segregated" "$transformed" "$loss" "$Ens" "$gpu_device" &

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
        done
    done
done

# Wait for all remaining background processes to finish
wait

echo "All done"
