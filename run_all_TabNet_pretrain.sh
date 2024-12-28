# === multi-output training obs targets === #

# Create a bash for loop for different stations
MAX_CONCURRENT_PROCESSES=6  # Limit to 10 concurrent processes
gpu_devices=(0)          # Array of available GPU devices
gpu_count=${#gpu_devices[@]} # Count the number of GPUs available

process_count=0  # Counter to track running processes

ranges=(
    "('2020-01-01T00:00:00', '2020-12-31T23:00:00')"
)
#("L1_loss" "MSE_loss" "profiler_loss" "Kho_loss" "Kho_loss_on_profile")
losses=("MSE_loss" "Kho_loss")

for range in "${ranges[@]}"; do
    for transformed in "transformed"; do
        for loss in "${losses[@]}"; do
            for Ens in $(seq 0 9); do
                # Select the GPU device in a round-robin manner
                gpu_device=${gpu_devices[$((process_count % gpu_count))]}
                echo "train_range "$range" transformed $transformed loss $loss Ens $Ens on GPU $gpu_device"
                # Run the Python script
                python pretrain_TabNet_multioutput_CERRA.py "$range" "$transformed" "$loss" "$Ens" "$gpu_device" &

                # Increment process count
                ((process_count++))

                # Wait if we haveve reached the max number of concurrent processes
                if ((process_count >= MAX_CONCURRENT_PROCESSES)); then
                    wait
                    process_count=0  # Reset counter after processes complete
                fi
            done
        done
    done
done

# Wait for all remaining background processes to finish
wait

echo "All done"
