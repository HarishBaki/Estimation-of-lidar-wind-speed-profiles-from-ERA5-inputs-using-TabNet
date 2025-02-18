: '
# === multi-output training obs targets === #
# Create a bash for loop for different stations
MAX_CONCURRENT_PROCESSES=9  # Limit to 10 concurrent processes
gpu_devices=(0 1 2)          # Array of available GPU devices
gpu_count=${#gpu_devices[@]} # Count the number of GPUs available

process_count=0  # Counter to track running processes

# Define the different combinations of train_dates_range
#stations="('PROF_CLYM','PROF_OWEG','PROF_STAT','PROF_STON','PROF_QUEE','PROF_SUFF','PROF_BUFF','PROF_BELL','PROF_TUPP','PROF_CHAZ')"
stations=("PROF_CLYM" "PROF_OWEG" "PROF_STAT" "PROF_STON" "PROF_QUEE" "PROF_SUFF" "PROF_BUFF" "PROF_BELL" "PROF_TUPP" "PROF_CHAZ")
ranges=(
    "('2021-01-01T00:00:00', '2023-12-31T23:00:00')"
)
#    "('2022-01-01T00:00:00', '2023-12-31T23:00:00')"
#    "('2021-01-01T00:00:00', '2022-12-31T23:00:00')"
#    "('2021-01-01T00:00:00', '2021-12-31T23:00:00')"
#    "('2022-01-01T00:00:00', '2022-12-31T23:00:00')"
#    "('2023-01-01T00:00:00', '2023-12-31T23:00:00')"

#("r2")
losses=("rmse")

for hourly_data_method in "Averaged_over_55th_to_5th_min"; do
    for range in "${ranges[@]}"; do
        for segregated in "segregated"; do
            for transformed in "not_transformed" "transformed"; do
                for loss in "${losses[@]}"; do
                    for Ens in $(seq 0 9); do
                        # Select the GPU device in a round-robin manner
                        gpu_device=${gpu_devices[$((process_count % gpu_count))]}
                        echo "Running station $stations hourly_data_method $hourly_data_method train_range "$range" segregated $segregated transformed $transformed loss $loss Ens $Ens on GPU $gpu_device" warm_start 0
                        # Run the Python script
                        python AutoML_single_to_stepwise_multioutput.py "$stations" "$hourly_data_method" "$range" "$segregated" "$transformed" "$loss" "$Ens" "$gpu_device" "1" &

                        # Increment process count
                        ((process_count++))

                        # Wait if we have reached the max number of concurrent processes
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
'

stations=("PROF_CLYM" "PROF_OWEG")

for station in "${stations[@]}"; do
    echo "Running station $station"
    python AutoML_single_to_stepwise_multioutput.py "$station" "Averaged_over_55th_to_5th_min" "('2021-01-01T00:00:00', '2023-12-31T23:00:00')" "segregated" "not_transformed" "rmse" "0" "0" "1"
done

echo "All done"