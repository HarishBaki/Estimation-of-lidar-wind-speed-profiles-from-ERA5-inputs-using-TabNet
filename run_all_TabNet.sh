# === multi output training obs targets ===#

# create a bash for loop for different stations
for station in "PROF_OWEG" "PROF_QUEE"; do
    for transformed in "transformed" "not_transformed"; do
        for loss in "L1_loss" "MSE_loss" "weighted_MSE_loss" "focal_MSE_loss" "profiler_loss"; do
            for Ens in $(seq 0 9); do
                python TabNet_multioutput_profilers.py "$station" "$Ens" "$transformed" "$loss" &
            done
            wait
        done
        wait
    done
    wait
done

echo "All done"