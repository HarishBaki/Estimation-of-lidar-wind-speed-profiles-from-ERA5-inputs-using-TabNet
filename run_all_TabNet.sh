# === multi output training obs targets ===#
for j in $(seq 0 9); do
    python TabNet_multioutput_profilers.py "PROF_OWEG" "$j" &
done
wait

echo "All done"