#!/bin/bash
# conda activate cdsapi
main_dir=$(pwd)

target_dir='/data/harish/WRF-PBL-Scheme-Evaluation-with-Chebyshev-Approximation-and-Lidar-Wind-Profiles/Input_DATA'
year=2023
mkdir -p $target_dir"/"$year
cd $target_dir"/"$year
    for month in $(seq 7 1 9)
    do
        if ls "PRES_SC_"$year"_"$month".grb" \
            && ls "PRES_UVT_"$year"_"$month".grb" \
            && ls "SFC_"$year"_"$month".grb" 1>/dev/null 2>&1;
        then
            x_files=($(ls "PRES_SC_"$year"_"$month".grb" "PRES_UVT_"$year"_"$month".grb" "SFC_"$year"_"$month".grb"))
            echo "All files exist:"$(ls -l ${x_files[*]})
        else
            echo "One or more files are missing" 
            python $main_dir/download_ERA5.py '/home/harish/.cdsapirc_hbakitudelft' $year $month &
        fi
    done
    wait
cd $main_dir

echo "All files downloaded"