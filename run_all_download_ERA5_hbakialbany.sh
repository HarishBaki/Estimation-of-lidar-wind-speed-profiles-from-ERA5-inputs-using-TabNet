#!/bin/bash
# conda activate cdsapi
main_dir=$(pwd)
cdsapi_file='/home/harish/.cdsapirc_hbakialbany'
target_dir='/data/harish/Estimation-of-lidar-wind-speed-profiles-from-ERA5-inputs-using-TabNet/data'
for year in $(seq 2011 4 2017)
do
    mkdir -p $target_dir"/"$year
    cd $target_dir"/"$year
        for month in $(seq 1 1 12)
        do
            # change month to 2 digits
            if [ $month -lt 10 ]
            then
                month="0"$month
            fi
            # --- downloading the PRES_UVT file --- #
            if ls "PRES_UVT_"$year"_"$month".nc" 1>/dev/null 2>&1;
            then
                x_files=($(ls "PRES_UVT_"$year"_"$month".nc"))
                echo "File exists:"$(ls -l ${x_files[*]})
            else
                echo "PRES_UVT File is missing"
                python $main_dir/download_ERA5_PRES_UVT.py $cdsapi_file $year $month
            fi

            # --- downloading the SFC file --- #
            if ls "SFC_"$year"_"$month".nc" 1>/dev/null 2>&1;
            then
                x_files=($(ls "SFC_"$year"_"$month".nc"))
                echo "File exists:"$(ls -l ${x_files[*]})
            else
                echo "SFC File is missing"
                python $main_dir/download_ERA5_SFC.py $cdsapi_file $year $month
            fi
        done
    cd $main_dir
done
echo "All files downloaded"