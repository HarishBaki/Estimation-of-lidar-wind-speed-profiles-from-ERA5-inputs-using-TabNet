# Estimation-of-lidar-wind-speed-profiles-from-ERA5-inputs-using-TabNet

** Install hsds in a conda environment **
- Run conda env create -f hsds_environment.yml
- Add the following lines to .bashrc

    `export AWS_S3_GATEWAY=http://s3.us-west-2.amazonaws.com`

    `export AWS_S3_NO_SIGN_REQUEST=1`
- Add the following lines to .hscfg

    `# Local HSDS server`

    `hs_endpoint = http://localhost:5101`

    `hs_bucket = nrel-pds-hsds`

