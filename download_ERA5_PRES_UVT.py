import sys
import calendar

# Make changes in the dates only
year = int(sys.argv[2])
month = int(sys.argv[3])

# based on month and year, create an array of strings of days in the month, e.g. ['01', '02', '03', ...]

# Get the number of days in the given month and year
num_days = calendar.monthrange(year, month)[1]

# Create a list of day strings (formatted with leading zeros if necessary)
days = [f"{day:02d}" for day in range(1, num_days + 1)]

# convert year and months into strings, e.g. '2020', '01'
year = str(year)
month = str(month).zfill(2)


###
# Import cdsapi and create a Client instance
import cdsapi
import yaml

cdsapirc_file = sys.argv[1]
with open(cdsapirc_file, 'r') as f:
	credentials = yaml.safe_load(f)
c = cdsapi.Client(url=credentials['url'], key=credentials['key'])

'''
The efficient way to download ERA5 data is to download monthly data-> level type -> all parameters. 
However, the pressure level data is too large to download at once, for the variables z, t, r, u, v.
Therefore, we download the pressure level data for z, t, and r together, and u and v together. 
The single level data is downloaded separately.

For the DL training, we need u, v, and t pressure level data. While all the surface level data can be retrieved at once.
'''

c.retrieve("reanalysis-era5-pressure-levels", {
        "product_type":   "reanalysis",
        "area":           [50, -90, 35, -60],
        "variable":       ["u","v","t"],
        "pressure_level": ["950","975","1000"],
        "year":           year,
        "month":          month,
        "day":            days,
        "time":           ["00","01","02","03","04","05","06","07","08","09","10","11",
                           "12","13","14","15","16","17","18","19","20","21","22","23"],
        "data_format": "netcdf",
        "download_format": "unarchived"
    }, "PRES_UVT_"+year+"_"+month+".nc")

