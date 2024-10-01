'''
The objective of this program is to provide the entire pipeline from directly consuming 
Carmen's interpolated data 

'''
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from itertools import chain
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import pickle
import os
import warnings
import netCDF4 as nc


def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


def mask_observations(observations, inversion):
    new_inversion = np.copy(inversion)
    for i in range(len(observations)):
        for j in range(len(observations[0])):
            if np.isnan(observations[i][j]):
                new_inversion[i][j] = np.nan
    return new_inversion

def compute_rho_i(date):
    reference_date = datetime(2010, 1, 1)
    date_format = "%Y%m%d"
    date_obj = datetime.strptime(date, date_format)
    difference = date_obj - reference_date
    nombre_de_jours = difference.days
    
    ice_type = nc.Dataset('/home/erb/masterproject/MSCI_project/snow_ice/carmen/icetype/icetype.nc').variables['Ice Type'][nombre_de_jours]

    rho_MYI = 916.0
    rho_FYI = 882.0

    density_map = np.copy(ice_type)

    for i in range(len(ice_type)):
        for j in range(len(ice_type[0])):
            if ice_type[i][j] == 3.0:
                density_map[i][j] = rho_MYI
            elif ice_type[i][j] == 2.0:
                density_map[i][j] = rho_FYI
    return density_map


'''
FROM DAILY DATA: 
fb_path1 = "../carmen/daily_numpys/AK_CPOM/FB_interp_2016-2017_25km_20170309.npy"
fb_path2 = "../carmen/daily_numpys/CS2_LARM/FB_interp_2016-2017_25km_20170309.npy"

'''
def main(
    month,
    fb_path1,
    fb_path2,
    verbose,
    date,
    window,
    minlat, maxlat,
    minlon, maxlon,
    parametrization,
    initial_cells,
    iterations_number,
    verbosity,
    independent_chains,
    temperature_levels,
    maximum_temperature,
    iterations_between_tempering_attempts,
    skipping,
    thinning,
    render_observations,
         ):


    if verbose:
        print("Starting inversion")
    
    '''
    Step 1: Data cleaning and adapting to the TransTessellate standard
    '''
    #ak = np.load(fb_path1)
    #cs2 = np.load(fb_path2)
    
    ak = []
    cs2 = []

    for k in range(len(date)):
        f = open(fb_path1,'rb')
        AK = pickle.load(f)

        f = open(fb_path2,'rb')
        CS2 = pickle.load(f)


        ak.append(AK[date[k]])
        cs2.append(CS2[date[k]])

    ak_std = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_AK_2p_LARM/error_map/2020-2021/AK_Period_15j.npy')
    cs2_std = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_AK_2p_LARM/error_map/2020-2021/CS2_Period_15j.npy')


    fb1_filename = os.path.split(fb_path1)[1]
    fb2_filename = os.path.split(fb_path2)[1]


    grid_x = np.load("/home/erb/masterproject/MSCI_project/new_x_25km.npy")
    grid_y = np.load("/home/erb/masterproject/MSCI_project/new_y_25km.npy")

    if date[0] == "20210408" :
        ice_type = nc.Dataset('/home/erb/masterproject/MSCI_project/snow_ice/carmen/icetype/CDS_icetype/ice_type_nh_ease2-250_icdr-v3p0_202104151200.nc').variables['ice_type'][0][:,:]

        ice_type = ice_type[::-1,:]
        ice_type = ice_type[36:396, 36:396]

        rho_MYI = 916.0
        rho_FYI = 882.0

        density_map = np.empty((360, 360))

        for i in range(len(ice_type)):
            for j in range(len(ice_type[0])):
                if ice_type[i][j] == 3.0 :
                    density_map[i][j] = rho_MYI
                elif ice_type[i][j] == 2.0 or ice_type[i][j] == 4.0:
                    density_map[i][j] = rho_FYI
                else :
                    density_map[i][j] = np.nan

    else : 
        density_map = compute_rho_i(date[7])


    epsilon = 1e-4

    ak_std[ak_std < epsilon] = 0.01
    cs2_std[cs2_std < epsilon] = 0.01

    ak_observations = []
    cs2_observations = []

    for k in range(len(date)):
        for i in range (360):
            for j in range (360):
                if not np.isnan(ak[k][i][j]) and not np.isnan(density_map[i][j]):
                    if ak_std[i][j] > 0:
                        ak_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, ak[k][i][j], ak_std[i][j], month, density_map[i][j]])
                    else : 
                        ak_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, ak[k][i][j], 0.01, month, density_map[i][j]])

                if not np.isnan(cs2[k][i][j]) and not np.isnan(density_map[i][j]):
                    if cs2_std[i][j] > 0:
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], cs2_std[i][j], month, density_map[i][j]])
                    else :                         
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], 0.01, month, density_map[i][j]])


    cs2_observations.extend(ak_observations)
    data = pd.DataFrame(cs2_observations, columns=["Longitude", "Latitude", "Type", "Value", "StdDev", "Month", "Rho_i"])


    if render_observations:
        fig, ax = plt.subplots(1, 2, figsize=(15, 12))
        
        data[data['Type']==0].plot(kind='scatter', x='Longitude', y='Latitude', c='Value', cmap='seismic', ax=ax[0], title=f'{fb1_filename}')
        data[data['Type']==1].plot(kind='scatter', x='Longitude', y='Latitude', c='Value', cmap='seismic', ax=ax[1], title=f'{fb2_filename}')

        plt.show()
    
    observations_matrix_subset = data.values

    np.savetxt("observations.txt", observations_matrix_subset, '%5.1f %5.1f %d %5.5f %5.5f %d %d')

    # Add the total number of observations at the top of the file
    with open('observations.txt', 'r') as original: data = original.read()
    with open('observations.txt', 'w') as modified: modified.write(f"{observations_matrix_subset.shape[0]}\n" + data)

    '''
    Step 2: Performing inversion
    '''
    # Hyperparameters
    # number_of_processes = 1
    #parametrization = 0 # 0 for Voronoi, 1 for Delaunay linear, 2 for Delaunay Clough-Tocher
    #iterations_number = 100000
    #verbosity = 50000
    #
    # Run inversion
    subprocess.run([
                "mpirun", "-np", str(independent_chains * temperature_levels),
                "./snow_icept", 
                "-i", "observations.txt", 
                "-o", "results/", 
                "-P", "priors/prior_snow.txt",
                "-P", "priors/prior_ice.txt", 
                "-M", "priors/positionprior_snow.txt", 
                "-M", "priors/positionprior_ice.txt",
                #"-H", "priors/hierarchical_snow.txt", 
                #"-H", "priors/hierarchical_ice.txt", 
                "-C", str(initial_cells), "-C", str(initial_cells),
                "-T", str(4000),
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization),
                "-t", str(iterations_number), 
                "-v", str(verbosity),
                "-c", str(independent_chains),    # Independent chains to run at each temperature
                "-K", str(temperature_levels),    # Number of temperature levels for parallel tempering
                "-m", str(maximum_temperature),  # Maximum temperature for the parallel tempering log temperature
                "-e", str(iterations_between_tempering_attempts)    # Number of iterations between parallel tempering exchange attempts
                ])

    # Step 3: Compute means 
    parameter_W = 360
    parameter_H = 360

    file_snow = "images/" + date[0] + ".npy" + "_snow"
    file_ice = "images/" + date[0] + ".npy" + "_ice"


    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_snow,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_snow + "_stddev"),
                #"-m", str(file_snow + "_median"),
                #"-V", str(file_snow + "_image"),
                "-I", str(0)])

    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_ice,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_ice + "_stddev"),
                #"-m", str(file_ice + "_median"),
                #"-V", str(file_ice + "_image"),
                "-I", str(1)])            
       
## Parameters for the inversion
date     = ["20210408", "20210409", "20210410", "20210411", "20210412", "20210413", "20210414", "20210415", "20210416", "20210417", "20210418", "20210419",
            "20210420", "20210421", "20210422"]
#date = ["20190415"]
month    = 6 #Number of month since October (ex : 3 for January)
window = 30
fb_path1 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/25km_ease_binned/AK_LARM/dailyFB_25km_2020-2021_season.pkl"
fb_path2 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/25km_ease_binned/CS2_LARM/dailyFB_25km_2020-2021_season.pkl"
verbose=False
minlat = -5000000.0/1000000
maxlat = 5000000.0/1000000
minlon = -5000000.0/1000000
maxlon = 5000000.0/1000000
parametrization = 1
initial_cells = 1500
iterations_number = 1500000
verbosity = 5000
independent_chains = 4
temperature_levels = 1
maximum_temperature = 2.0
iterations_between_tempering_attempts = 10
skipping = 500000
thinning = 5
render_observations=False


if __name__ == "__main__":
    main(month, fb_path1, fb_path2, verbose, date, window, minlat, maxlat, minlon, maxlon, parametrization, initial_cells, iterations_number, verbosity, 
         independent_chains, temperature_levels, maximum_temperature, iterations_between_tempering_attempts, skipping, thinning, render_observations)
