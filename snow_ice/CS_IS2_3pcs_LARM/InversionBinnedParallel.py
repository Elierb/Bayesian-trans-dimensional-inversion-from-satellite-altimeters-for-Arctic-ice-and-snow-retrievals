'''
The objective of this program is to provide the entire pipeline from directly consuming CPOM 
data and directly producing plots with the results. 

This time including support for multi-threading.
'''
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from itertools import chain
from mpl_toolkits.basemap import Basemap
from datetime import datetime
import os
import pickle
import warnings
import netCDF4 as nc


## Parameters for the inversion
date     = ["20190408", "20190409", "20190410", "20190411", "20190412", "20190413", "20190414", "20190415", "20190416", "20190417", "20190418", "20190419",
            "20190420", "20190421", "20190422"]
month    = 7 #Number of month since October (ex : 3 for January)
fb_path1 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/IS2_ATLv6/IS2/dailyFB_25km_2018-2019_season.pkl"
fb_path2 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/25km_ease_binned/CS2_LARM/dailyFB_25km_2018-2019_season.pkl"
verbose=False
minlat = -5000000.0/1000000 
maxlat = 5000000.0/1000000
minlon = -5000000.0/1000000
maxlon = 5000000.0/1000000
parametrization = 0
initial_cells = 1500
iterations_number = 1500000
verbosity = 5000
independent_chains = 4
temperature_levels = 1
maximum_temperature = 2
iterations_between_tempering_attempts = 10
skipping = 500000
thinning = 5
render_observations=False


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


def main(
    month,
    fb_path1,
    fb_path2,
    verbose,
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
    Creation of a file containing the parameters of the inversion
    '''
    parameter_path = "/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_3pcs_LARM/images/parameters.txt"
    with open(parameter_path, 'w') as fichier:
        tesselation = "Tesselation = " + str(parametrization) + "\n"
        fichier.writelines(tesselation)
        initialcells = "Initial cells = " + str(initial_cells) + "\n"
        fichier.writelines(initialcells)
        iteration = "Iterations = " + str(iterations_number) + "\n"
        fichier.writelines(iteration)
        skip = "Number of models skipped = " + str(skipping) + "\n"
        fichier.writelines(skip)
        thin = "Thinning = " + str(thinning) + "\n"
        fichier.writelines(thin)
        chains = "Independant chains = " + str(independent_chains) + "\n"
        fichier.writelines(chains)
        temperature = "Temperature levels = " + str(temperature_levels) + "\n"
        fichier.writelines(temperature)
    
    '''
    Step 1: Data cleaning and adapting to the TransTessellate standard
    '''
    #is2 = np.load(fb_path1)
    #cs2 = np.load(fb_path2)
    is2 = []
    cs2 = []

    for k in range(len(date)):
        f = open(fb_path1,'rb')
        IS2 = pickle.load(f)

        f = open(fb_path2,'rb')
        CS2 = pickle.load(f)


        is2.append(IS2[date[k]])
        cs2.append(CS2[date[k]])

    fb1_filename = os.path.split(fb_path1)[1]
    fb2_filename = os.path.split(fb_path2)[1]


    is2_std = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/error_map/2018-2019/IS2_Period_15j.npy')
    cs2_std = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/error_map/2018-2019/CS2_Period_15j.npy')



    grid_x = np.load("/home/erb/masterproject/MSCI_project/new_x_25km.npy")
    grid_y = np.load("/home/erb/masterproject/MSCI_project/new_y_25km.npy")


    is2_observations = []
    cs2_observations = []

    density_map = compute_rho_i(date[7])


    for k in range(len(date)):
        for i in range (360):
            for j in range (360):
                if not np.isnan(is2[k][i][j]) and not np.isnan(density_map[i][j]):
                    if is2_std[i][j] > 0:
                        is2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, is2[k][i][j], is2_std[i][j], month, density_map[i][j]])
                    else : 
                        is2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, is2[k][i][j], 0.01, month, density_map[i][j]])

                if not np.isnan(cs2[k][i][j])and not np.isnan(density_map[i][j]):
                    if cs2_std[i][j] > 0:
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], cs2_std[i][j], month, density_map[i][j]])
                    else :                         
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], 0.01, month, density_map[i][j]])

    cs2_observations.extend(is2_observations)
    data = pd.DataFrame(cs2_observations, columns=["Longitude", "Latitude", "Type", "Value", "StdDev", "Month", "Rho_ice"])




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

    # Run inversion
    subprocess.run([
                "mpirun", "-np", str(independent_chains * temperature_levels),
                "./snow_icept", 
                "-i", "observations.txt", 
                "-o", "results/", 
                "-P", "priors/prior_snow.txt",
                "-P", "priors/prior_ice.txt", 
                "-P", "priors/prior_penetration.txt",
                "-M", "priors/positionprior_snow.txt", 
                "-M", "priors/positionprior_ice.txt",
                "-M", "priors/positionprior_penetration.txt",
                #"-H", "priors/hierarchical_snow.txt", 
                #"-H", "priors/hierarchical_ice.txt",
                #"-H", "priors/hierarchical_penetration.txt", 
                "-C", str(initial_cells), "-C", str(initial_cells), "-C", str(initial_cells),
                "-T", str(4000),
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization), "-A", str(parametrization),
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

    file_snow = f"images/" + date[7] + ".npy" +"_snow"
    file_ice = f"images/" + date[7] + ".npy" + "_ice"
    file_cs_penetration = f"images/" + date[7] + ".npy" + "_penetration"

   
    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_snow,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_snow + "_stddev"),
                "-V", 'images/',#str(file_snow + "_image-000"),
                "-I", str(0)])

    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_ice,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_ice + "_stddev"),
                "-V", 'images/',#str(file_ice + "_image-000"),
                "-I", str(1)])

    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_cs_penetration,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_cs_penetration + "_stddev"),
                "-V", 'images/',#str(file_cs_penetration + "_image-000"),
                "-I", str(2)])
    
if __name__ == "__main__":
    main(month, fb_path1, fb_path2, verbose, minlat, maxlat, minlon, maxlon, parametrization, initial_cells, iterations_number, verbosity, 
         independent_chains, temperature_levels, maximum_temperature, iterations_between_tempering_attempts, skipping, thinning, render_observations)
