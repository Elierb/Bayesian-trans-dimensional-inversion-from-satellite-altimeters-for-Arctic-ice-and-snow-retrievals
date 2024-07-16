
'''
The objective of this program is to provide the entire pipeline from directly consuming 
Carmen's interpolated data 

This time including support for multi-threading.
'''
import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from itertools import chain
from mpl_toolkits.basemap import Basemap
import pickle
import os
import warnings
import netCDF4 as nc

## Parameters for the inversion
date     = ["20190413", "20190414", "20190415", "20190416", "20190417", "20190418"]
month    = 7 #Number of month since October (ex : 3 for January)
window = 30
fb_path1 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/freeboard_daily_processed/IS2/dailyFB_25km_2018-2019_season.pkl"
fb_path2 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/freeboard_daily_processed/CS2_LARM/dailyFB_25km_2018-2019_season.pkl"
verbose=False
minlat = -5000000.0/1000000
maxlat = 5000000.0/1000000
minlon = -5000000.0/1000000
maxlon = 5000000.0/1000000
parametrization = 1
initial_cells = 1000
iterations_number = 1250000
verbosity = 5000
independent_chains = 4
temperature_levels = 1
maximum_temperature = 2.0
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

'''
FROM DAILY DATA: 
fb_path1 = "../carmen/daily_numpys/AK_CPOM/FB_interp_2016-2017_25km_20170309.npy"
fb_path2 = "../carmen/daily_numpys/CS2_CPOM/FB_interp_2016-2017_25km_20170309.npy"

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
    Creation of a file containing the parameters of the inversion
    '''
    parameter_path = "/home/erb/masterproject/MSCI_project/snow_ice/CS_IS_2p_LARM/images/parameters.txt"
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

    is2_std = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/error_map/2018-2019/IS2_Period_7j.npy')
    cs2_std = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/error_map/2018-2019/CS2_Period_7j.npy')


    fb1_filename = os.path.split(fb_path1)[1]
    fb2_filename = os.path.split(fb_path2)[1]


    grid_x = np.load("/home/erb/masterproject/MSCI_project/new_x_25km.npy")
    grid_y = np.load("/home/erb/masterproject/MSCI_project/new_y_25km.npy")


    #is2_mean = np.nanmean(is2, axis = 0)
    #cs2_mean = np.nanmean(cs2, axis = 0)


    is2_observations = []
    cs2_observations = []

    for k in range(len(date)):
        for i in range (360):
            for j in range (360):
                if not np.isnan(is2[k][i][j]):
                    if is2_std[i][j] > 0:
                        is2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, is2[k][i][j], is2_std[i][j], month])
                    else : 
                        is2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, is2[k][i][j], 0.01, month])

                if not np.isnan(cs2[k][i][j]):
                    if cs2_std[i][j] > 0:
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], cs2_std[i][j], month])
                    else :                         
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], 0.01, month])




    cs2_observations.extend(is2_observations)
    data = pd.DataFrame(cs2_observations, columns=["Longitude", "Latitude", "Type", "Value", "StdDev", "Month"])


    if render_observations:
        fig, ax = plt.subplots(1, 2, figsize=(15, 12))
        
        data[data['Type']==0].plot(kind='scatter', x='Longitude', y='Latitude', c='Value', cmap='seismic', ax=ax[0], title=f'{fb1_filename}')
        data[data['Type']==1].plot(kind='scatter', x='Longitude', y='Latitude', c='Value', cmap='seismic', ax=ax[1], title=f'{fb2_filename}')

        plt.show()
    
    observations_matrix_subset = data.values


    np.savetxt("observations.txt", observations_matrix_subset, '%5.1f %5.1f %d %5.5f %5.5f %d')

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
                #"-I", "results/finalmodel.txt.A-000"
                "-o", "results/", 
                "-P", "priors/prior_snow.txt",
                "-P", "priors/prior_ice.txt", 
                "-M", "priors/positionprior_snow.txt", 
                "-M", "priors/positionprior_ice.txt",
                #"-H", "priors/hierarchical_snow.txt", 
                #"-H", "priors/hierarchical_ice.txt", 
                "-C", str(initial_cells), "-C", str(initial_cells),
                "-T", str(3000),
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


    
    file_snow = f"images_2/" + date[0] + ".npy" +"_snow"
    file_ice = f"images_2/" + date[0] + ".npy" + "_ice"
    
    file_snow_inter = f"images/" + date[0] + "_snow" + ".nc"
    file_ice_inter = f"images/" + date[0] + "_ice" + ".nc"



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
                #"-V", str(file_snow_inter + "_image"),
                #"-g", str(file_snow + "_histo"),
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
                #"-g", str(file_ice + "_histo"),
                #"-V", str(file_ice_inter + "_image"),
                "-I", str(1)])            


if __name__ == "__main__":
    main(month, fb_path1, fb_path2, verbose, date, window, minlat, maxlat, minlon, maxlon, parametrization, initial_cells, iterations_number, verbosity, 
         independent_chains, temperature_levels, maximum_temperature, iterations_between_tempering_attempts, skipping, thinning, render_observations)
