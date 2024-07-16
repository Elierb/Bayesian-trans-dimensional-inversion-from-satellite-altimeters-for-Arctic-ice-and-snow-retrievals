# This code is used to perform the inversion on a certain time scale 

# Libraries 

import numpy as np
import pandas as pd
import subprocess
import matplotlib.pyplot as plt
from itertools import chain
from mpl_toolkits.basemap import Basemap
import os
import warnings
import netCDF4 as nc
import pickle
import datetime

# Main code

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
        #fb_path1 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/daily_numpys/IS2/FB_interp_2019_25km_20190406.npy",
        #fb_path2 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/daily_numpys/CS2_LARM/FB_interp_2018-2019_25km_20190406.npy",
    fb_path1,
    fb_path2,
    month,  
    date,
    verbose,
    minlat,
    maxlat,
    minlon,
    maxlon,
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
    render_observations
         ):

    if verbose:
        print("Starting inversion")

    '''
    Creation of a file containing the parameters of the inversion
    '''
    parameter_path = "/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/images/parameters.txt"
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

    #window = [date]   # This array will contain the 5-days taken for each dates of the loop
    #init_date = 15    # the first date is always the 15th of the month
    #for i in range(3):
    #    date = date[:6]
    #    init_date += 1

    #    date = date + str(init_date)
    #    window.append(date)    # we add the new date in the array

    is2 = []
    cs2 = []

    for k in range(len(date)):   # loop to add the observations from the 5 dates to is2 and cs2 arrays
        f = open(fb_path1,'rb')
        IS2 = pickle.load(f)

        f = open(fb_path2,'rb')
        CS2 = pickle.load(f)


        is2.append(IS2[date[k]])
        cs2.append(CS2[date[k]])


    fb1_filename = os.path.split(fb_path1)[1]
    fb2_filename = os.path.split(fb_path2)[1]


    grid_x = np.load("/home/erb/masterproject/MSCI_project/new_x_25km.npy")
    grid_y = np.load("/home/erb/masterproject/MSCI_project/new_y_25km.npy")

    is2_error = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/error_map/2018-2019/IS2_Period_15j.npy')
    cs2_error = np.load('/home/erb/masterproject/MSCI_project/snow_ice/CS_IS2_4p_LARM/error_map/2018-2019/CS2_Period_15j.npy')

    default_error = 0.01


    is2_observations = []
    cs2_observations = []

    for k in range(len(is2)):
        for i in range (360):
            for j in range (360):
                if not np.isnan(is2[k][i][j]):
                    if is2_error[i][j] > 0:
                        is2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, is2[k][i][j], is2_error[i][j], month])
                    else :
                        is2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 1, is2[k][i][j], default_error, month])
                if not np.isnan(cs2[k][i][j]):
                    if cs2_error[i][j] > 0:
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], cs2_error[i][j], month])
                    else : 
                        cs2_observations.append([grid_x[i][j]/1000000, grid_y[i][j]/1000000, 0, cs2[k][i][j], default_error, month])



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
                "-o", "results/", 
                "-P", "priors/prior_snow.txt",
                "-P", "priors/prior_ice.txt", 
                "-P", "priors/prior_cs2_penetration.txt",
                "-P", "priors/prior_is2_penetration.txt",
                "-M", "priors/positionprior_snow.txt", 
                "-M", "priors/positionprior_ice.txt",
                "-M", "priors/positionprior_cs2_penetration.txt",
                "-M", "priors/positionprior_is2_penetration.txt",
                #"-H", "priors/hierarchical_snow.txt", 
                #"-H", "priors/hierarchical_ice.txt", 
                #"-H", "priors/hierarchical_cs2_penetration.txt", 
                #"-H", "priors/hierarchical_is2_penetration.txt", 
                "-C", str(initial_cells),
                "-T", str(3000),
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-A", str(parametrization), "-A", str(parametrization),
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

    file_snow = f"images/" + date[7] + ".npy" +"_snow"
    file_ice = f"images/" + date[7] + ".npy" + "_ice"
    file_cs2_penetration = f"images/" + date[7] + ".npy" + "cs2_penetration"
    file_is2_penetration = f"images/" + date[7] + ".npy" + "is2_penetration"

    subprocess.run([
                "mpirun", "-np", str(independent_chains),
                "./post_mean_mpi", "-i", 
                "results/ch.dat", "-o", file_snow,
                "-x", str(minlon), "-X", str(maxlon),
                "-y", str(minlat), "-Y", str(maxlat),
                "-s", str(skipping),
                "-t", str(thinning),
                "-A", str(parametrization), "-A", str(parametrization),
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_snow + "_stddev"),
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
                "-A", str(parametrization), "-A", str(parametrization),
                "-W", str(parameter_W), "-H", str(parameter_H),
                "-D", str(file_ice + "_stddev"),
                #"-V", str(file_snow + "_image"),
                "-I", str(1)])            
       
    subprocess.run([
            "mpirun", "-np", str(independent_chains),
            "./post_mean_mpi", "-i", 
            "results/ch.dat", "-o", file_cs2_penetration,
            "-x", str(minlon), "-X", str(maxlon),
            "-y", str(minlat), "-Y", str(maxlat),
            "-s", str(skipping),
            "-t", str(thinning),
            "-A", str(parametrization), "-A", str(parametrization),
            "-A", str(parametrization), "-A", str(parametrization),
            "-W", str(parameter_W), "-H", str(parameter_H),
            "-D", str(file_cs2_penetration + "_stddev"),
            #"-V", str(file_cs2_penetration + "_image-"),
            "-I", str(2)])            
       
    subprocess.run([
            "mpirun", "-np", str(independent_chains),
            "./post_mean_mpi", "-i", 
            "results/ch.dat", "-o", file_is2_penetration,
            "-x", str(minlon), "-X", str(maxlon),
            "-y", str(minlat), "-Y", str(maxlat),
            "-s", str(skipping),
            "-t", str(thinning),
            "-A", str(parametrization), "-A", str(parametrization),
            "-A", str(parametrization), "-A", str(parametrization),
            "-W", str(parameter_W), "-H", str(parameter_H),
            "-D", str(file_is2_penetration + "_stddev"),
            #"-V", str(file_is2_penetration + "_image-"),
            "-I", str(3)])            


#if __name__ == "__main__":
def generate_date_table(start_date, end_date, interval_days):
    date_table = []
    current_date = start_date
    sub_table = []

    while current_date <= end_date:
        sub_table.append(current_date.strftime('%Y%m%d'))
        if len(sub_table) == interval_days:
            date_table.append(sub_table)
            sub_table = []
        current_date += datetime.timedelta(days=1)
    if sub_table:
        date_table.append(sub_table)

    return date_table

# choice of the Winter
start_date = datetime.date(2018, 11, 1)    
end_date = datetime.date(2019, 4, 30)
# Number of days of data
interval_days = 15

# Generate the arrays containing all the days
#date = generate_date_table(start_date, end_date, interval_days)

date = [
["20181108", "20181109", "20181110", "20181111", "20181112", "20181113", "20181114", "20181115", "20181116", "20181117", "20181118", "20181119", "20181120", "20181121", "20181122"],
["20181208", "20181209", "20181210", "20181211", "20181212", "20181213", "20181214", "20181215", "20181216", "20181217", "20181218", "20181219", "20181220", "20181221", "20181222"],
["20190108", "20190109", "20190110", "20190111", "20190112", "20190113", "20190114", "20190115", "20190116", "20190117", "20190118", "20190119", "20190120", "20190121", "20190122"],
["20190208", "20190209", "20190210", "20190211", "20190212", "20190213", "20190214", "20190215", "20190216", "20190217", "20190218", "20190219", "20190220", "20190221", "20190222"],
["20190308", "20190309", "20190310", "20190311", "20190312", "20190313", "20190314", "20190315", "20190316", "20190317", "20190318", "20190319", "20190320", "20190321", "20190322"],
["20190408", "20190409", "20190410", "20190411", "20190412", "20190413", "20190414", "20190415", "20190416", "20190417", "20190418", "20190419", "20190420", "20190421", "20190422"]
]
main_path = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/daily_numpys/"
month = [2, 3, 4, 5, 6, 7]

for i in range(len(date)):
    #if i < 2 :
    #    fb_path1  = main_path + "IS2/FB_interp_" + "2018_25km_" + date[i] + ".npy"
    #else : 
    #    fb_path1  = main_path + "IS2/FB_interp_" + "2019_25km_" + date[i] + ".npy"
    #fb_path1  = main_path + "IS2/FB_interp_" + "2019_25km_" + date[i] + ".npy"
    #fb_path2  = main_path + "CS2_LARM/FB_interp_"  + "2018-2019_25km_" + date[i] + ".npy"
    fb_path1 = "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/freeboard_daily_processed/IS2/dailyFB_25km_2018-2019_season.pkl"
    fb_path2 =  "/home/erb/masterproject/MSCI_project/snow_ice/carmen/non_interpolated_data/freeboard_daily_processed/CS2_LARM/dailyFB_25km_2018-2019_season.pkl"
    mois = month[i]
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

    main(fb_path1,
    fb_path2,
    mois,
    date[i],
    verbose,
    minlat,
    maxlat,
    minlon,
    maxlon,
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
    render_observations)