import pandas as pd
from pandas.io.json import json_normalize

import numpy as np

from scipy import interpolate
import geopy.distance

import geopandas as gpd
from shapely.geometry import Point

import datetime

from joblib import Parallel, delayed
import multiprocessing
num_cores = multiprocessing.cpu_count()

# DATA
# Madrid Center Coordinates
center_x = -3.703791298326508
center_y = 40.416798812634816

# Distance to Latitude/Longitude Jump
jump_y = 0.1
coords_1 = (center_y, center_x)
coords_2 = (center_y+jump_y, center_x)

dist_y = geopy.distance.distance(coords_1,coords_2).m
lat_dist_ratio = dist_y/jump_y #meters/lat unit

jump_x = 0.1
coords_1 = (center_y, center_x)
coords_2 = (center_y, center_x+jump_x)

dist_x = geopy.distance.distance(coords_1,coords_2).m
lon_dist_ratio = dist_x/jump_x #meters/lon unit

#print('A jump of 0.1º Lat in Madrid -> {} Meters'.format(lat_dist_ratio/10))
#print('A jump of 0.1º Lon in Madrid -> {} Meters'.format(lon_dist_ratio/10))

# Meters Jump to Latitude/Longitude Jump
dist_y = 12500
dist_x = 12500
lat_jump = dist_y/lat_dist_ratio
lon_jump = dist_x/lon_dist_ratio

# Side of the rectangles (squares if equal)
d_x = 25
d_y = d_x

print('GENERATING SQUARES LATTICE - d_x = {}m'.format(d_x))

#FUNCTIONS
def get_cells() :
    # Summary of the cells to be constructed
    lower_left_vertex = (center_x-lon_jump,center_y-lat_jump)
    div_size_x = d_x/lon_dist_ratio
    div_size_y = d_y/lat_dist_ratio
    n_divs_x = int(2*lon_jump/div_size_x)
    n_divs_y = int(2*lat_jump/div_size_y)
    n_divs_tot = n_divs_x*n_divs_y
    print('There are {} divs in x axis of size {}º Lon'.format(n_divs_x,div_size_x))
    print('There are {} divs in y axis of size {}º Lat'.format(n_divs_y,div_size_y))
    print('There are {} total divisions'.format(n_divs_tot))

    # Obtain all the centers of the potential cells
    st_lon = lower_left_vertex[0]
    st_lat = lower_left_vertex[1]
    cells = {}
    cells_list = []
    for i_x in range(n_divs_x) :
        cells[i_x] = {}
        for i_y in range(n_divs_y) :
            c_x = st_lon+i_x*div_size_x+div_size_x/2;
            c_y = st_lat+i_y*div_size_y+div_size_y/2;

            cells[i_x][i_y] = {
                'cx' : c_x,
                'cy' : c_y
            }
            
            cells_list.append({
                'ix' : i_x,
                'iy' : i_y,
                'cx' : c_x,
                'cy' : c_y
            })
    
    # Turn dictionary into dataframe to ease the plots
    cells_df = pd.DataFrame(cells_list)

    return cells_df

def interpolate_elevations(cells_df,elev) :
    # Original points and values
    x = elev.cx.tolist()
    y = elev.cy.tolist()
    z = elev.elevation.tolist()

    # Interpolate elevations
    f = interpolate.NearestNDInterpolator(list(zip(x,y)),z)
    x_new = cells_df.cx.tolist()
    y_new = cells_df.cy.tolist()

    # Interpolate elevations
    elevations = f(x_new,y_new)
    cells_df['elevation'] = elevations

    return cells_df

def identify_buildings(cells_df,buildings_gdf) :
    n_splits =  num_cores*4

    inputs = list(range(n_splits))
    x_splitted = np.array_split(cells_df.cx.tolist(),n_splits)
    y_splitted = np.array_split(cells_df.cy.tolist(),n_splits)
    
    th = 0.001

    # Function to execute in parallel for each split
    def id_buildings(i,x_splitted,y_splitted,th) :
        buildings_i = []
        x_i = x_splitted[i]
        y_i = y_splitted[i]

        for k in range(len(x_i)) :
            buildings_close = buildings_gdf[
                (buildings_gdf.centroid_x < (x_i[k] + th)) & \
                (buildings_gdf.centroid_x > (x_i[k] - th)) & \
                (buildings_gdf.centroid_y < (y_i[k] + th)) & \
                (buildings_gdf.centroid_y > (y_i[k] - th))]
            is_building = buildings_close.contains(Point(x_i[k],y_i[k])).sum() > 0
            buildings_i.append(is_building)
        
        print('Finished processing split {}'.format(i))

        return buildings_i

    # Execute and gather parallel results
    print('PROCESSING {} SPLITS IN PARALLEL'.format(n_splits))
    results = Parallel(n_jobs=num_cores, verbose=10)(delayed(id_buildings)(i,x_splitted,y_splitted,th) for i in inputs)
    
    buildings = sum(results,[])
    
    cells_df['building'] = buildings
    return cells_df


# MAIN
def main():
    path = 'data/'

    # WE READ ELEVATIONS AND BUILDINGS DATA
    elev = pd.read_csv(path+'elev_25m.csv')
    buildings_gdf = gpd.read_file(path+'buildings_madrid.geojson')

    now = datetime.datetime.now()
    print('\n-------------------------------------------------------------------')
    print('Computing cells... - {}\n'.format(now))

    cells_df = get_cells()

    lapsed_seconds = round((datetime.datetime.now()-now).total_seconds(),3)
    print(cells_df.describe())
    print('\nFinished in {} seconds'.format(lapsed_seconds))
    print('-------------------------------------------------------------------\n\n')

    now = datetime.datetime.now()
    print('\n-------------------------------------------------------------------')
    print('Interpolating cells elevation... - {}\n'.format(now))

    cells_df_elev = interpolate_elevations(cells_df,elev)

    lapsed_seconds = round((datetime.datetime.now()-now).total_seconds(),3)
    print(cells_df_elev.describe())
    print('\nFinished in {} seconds'.format(lapsed_seconds))
    print('-------------------------------------------------------------------\n\n')

    now = datetime.datetime.now()
    print('\n-------------------------------------------------------------------')
    print('Identifying cells as buildings or free space... - {}\n'.format(now))

    cells_df_elev_building = identify_buildings(cells_df_elev,buildings_gdf)

    lapsed_seconds = round((datetime.datetime.now()-now).total_seconds(),3)
    print(cells_df_elev_building.describe())
    print('\nFinished in {} seconds'.format(lapsed_seconds))
    print('-------------------------------------------------------------------\n\n')

    now = datetime.datetime.now()
    print('\n-------------------------------------------------------------------')
    print('Saving cells_df to {}... - {}\n'.format(path+'squares_df_{}m_elev_buildings.csv'.format(d_x),now))

    cells_df_elev_building.to_csv(path+'squares_df_{}m_elev_buildings.csv'.format(d_x), index=False, header=True)

    lapsed_seconds = round((datetime.datetime.now()-now).total_seconds(),3)
    print('\nFinished in {} seconds'.format(lapsed_seconds))
    print('-------------------------------------------------------------------\n\n')

if __name__== "__main__":
    main()
